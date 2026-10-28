import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/executionProgress.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'execution.ts', { loader: 'ts' });
const { executionPhase, phaseLabel, inferencePreparation, latestBenchmarkEvent, activationExecution, benchmarkCounts, latestExecution } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const hb = (id, hour, phase) => ({id, started_at: `2026-10-26T${hour}:00:00Z`, activation_phase: phase});
const event = (hour, heartbeat, execution, name, complete = false) => ({ts: `2026-10-26T${hour}:10:00Z`, heartbeat_id: heartbeat,
  phase: 'generate', extras: {execution_id: execution, benchmark_name: name, benchmark_total: 3, benchmark_complete: complete}});

test('old activation cannot display later repair counters', () => {
  const activations = [hb('a', '10', 'submit'), hb('b', '11', 'repair')];
  const events = [event('10', 'a', 'first', 'A', true), event('11', 'b', 'second', 'A', true),
    event('11', 'b', 'second', 'B', true), event('11', 'b', 'second', 'C', true)];
  assert.deepEqual(benchmarkCounts(activationExecution(events, activations[0], activations)), {completed: 1, total: 3});
  assert.deepEqual(benchmarkCounts(activationExecution(events, activations[1], activations)), {completed: 3, total: 3});
});

test('repair without a workload is empty, collect and continue retain the execution', () => {
  const events = [event('10', 'a', 'first', 'A', true)];
  assert.deepEqual(activationExecution(events, hb('b', '11', 'repair')), []);
  for (const phase of ['collect', 'continue', 'resume']) {
    assert.deepEqual(benchmarkCounts(activationExecution(events, hb('b', '11', phase))), {completed: 1, total: 3});
  }
});

test('new execution starts at zero; replay and replicas do not double count', () => {
  const first = event('10', 'a', 'first', 'A', true);
  const fresh = event('11', 'b', 'second', 'A');
  assert.deepEqual(benchmarkCounts(latestExecution([first, fresh])), {completed: 0, total: 3});
  const replay = {...first, heartbeat_id: 'b'};
  assert.deepEqual(benchmarkCounts(activationExecution([first, replay], hb('b', '11', 'collect'))), {completed: 1, total: 3});
  assert.deepEqual(benchmarkCounts([first, first], ['A']), {completed: 1, total: 1});
});

test('evaluation shares deduplication but counts scoring completion markers', () => {
  const events = [{phase: 'scoring 1/2: A', extras: {}},
    {phase: 'scoring', extras: {phase: 'benchmark_complete', benchmark_id: 'validation:0'}}];
  assert.deepEqual(benchmarkCounts(events, undefined, "evaluation"), {completed: 1, total: 2});
  assert.deepEqual(benchmarkCounts([{phase: 'scoring', extras: {}}], undefined, "evaluation"), {completed: 0, total: 1});
});

 test('mid-training scoring cannot count inference completion as scored benchmarks', () => {
  const generated = event('10', 'a', 'train', 'A', true);
  assert.equal(benchmarkCounts([generated], undefined, "evaluation"), null);
  const scoring = {phase: 'scoring 1/1: A', extras: {benchmark_total: 1}};
  assert.deepEqual(benchmarkCounts([generated, scoring], undefined, "evaluation"), {completed: 0, total: 1});
});

test('benchmark name survives output writing and completion events', () => {
  const generation = event('10', 'a', 'first', 'val-omni', true);
  const output = {...generation, phase: 'write_outputs', extras: {execution_id: 'first'}};
  assert.equal(latestBenchmarkEvent([generation, output]), generation);
  assert.equal(latestBenchmarkEvent([output]), undefined);
});

test('verified collect fills the originating execution without leaking a retry', () => {
  const activations = [hb('a', '10', 'submit'), {...hb('b', '11', 'repair'), execution_phase: 'collect'}];
  const events = [event('10', 'a', 'first', 'A', true), event('11', 'a', 'first', 'B', true),
    event('11', 'a', 'first', 'C', true), event('12', 'c', 'second', 'D', true)];
  assert.deepEqual(benchmarkCounts(activationExecution(events, activations[0], activations)), {completed: 3, total: 3});
  assert.deepEqual(benchmarkCounts(activationExecution(events.slice(0, 3), activations[1], activations)), {completed: 3, total: 3});
});


test('preparation reports phases without fabricating generation progress', () => {
  const loading = {...event('10', 'a', 'first', ''), phase: 'load_model'};
  assert.equal(inferencePreparation([loading]).label, 'Loading model');
  const calibration = {...loading, ts: '2026-10-26T10:15:00Z', phase: 'length_calibration'};
  assert.deepEqual(inferencePreparation([loading, calibration]), {label: 'Calibrating generation length', startedAt: calibration.ts});
  assert.equal(inferencePreparation([loading, event('11', 'a', 'first', 'A')]), null);
});


test('same-second preparation follows stdout sequence regardless of API row order', () => {
  const setup = {...event('10', 'a', 'first', ''), phase: 'setup'};
  const loading = {...setup, phase: 'load_model'};
  for (const events of [[setup, loading], [loading, setup]]) {
    assert.equal(inferencePreparation(events).label, 'Loading model');
  }
  setup.extras = {...setup.extras, telemetry_sequence: 1};
  loading.extras = {...loading.extras, telemetry_sequence: 2};
  assert.equal(inferencePreparation([loading, setup]).label, 'Loading model');
  // A later explicit setup marker is authoritative, even with a tied timestamp.
  setup.extras.telemetry_sequence = 3;
  assert.equal(inferencePreparation([setup, loading]).label, 'Checking environment');
});


test('runtime preparation and library startup milestones have distinct labels', () => {
  for (const [phase, label] of [['prepare_runtime', 'Preparing runtime cache'],
    ['resolve_model', 'Resolving model architecture'], ['initialize_engine', 'Starting inference engine'],
    ['load_weights', 'Loading model weights'], ['compile_model', 'Compiling inference kernels']]) {
    assert.equal(inferencePreparation([{...event('10', 'a', 'first', ''), phase}]).label, label);
  }
});


test('Data and Train share phase labels without inventing benchmark progress', () => {
  for (const [agent, phase, label] of [
    ['data', 'generate', 'Generating training data'],
    ['data', 'validating_data_artifacts', 'Validating data'],
    ['train', 'loading_model', 'Loading model'],
    ['train', 'loading_checkpoint', 'Restoring checkpoint'],
    ['train', 'saving_model', 'Saving model'],
    ['evaluation', 'loading_predictions', 'Loading predictions'],
  ]) {
    const result = executionPhase([{...event('10', 'a', 'first', ''), event_type: 'phase', phase}], agent);
    assert.equal(result.label, label);
    assert.equal(result.startedAt, '2026-10-26T10:10:00Z');
    assert.equal(result.total, undefined);
  }
  assert.equal(phaseLabel('deduplicate_records', 'data'), 'Deduplicate records');
});

test('repeated phase timers start at the current occurrence and terminal phases disappear', () => {
  const events = ['training', 'saving_checkpoint', 'training', 'training'].map((phase, index) =>
    ({...event(String(10 + index), 'a', 'first', ''), event_type: 'phase', phase}));
  assert.equal(executionPhase(events, 'train').startedAt, events[2].ts);
  assert.equal(executionPhase([...events, {...events[3], ts: '2026-10-26T14:00:00Z', phase: 'completed'}], 'train'), null);
  assert.equal(executionPhase([], 'data'), null);
});

test('evaluation scoring retains its benchmark and judge card instead of a duplicate phase card', () => {
  const scoring = {...event('10', 'a', 'first', 'A'), event_type: 'phase', phase: 'scoring 1/2: A'};
  assert.equal(executionPhase([scoring], 'evaluation'), null);
});


test('inference exposes output saving after generation without resurrecting completed preparation', () => {
  const generate = {...event('10', 'a', 'first', 'A'), event_type: 'phase'};
  const saving = {...event('11', 'a', 'first', ''), event_type: 'phase', phase: 'write_outputs'};
  assert.equal(executionPhase([generate, saving], 'inference').label, 'Saving outputs');
  assert.equal(executionPhase([generate, saving, {...saving, ts: '2026-10-26T12:00:00Z', phase: 'done'}], 'inference'), null);
});
