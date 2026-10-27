import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/executionProgress.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'execution.ts', { loader: 'ts' });
const { latestBenchmarkEvent, activationExecution, benchmarkCounts, latestExecution } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
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
