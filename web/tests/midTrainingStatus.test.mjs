import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const telemetry = readFileSync(new URL('../src/lib/trainingTelemetry.ts', import.meta.url), 'utf8');
const source = readFileSync(new URL('../src/lib/midTrainingStatus.ts', import.meta.url), 'utf8')
  .replace('import { trainingActivity, isMidTrainingValidationEvent } from "./trainingTelemetry";', '');
const { code } = await transformWithEsbuild(`${telemetry}\n${source}`, 'status.ts', { loader: 'ts' });
const { midTrainingExecution, validationStatus, decisionStatus, inferenceProgress, midTrainingHasStarted } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const event = (ts, phase, extras = {}) => ({ ts, phase, event_type: 'progress', loss: -1,
  current_step: 0, total_steps: 100, extras: { execution_purpose: 'workload', ...extras } });
const ticket = (events, validation = {}) => ({ status: 'running', execution_events: events, mid_training_validation: validation });

test('resume discards previous checkpoint counters and ignores unverified events', () => {
  const old = event('01', 'generate:old', { benchmark_name: 'A', suite_rows_completed: 100, suite_rows_total: 100 });
  const training = event('02', 'train');
  const fresh = event('03', 'generate:new', { benchmark_name: 'A', suite_rows_completed: 5, suite_rows_total: 100 });
  const unverified = event('04', 'training', { execution_purpose: 'unverified' });
  const execution = midTrainingExecution(ticket([unverified, fresh, old, training]));
  assert.deepEqual(execution.events, [fresh]);
  assert.equal(execution.phase, 'inference');
  assert.equal(inferenceProgress(execution.events).completed, 5);
});

test('validation badges describe validation instead of copying Train status', () => {
  const infer = ticket([event('01', 'generate:a', { benchmark_name: 'A' })]);
  assert.equal(validationStatus(infer, false), 'running');
  assert.equal(validationStatus(infer, true), 'paused');
  assert.equal(validationStatus({ ...infer, status: 'failed' }, false), 'failed');
  assert.equal(validationStatus(ticket([event('02', 'train')]), false), 'queued');
  assert.equal(validationStatus(ticket([event('03', 'adaptive_validation')], { awaiting_decision: true }), false), 'succeeded');
  assert.equal(validationStatus(ticket([], { complete: true }), false), 'succeeded');
});

test('decision badges separate live, failed and accepted checkpoint activations', () => {
  const accepted = { exit_code: 0, is_live: false, mid_training: { action: 'continue' } };
  assert.equal(decisionStatus([], false), 'queued');
  assert.equal(decisionStatus([accepted], false), 'succeeded');
  assert.equal(decisionStatus([accepted], false, true), 'queued');
  assert.equal(decisionStatus([accepted, { exit_code: 1, is_live: false }], false), 'failed');
  assert.equal(decisionStatus([{ is_live: true }], false), 'running');
  assert.equal(decisionStatus([{ is_live: true }], true), 'paused');
});

test('scoring selection keeps only the current checkpoint events', () => {
  const earlierJudge = event('01', 'scoring', { phase: 'model_judge' });
  const train = event('02', 'training');
  const currentJudge = event('03', 'scoring', { phase: 'model_judge' });
  const execution = midTrainingExecution(ticket([earlierJudge, train, currentJudge]));
  assert.equal(execution.phase, 'scoring');
  assert.deepEqual(execution.events, [currentJudge]);
});


test('validation branch appears only on first real validation and survives resume', () => {
  assert.equal(midTrainingHasStarted(undefined), false);
  assert.equal(midTrainingHasStarted(ticket([])), false);
  assert.equal(midTrainingHasStarted(ticket([event('01', 'train')])), false);
  assert.equal(midTrainingHasStarted(ticket([event('02', 'generate:A', {
    benchmark_name: 'A', execution_purpose: 'preflight' })])), false);
  const validation = event('03', 'generate:A', { benchmark_name: 'A' });
  assert.equal(midTrainingHasStarted(ticket([validation])), true);
  assert.equal(midTrainingHasStarted(ticket([validation, event('04', 'train')])), true);
  assert.equal(midTrainingHasStarted(ticket([], { measurements: [{ step: 100 }] })), true);
  assert.equal(midTrainingHasStarted(undefined, 1), true);
});
