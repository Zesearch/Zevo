import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/trainingTelemetry.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'trainingTelemetry.ts', { loader: 'ts' });
const { isMidTrainingValidationEvent, trainingActivity } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`,
);

const event = (phase, extras = {}, ts = '2026-10-26T09:00:00Z') => ({
  event_type: 'progress', phase, current_step: 100, total_steps: 2757, loss: -1,
  ts, extras: { execution_purpose: 'workload', ...extras },
});

test('validation row counters and attempts cannot select the training curve or its axis', () => {
  const training = event('training', { loss: .4, 'rewards/chosen': 2 });
  const inference = { ...event('generate:member', { benchmark_index: 1, benchmark_total: 3 }),
    attempt_id: 'validation', current_step: 5604, total_steps: 5604 };
  const decision = event('adaptive_validation', { adaptive_request: { step: 552 } });
  assert.deepEqual([training, inference, decision].filter(e => !isMidTrainingValidationEvent(e)), [training]);
  assert.equal(isMidTrainingValidationEvent(event('training', { eval_loss: .2 })), false);
  assert.equal(isMidTrainingValidationEvent(event('rollout', { reward: 1, kl: .1 })), false);
});

test('actual workload activity distinguishes inference, scoring and decision waiting', () => {
  const inference = event('generate:member', { benchmark_name: 'CareQA' });
  const scoring = event('scoring 1/3: CareQA', {}, '2026-10-26T09:01:00Z');
  assert.equal(trainingActivity([inference], null, 'running'), 'Mid-training Validation · Inference');
  assert.equal(trainingActivity([scoring, inference], null, 'running'), 'Mid-training Validation · Scoring');
  assert.equal(trainingActivity([scoring], { awaiting_decision: true }, 'running'), 'Awaiting training decision');
  assert.equal(trainingActivity([scoring], { awaiting_decision: true }, 'running', true), 'Training paused');
  assert.equal(trainingActivity([scoring], { awaiting_decision: true }, 'failed'), null);
  assert.equal(trainingActivity([event('training', { loss: .4 })], null, 'running'), 'Training');
});

test('unverified logs cannot replace the current workload status', () => {
  const training = event('training', { loss: .4 });
  const stale = event('scoring', { execution_purpose: 'unverified' }, '2026-10-26T09:02:00Z');
  assert.equal(trainingActivity([training, stale], null, 'running'), 'Training');
});
