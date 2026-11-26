import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/trainingTelemetry.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'trainingTelemetry.ts', { loader: 'ts' });
const { continuousTrainingEvents, isMidTrainingValidationEvent, trainingActivity, trainingLoss } = await import(
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

test('checkpoint resumes retain earlier steps and replace a rolled-back suffix', () => {
  const reading = (step, attempt, ts) => ({ ...event('training', { loss: .4 }, ts), current_step: step, attempt_id: attempt });
  const first = reading(100, 'first', '01');
  const checkpoint = reading(552, 'first', '02');
  const abandoned = reading(600, 'first', '03');
  const resumed = reading(560, 'resume', '04');
  const validation = { ...event('generate:member', { benchmark_name: 'CareQA' }, '05'), attempt_id: 'validation' };
  assert.deepEqual(continuousTrainingEvents([resumed, first, validation, checkpoint, abandoned]), [first, checkpoint, resumed]);
  const restart = reading(1, 'restart', '06');
  assert.deepEqual(continuousTrainingEvents([first, checkpoint, resumed, restart]), [restart]);
});


test('20,000 prepared rows cannot enlarge a 1,250-step training chart or become metrics', () => {
  const preparation = { ...event('data_preparation', { elapsed_seconds: 24.6 }, '01'),
    current_step: 20000, total_steps: 20000 };
  const training = { ...event('training', { loss: .5 }, '02'),
    current_step: 20, total_steps: 1250, attempt_id: 'first' };
  const validationLoss = { ...event('training', { eval_loss: .4 }, '03'),
    current_step: 20, total_steps: 1250, attempt_id: 'first' };
  const points = continuousTrainingEvents([preparation, training, validationLoss]);
  assert.deepEqual(points, [training, validationLoss]);
  assert.equal(Math.max(...points.map(p => p.total_steps)), 1250);
  assert.deepEqual(continuousTrainingEvents([preparation]), []);
});

test('preparation in a new attempt does not distort the old training axis', () => {
  const training = { ...event('training', { loss: .5 }, '01'),
    current_step: 1250, total_steps: 1250, attempt_id: 'first' };
  const preparation = { ...event('data_preparation', { elapsed_seconds: 25 }, '02'),
    current_step: 20000, total_steps: 20000, attempt_id: 'second' };
  assert.deepEqual(continuousTrainingEvents([training, preparation]), [training]);
});


test('RFT generation and selection cannot enlarge the optimizer-step axis', () => {
  const generation = { ...event('generation', { acceptance_rate: .46 }, '01'),
    current_step: 24000, total_steps: 24000 };
  const selection = { ...event('selection', { phase_detail: 'rendering' }, '02'),
    current_step: 18002, total_steps: 18002 };
  const training = { ...event('training', { loss: .0833 }, '03'),
    current_step: 280, total_steps: 282, attempt_id: 'supervised' };
  const saving = { ...event('saving', {}, '04'), total_steps: 24000 };
  const points = continuousTrainingEvents([generation, selection, training, saving]);
  assert.deepEqual(points, [training]);
  assert.equal(Math.max(...points.map(p => p.total_steps)), 282);
  assert.deepEqual(continuousTrainingEvents([generation, selection]), []);
});

test('training phases retain reward and KL metrics, while unknown phases cannot set the axis', () => {
  for (const phase of ['train', 'training', 'adaptive_training', 'training:optimizer']) {
    const training = event(phase, { reward: .7, kl: .02 });
    const unknown = { ...event('new_data_stage', { reward: 1 }, '2026-10-26T09:01:00Z'),
      total_steps: 1000000 };
    assert.deepEqual(continuousTrainingEvents([training, unknown]), [training]);
  }
});


test('signed RL loss survives while the legacy missing-value sentinel does not', () => {
  for (const loss of [-.007754, -.001236, 0, .000944, -1]) {
    assert.equal(trainingLoss(event('train', { loss })), loss);
  }
  assert.equal(trainingLoss(event('train')), undefined);
  assert.equal(trainingLoss({ ...event('train'), loss: -.005 }), -.005);
  assert.equal(trainingLoss(event('train', { eval_loss: -.2 })), undefined);
  assert.equal(trainingLoss(event('train', { loss: NaN })), undefined);
});

test('negative-loss checkpoint resumes replace the abandoned suffix', () => {
  const reading = (step, attempt, ts) => ({ ...event('train', { loss: -.004 }, ts),
    current_step: step, attempt_id: attempt });
  const first = reading(20, 'first', '01');
  const checkpoint = reading(40, 'first', '02');
  const abandoned = reading(80, 'first', '03');
  const resumed = reading(60, 'resume', '04');
  assert.deepEqual(continuousTrainingEvents([first, checkpoint, abandoned, resumed]),
    [first, checkpoint, resumed]);
  const restart = reading(1, 'restart', '05');
  assert.deepEqual(continuousTrainingEvents([first, checkpoint, resumed, restart]), [restart]);
});
