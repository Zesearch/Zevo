import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';

const source = readFileSync(new URL('../src/lib/launchStepStatus.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'launchStepStatus.ts', { loader: 'ts' });
const { trainingStepReady, runtimeStepReady } = await import(
  `data:text/javascript;base64,${Buffer.from(code).toString('base64')}`,
);

const training = { dataset: '', dataQuery: '', baseModel: '', modelQuery: '', trainingMethod: '',
  methodQuery: '', teacherModel: '', rewardModel: '', usePeft: '' };
const runtime = { numGpus: '', iterations: '', budget: '', timeLimitHours: '', queueWaitHours: '',
  stopThreshold: '', gpuAllocationMode: 'per_run', generation_backend: '', adaptiveMidTrainingEvaluation: false };

test('optional steps can proceed using defaults without artificial values', () => {
  assert.equal(trainingStepReady(training), true);
  assert.equal(runtimeStepReady(runtime), true);
});

test('method-specific model requirements block review until supplied', () => {
  assert.equal(trainingStepReady({ ...training, trainingMethod: 'gkd' }), false);
  assert.equal(trainingStepReady({ ...training, trainingMethod: 'gkd', teacherModel: 'owner/teacher' }), true);
  assert.equal(trainingStepReady({ ...training, trainingMethod: 'online_dpo', rewardModel: ' ' }), false);
  assert.equal(trainingStepReady({ ...training, trainingMethod: 'online_dpo', rewardModel: 'owner/reward' }), true);
});

test('invalid runtime limits cannot be treated as a ready optional step', () => {
  for (const field of ['numGpus', 'iterations', 'budget', 'timeLimitHours', 'queueWaitHours']) {
    for (const value of ['-1', 'NaN', 'Infinity']) {
      assert.equal(runtimeStepReady({ ...runtime, [field]: value }), false, `${field}=${value}`);
    }
  }
  assert.equal(runtimeStepReady({ ...runtime, queueWaitHours: '168' }), true);
  assert.equal(runtimeStepReady({ ...runtime, queueWaitHours: '169' }), false);
});

test('zero limits and negative metric thresholds retain their intended semantics', () => {
  assert.equal(runtimeStepReady({ ...runtime, iterations: '0', budget: '0', stopThreshold: '-12.5' }), true);
  assert.equal(runtimeStepReady({ ...runtime, stopThreshold: 'Infinity' }), false);
});
