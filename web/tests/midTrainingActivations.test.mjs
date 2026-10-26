import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/midTrainingActivations.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'midTrainingActivations.ts', { loader: 'ts' });
const { isMidTrainingActivation, midTrainingActivationLabel } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const first = { id: 'first', action: 'wait', exit_code: 1, mid_training: { request_id: 'r1', train_ticket_id: 'train', step: 552, action: '' } };
const second = { ...first, id: 'second', exit_code: 0, mid_training: { ...first.mid_training, action: 'continue', next_step: 1104 } };
test('ordinary scheduling and pipeline actions stay on the main timeline', () => {
  assert.equal(isMidTrainingActivation(first), true);
  assert.equal(isMidTrainingActivation({ ...first, mid_training: null }), false);
  for (const action of ['emit_ticket', 'mark_done', 'mark_failed']) assert.equal(isMidTrainingActivation({ ...first, action }), false);
  assert.equal(isMidTrainingActivation({ ...first, activation_phase: 'instruction' }), false);
});
test('each activation retains its own decision and checkpoint retries remain distinct', () => {
  assert.equal(midTrainingActivationLabel(first, [first, second]), 'Step 552 · Failed');
  assert.equal(midTrainingActivationLabel(second, [first, second]), 'Step 552 · Retry 2 · Continue → 1,104');
  assert.equal(midTrainingActivationLabel({ ...first, is_live: true }, [first]), 'Step 552 · Deciding…');
  const next = { ...second, id: 'third', mid_training: { ...second.mid_training, step: 1104, action: 'stop' } };
  assert.equal(midTrainingActivationLabel(next, [first, second, next]), 'Step 1,104 · Stop');
});
