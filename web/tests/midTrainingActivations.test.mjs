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
  assert.equal(isMidTrainingActivation({ ...first, action: 'emit_ticket', child_ticket_id: 'mid-infer-r1' }), true);
  assert.equal(isMidTrainingActivation({ ...first, mid_training: null, action: 'emit_ticket' }), false);
  assert.equal(isMidTrainingActivation({ ...first, activation_phase: 'instruction' }), false);
});
test('each activation retains its own decision and checkpoint retries remain distinct', () => {
  assert.equal(midTrainingActivationLabel(first, [first, second]), 'Step 552 · Failed');
  assert.equal(midTrainingActivationLabel(second, [first, second]), 'Step 552 · Retry 2 · Continue → 1,104');
  assert.equal(midTrainingActivationLabel({ ...first, is_live: true }, [first]), 'Step 552 · Deciding…');
  const next = { ...second, id: 'third', mid_training: { ...second.mid_training, step: 1104, action: 'stop' } };
  assert.equal(midTrainingActivationLabel(next, [first, second, next]), 'Step 1,104 · Stop');
});

test('mid-training branch names the concrete action instead of the agent', async () => {
  const { midTrainingActionLabel: label } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  assert.equal(label({...first, action:'emit_ticket', mid_training:{...first.mid_training, phase:'inference'}}), 'Start Inference');
  assert.equal(label({...first, action:'emit_ticket', mid_training:{...first.mid_training, phase:'evaluation'}}), 'Start Evaluation');
  assert.equal(label(second), 'Resume training');
  assert.equal(label({...second,mid_training:{...second.mid_training,action:'stop'}}), 'Stop training');
  assert.equal(label({...first, action:'mark_failed'}), 'Stop Run');
  assert.equal(label({...first,action:'',is_live:true}), 'Decide next training step…');
});

test('handoff failures are reviews, not validation decisions', () => {
  const failed = { ...first, exit_code: 0, mid_training: { ...first.mid_training, phase: 'handoff_failed' } };
  assert.equal(midTrainingActivationLabel(failed, [failed]), 'Step 552 · Review handoff · Repair');
});

const { checkpointOrchestrationView: view } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const old = { ...second, started_at: '2026-11-10T09:00:00Z', mid_training: { ...second.mid_training, request_id: '735', step: 735 } };
const startup = { request_id: '1468', train_ticket_id: 'train', step: 1468, phase: 'inference',
  wakeup_id: 'failed', status: 'failed', created_at: '2026-11-10T10:00:00Z', heartbeat_id: '', error: 'Missing input' };

test('a new checkpoint startup failure never selects the previous checkpoint success', () => {
  const current = view([old], [startup]);
  assert.equal(current.requestId, '1468');
  assert.deepEqual(current.activations, []);
  assert.equal(current.pendingAttempt.error, 'Missing input');
  const previous = view([old], [startup], '735');
  assert.deepEqual(previous.activations, [old]);
  assert.equal(previous.pendingAttempt, undefined);
});

test('retry success selects its activation and retains the earlier startup failure', () => {
  const fresh = { ...old, id: 'new-hb', started_at: '2026-11-10T11:00:01Z',
    mid_training: { ...old.mid_training, request_id: '1468', step: 1468 } };
  const retry = { ...startup, wakeup_id: 'retry', status: 'completed', heartbeat_id: 'new-hb', created_at: '2026-11-10T11:00:00Z' };
  const current = view([old, fresh], [startup, retry]);
  assert.deepEqual(current.activations, [fresh]);
  assert.equal(current.pendingAttempt, undefined);
  assert.equal(current.attempts[0].status, 'failed');
});

test('a later failed phase at the same checkpoint does not display its earlier success', () => {
  const fresh = { ...old, started_at: '2026-11-10T09:30:00Z', mid_training: { ...old.mid_training, request_id: '1468', step: 1468 } };
  assert.equal(view([fresh], [startup]).pendingAttempt.wakeup_id, 'failed');
  assert.equal(view([old], [{ ...startup, status: 'queued', error: '' }]).pendingAttempt.status, 'queued');
});

test('a new recorded heartbeat must load before replacing its loading state with a transcript', () => {
  const phaseOne = { ...old, mid_training: { ...old.mid_training, request_id: '1468', step: 1468 } };
  const phaseTwo = { ...startup, status: 'running', heartbeat_id: 'phase-two', phase: 'evaluation' };
  const current = view([phaseOne], [phaseTwo]);
  assert.equal(current.defaultHeartbeatId, 'phase-two');
  assert.equal(current.awaitingHeartbeat, true);
  const fresh = { ...phaseOne, id: 'phase-two', started_at: '2026-11-10T10:00:01Z' };
  assert.equal(view([phaseOne, fresh], [phaseTwo]).awaitingHeartbeat, false);
});

test('legacy records without an activation link do not become fabricated startup states', () => {
  const legacy = { ...startup, status: 'completed', activation_link_known: false };
  assert.equal(view([old], [legacy]).pendingAttempt, undefined);
});
