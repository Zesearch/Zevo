import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/lib/heartbeatGroups.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { heartbeatGroups, settledHeartbeat, heartbeatGroupFor, heartbeatStatus } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const hb = (id, phase = '', exit = 0, ticket = 'orchestrator') => ({
  id, ticket_id: ticket, started_at: id, activation_phase: phase,
  exit_code: exit, is_live: exit === -1, action: '', child_ticket_id: '',
});

test('repair chains share a timeline row while later supervisor decisions stay separate', () => {
  const first = hb('1', '', 1);
  const repair = { ...hb('2', 'repair'), action: 'emit_ticket', child_ticket_id: 'infra-1' };
  const groups = heartbeatGroups([first, repair, hb('3'), hb('4', 'repair', 1), hb('5', 'repair', -1)]);
  assert.deepEqual(groups.map(group => group.map(h => h.id)), [['1', '2'], ['3', '4', '5']]);
  const settled = settledHeartbeat(groups[0]);
  assert.equal(settled.id, first.id);
  assert.equal(settled.started_at, first.started_at);
  assert.equal(settled.exit_code, 0);
  assert.equal(settled.child_ticket_id, 'infra-1');
  assert.equal(settledHeartbeat(groups[1]).is_live, true);
  assert.equal(first.exit_code, 1);
});

test('unrecovered failures and orphan repairs remain visible', () => {
  const groups = heartbeatGroups([hb('1', 'repair', 1), hb('2', 'repair', 1, 'other')]);
  assert.equal(groups.length, 2);
  assert.equal(settledHeartbeat(groups[0]).exit_code, 1);
});


test('checkpoint repair activations never merge into an unrelated supervisor action', () => {
  const ordinary = hb('1');
  const checkpoint = { ...hb('2', 'repair', 1), mid_training: { request_id: 'checkpoint-1' } };
  const retry = { ...hb('3', 'repair'), mid_training: { request_id: 'checkpoint-1' } };
  const next = { ...hb('4', 'repair'), mid_training: { request_id: 'checkpoint-2' } };
  assert.deepEqual(heartbeatGroups([ordinary, checkpoint, retry, next]).map(group => group.map(h => h.id)),
    [['1'], ['2', '3'], ['4']]);
});

test('collect repair stays separate from submit while same-phase retries group', () => {
  const groups = heartbeatGroups([{...hb('1', 'submit'), execution_phase: 'submit'},
    {...hb('2', 'repair'), execution_phase: 'collect'}, {...hb('3', 'repair'), execution_phase: 'collect'}]);
  assert.deepEqual(groups.map(g => g.map(h => h.id)), [['1'], ['2', '3']]);
});

test('Start Inference opens its own activation when instruction wakes are hidden', () => {
  const instruction = { ...hb('instruction', 'instruction'), child_ticket_id: 'train-001' };
  const midEvaluation = { ...hb('mid-evaluation'), mid_training: { request_id: 'checkpoint-11817' },
    action: 'emit_ticket', child_ticket_id: 'mid-eval-11817' };
  const candidateInference = { ...hb('candidate-inference'), action: 'emit_ticket', child_ticket_id: 'infer-002' };
  const full = [hb('baseline'), instruction, midEvaluation, candidateInference];
  const visible = full.filter(h => h.activation_phase !== 'instruction');
  const clicked = visible.at(-1);
  // The old positional lookup opened the preceding mid-training evaluation.
  assert.equal(full[visible.indexOf(clicked)].id, midEvaluation.id);
  assert.equal(heartbeatGroupFor(full, clicked.id).at(-1).child_ticket_id, 'infer-002');
  assert.equal(heartbeatGroupFor(full.slice(1), clicked.id).at(-1).id, clicked.id);
  assert.equal(heartbeatGroupFor(full, 'not-yet-fetched'), undefined);
});

test('selection keeps repair history together without merging later decisions', () => {
  const failed = hb('failed', '', 1);
  const retry = hb('retry', 'repair');
  const later = hb('later', '', -1);
  const history = [hb('instruction', 'instruction'), failed, retry, later];
  assert.deepEqual(heartbeatGroupFor(history, failed.id), [failed, retry]);
  assert.deepEqual(heartbeatGroupFor(history, retry.id), [failed, retry]);
  assert.equal(heartbeatStatus(failed), 'failed');
  assert.equal(heartbeatStatus(heartbeatGroupFor(history, failed.id).at(-1)), 'succeeded');
  assert.equal(heartbeatStatus(later), 'running');
  assert.equal(heartbeatStatus(hb('repairing', 'repair', -1)), 'repairing');
  assert.equal(heartbeatStatus({...hb('stop'), action: 'mark_failed'}), 'failed');
});
