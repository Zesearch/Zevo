import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/lib/heartbeatGroups.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { heartbeatGroups, settledHeartbeat } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
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
