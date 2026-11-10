import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

function moduleUrl(name) {
  let source = readFileSync(new URL(`../src/lib/${name}.ts`, import.meta.url), 'utf8');
  for (const dependency of ['stageStatus', 'timelineStatus']) {
    const specifier = `"./${dependency}"`;
    if (source.includes(specifier)) source = source.replace(specifier, JSON.stringify(moduleUrl(dependency)));
  }
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  });
  return `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`;
}
const { pipelineState, pipelineTicketStatuses, stationStateFromTickets, PIPELINE_LAMPS } = await import(moduleUrl('pipelineStatus'));
const { timelineTicketStatus } = await import(moduleUrl('timelineStatus'));
const train = { id: 'train', agent_id: 'train', lane: 'optimization', iteration: 2,
  operation: 'train', model_source: 'checkpoint', test_set_name: '', status: 'waiting_external',
  created_at: '2026-11-09T18:00:00Z', updated_at: '2026-11-09T18:01:00Z' };
const resource = { ticket_id: train.id, created_at: '2026-11-09T19:00:00Z',
  meta: { resource_request: true, scheduler_state: 'PENDING' } };

function both(tickets, resources = [], heartbeats = [], instructions = []) {
  const statuses = pipelineTicketStatuses(tickets, resources, heartbeats, instructions);
  const visible = tickets.map(t => ({ ...t, status: statuses.get(t.id) }));
  const timeline = pipelineState(timelineTicketStatus(visible));
  const loop = stationStateFromTickets(visible);
  assert.equal(timeline, loop);
  return timeline;
}

test('iteration and loop use the same five colors, including waiting and repair', () => {
  assert.equal(new Set(Object.values(PIPELINE_LAMPS).map(l => l.color)).size, 5);
  for (const status of ['queued', 'awaiting_input', 'waiting_external', 'paused', 'needs_input']) {
    assert.equal(both([{ ...train, status }]), 'waiting', status);
  }
  for (const status of ['running', 'repairing']) assert.equal(both([{ ...train, status }]), 'active');
  for (const status of ['succeeded', 'skipped']) assert.equal(both([{ ...train, status }]), 'done');
  for (const status of ['failed', 'degraded']) assert.equal(both([{ ...train, status }]), 'failed');
  assert.equal(both([{ ...train, status: 'cancelled' }]), 'idle');
  assert.equal(both([]), 'idle');
});

test('remote queued/running transitions apply to both displays after the submit Agent exits', () => {
  assert.equal(both([train], [resource]), 'waiting');
  const running = { ...resource, meta: { ...resource.meta, scheduler_state: 'RUNNING+' } };
  assert.equal(both([train], [running]), 'active');
  const finished = [{ ...train, status: 'succeeded' }];
  assert.equal(both(finished, [running]), 'done');
});

test('newest resource attempt wins regardless of response order, ignoring superseded requests', () => {
  const old = { ...resource, created_at: train.created_at, meta: { ...resource.meta, scheduler_state: 'RUNNING' } };
  const superseded = { ...old, created_at: '2026-11-09T20:00:00Z',
    meta: { ...old.meta, superseded_by_instruction_id: 'instruction' } };
  for (const resources of [[old, resource, superseded], [superseded, resource, old]]) {
    assert.equal(both([train], resources), 'waiting');
  }
});

test('pause affects the same tickets in both displays without hiding a live remote worker', () => {
  const instruction = { status: 'needs_input', target_ticket_id: 'train' };
  assert.equal(both([{ ...train, status: 'running' }], [], [], [instruction]), 'waiting');
  const running = { ...resource, meta: { ...resource.meta, scheduler_state: 'RUNNING' } };
  assert.equal(both([train], [running], [], [instruction]), 'active');
  assert.equal(both([train], [resource], [], [instruction]), 'waiting');
  assert.equal(both([{ ...train, status: 'running', lane: 'held_out_test' }], [], [],
    [{ ...instruction, target_ticket_id: null }]), 'active');
});

test('live repair remains running in the five-color display while preserving its detailed state', () => {
  const ticket = { ...train, status: 'running' };
  const heartbeat = { ticket_id: train.id, is_live: true, activation_phase: 'repair', started_at: resource.created_at };
  assert.equal(pipelineTicketStatuses([ticket], [], [heartbeat], []).get(train.id), 'repairing');
  assert.equal(both([ticket], [], [heartbeat]), 'active');
});

test('replacement tickets clear historical failures while independent held-out failures remain visible', () => {
  const failed = { ...train, status: 'failed' };
  const replacement = { ...train, id: 'new-train', created_at: resource.created_at, status: 'queued' };
  assert.equal(both([failed, replacement]), 'waiting');
  assert.equal(both([failed, { ...replacement, status: 'succeeded' }]), 'done');
  assert.equal(stationStateFromTickets([failed, { ...replacement, iteration: 3, status: 'succeeded' }]), 'done');
  assert.equal(stationStateFromTickets([{ ...failed, lane: 'held_out_test' }, { ...replacement, status: 'succeeded' }]), 'failed');
});

test('repair metadata remains visible after the repair agent hands off to a remote workload', () => {
  const repair = { ...train, repair_attempts: 1, repair_route: 'self', error_message: 'Retrying interrupted work' };
  const running = { ...resource, meta: { ...resource.meta, scheduler_state: 'RUNNING' } };
  assert.equal(pipelineTicketStatuses([repair], [running], [], []).get(train.id), 'repairing');
  assert.equal(pipelineTicketStatuses([repair], [resource], [], []).get(train.id), 'queued');
  assert.equal(both([repair], [running]), 'active');
  assert.equal(pipelineTicketStatuses([{ ...repair, status: 'succeeded' }], [running], [], []).get(train.id), 'succeeded');
  assert.equal(pipelineTicketStatuses([{ ...repair, repair_route: 'orchestrator' }], [running], [], []).get(train.id), 'running');
});
