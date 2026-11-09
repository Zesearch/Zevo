import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const shared = readFileSync(new URL('../src/lib/executionProgress.ts', import.meta.url), 'utf8');
const source = readFileSync(new URL('../src/lib/measurementProgress.ts', import.meta.url), 'utf8').replace('import { benchmarkCounts } from "./executionProgress";', '');
const { code } = await transformWithEsbuild(`${shared}\n${source}`, 'progress.ts', { loader: 'ts' });
const { measurementProgress } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
test('new revision starts at zero and counts only active members and current workload events', () => {
  const event = (ts, name, purpose = 'workload') => ({ts, extras: { benchmark_name: name, benchmark_complete: true, execution_purpose: purpose }});
  const ticket = {status: 'queued', measurement_attempt: {revision: 'new', started_at: '2026-10-26T10:00:00Z', active_members: ['A']},
    execution_events: [event('2026-10-26T09:00:00Z','A'), event('2026-10-26T10:01:00Z','B'), event('2026-10-26T10:01:00Z','A','preflight')]};
  assert.deepEqual(measurementProgress(ticket), {completed: 0, total: 1});
  ticket.execution_events.push(event('2026-10-26T10:02:00Z','A'));
  assert.deepEqual(measurementProgress(ticket), {completed: 1, total: 1});
  ticket.measurement_attempt.started_at = '2026-10-26T11:00:00Z';
  assert.deepEqual(measurementProgress(ticket), {completed: 0, total: 1});
});

test('ticket success does not fabricate completion for a scoped execution', () => {
  const ticket = {status: 'succeeded', measurement_attempt: {revision: 'new', started_at: '2026-10-26T10:00:00Z', active_members: ['A']}, execution_events: []};
  assert.deepEqual(measurementProgress(ticket, []), {completed: 0, total: 1});
  assert.equal(measurementProgress(ticket, [], false), null);
});

const assigned = Array.from({length: 12}, (_, i) => ({id: `validation:${i}`, name: `Benchmark ${i}`}));
const complete = (index, batchTotal = 11) => ({ts: '2026-10-26T10:02:00Z', extras: {
  execution_purpose: 'workload', benchmark_id: assigned[index].id,
  benchmark_name: assigned[index].name, benchmark_complete: true, benchmark_total: batchTotal,
}});

test('11 CSV benchmarks plus one JSONL benchmark use the 12-member assignment throughout', () => {
  const ticket = {measurement_members: assigned, execution_events: []};
  assert.deepEqual(measurementProgress(ticket), {completed: 0, total: 12});
  ticket.execution_events = assigned.slice(0, 11).map((_, i) => complete(i));
  assert.deepEqual(measurementProgress(ticket), {completed: 11, total: 12});
  ticket.execution_events.push(complete(11, 1));
  assert.deepEqual(measurementProgress(ticket), {completed: 12, total: 12});
  // Replay and artifact verification cannot count one benchmark twice.
  ticket.execution_events.push(complete(11, 12), complete(0, 12));
  assert.deepEqual(measurementProgress(ticket), {completed: 12, total: 12});
});

test('assigned identities reject foreign IDs and preflight, and tolerate legacy name-only markers', () => {
  const event = complete(0);
  const ticket = {measurement_members: assigned, execution_events: [
    {...event, extras: {...event.extras, benchmark_id: 'validation:99'}},
    {...event, extras: {...event.extras, progress_scope: 'preflight'}},
    {...event, extras: {...event.extras, execution_purpose: 'diagnostic'}},
    {...event, extras: {...event.extras, benchmark_id: '', benchmark_name: 'unknown'}},
  ]};
  assert.deepEqual(measurementProgress(ticket), {completed: 0, total: 12});
  ticket.execution_events.push({...event, extras: {...event.extras, benchmark_id: ''}},
    {...event, extras: {...event.extras, benchmark_name: 'different display name'}});
  assert.deepEqual(measurementProgress(ticket), {completed: 1, total: 12});
});

test('assigned suite respects remeasurement subset, start boundary and selected execution', () => {
  const ticket = {measurement_members: assigned, measurement_attempt: {
    revision: 'new', started_at: '2026-10-26T10:00:00Z', active_members: ['Benchmark 1', 'Benchmark 11'],
  }, execution_events: [complete(0), {...complete(1), ts: '2026-10-26T09:00:00Z'}, complete(11, 1)]};
  assert.deepEqual(measurementProgress(ticket), {completed: 1, total: 2});
  assert.deepEqual(measurementProgress(ticket, []), {completed: 0, total: 2});
  ticket.execution_events.push(complete(1));
  assert.deepEqual(measurementProgress(ticket), {completed: 2, total: 2});
  assert.deepEqual(measurementProgress(ticket, [complete(0)], false), {completed: 1, total: 12});
});

test('single-member assigned inference ignores batch totals', () => {
  const ticket = {measurement_members: [assigned[11]], execution_events: [complete(11, 12)]};
  assert.deepEqual(measurementProgress(ticket), {completed: 1, total: 1});
});
