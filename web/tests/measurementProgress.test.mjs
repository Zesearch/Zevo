import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/measurementProgress.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'progress.ts', { loader: 'ts' });
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
