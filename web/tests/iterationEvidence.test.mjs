import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/iterationEvidence.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'iterationEvidence.ts', { loader: 'ts' });
const { completedTrainingData, hasIterationEvidence } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const data = { ticket_id: 'data-baseline', role: 'training_dataset' };
const ticket = { id: data.ticket_id, agent_id: 'data', lane: 'optimization', status: 'succeeded' };

test('a training round reusing completed baseline Data appears before its final outputs', () => {
  const active = completedTrainingData([data], new Map([[ticket.id, ticket]]));
  assert.equal(hasIterationEvidence([], active), true);
  assert.equal(active[0], data); // Keep the original artifact for preview and row counts.
});

test('a ticket alone or unpublished/unfinished Data does not create an empty card', () => {
  assert.equal(hasIterationEvidence([], []), false);
  for (const status of ['queued', 'running', 'failed', 'cancelled']) {
    const active = completedTrainingData([data], new Map([[ticket.id, {...ticket, status}]]));
    assert.equal(hasIterationEvidence([data], active), false, status);
  }
});

test('held-out preparation is not training Data, and existing result visibility remains intact', () => {
  assert.deepEqual(completedTrainingData([data], new Map([[ticket.id, {...ticket, lane: 'held_out_test'}]])), []);
  for (const role of ['checkpoint', 'predictions']) {
    assert.equal(hasIterationEvidence([{ticket_id: 'result', role}], []), true);
  }
});
