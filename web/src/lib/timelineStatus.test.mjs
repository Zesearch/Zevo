import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const source = readFileSync(new URL('./timelineStatus.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
const { timelineTicketStatus } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
const failed = Object.freeze({ agent_id: 'inference', iteration: 0, lane: 'optimization', operation: 'run_inference', model_source: 'base_model', test_set_name: '', created_at: '2026-10-08T04:00:00Z', status: 'failed' });
const success = { ...failed, created_at: '2026-10-08T05:00:00Z', status: 'succeeded' };

assert.equal(timelineTicketStatus([failed, success]), 'succeeded');
assert.equal(timelineTicketStatus([success, failed]), 'succeeded');
assert.equal(timelineTicketStatus([failed]), 'failed');
assert.equal(timelineTicketStatus([success, { ...failed, created_at: '2026-10-08T06:00:00Z' }]), 'failed');
for (const differentStage of [
  { iteration: 1 }, { lane: 'held_out_test' }, { agent_id: 'train' },
  { operation: 'release' }, { model_source: 'checkpoint' }, { test_set_name: 'another-benchmark' },
]) {
  assert.equal(timelineTicketStatus([failed, { ...success, ...differentStage }]), 'failed');
}
assert.equal(timelineTicketStatus([failed, { ...success, status: 'running' }]), 'running');
assert.equal(timelineTicketStatus([failed, { ...success, status: 'repairing' }]), 'running');
assert.equal(timelineTicketStatus([failed, { ...success, status: 'skipped' }]), 'failed');
assert.equal(timelineTicketStatus([failed, success, { ...success, agent_id: 'evaluation', status: 'queued' }]), 'pending');
assert.equal(timelineTicketStatus([{ ...failed, status: 'degraded' }, success]), 'succeeded');
assert.equal(timelineTicketStatus([{ ...failed, status: 'cancelled' }, success]), 'succeeded');
assert.equal(timelineTicketStatus([failed, { ...failed, created_at: '2026-10-08T04:30:00Z' }, success]), 'succeeded');
assert.equal(timelineTicketStatus([]), 'pending');
assert.equal(failed.status, 'failed');
console.log('timeline recovery regression checks passed');

const cancelled = { ...failed, status: 'cancelled' };
assert.equal(timelineTicketStatus([cancelled]), 'cancelled');
assert.equal(timelineTicketStatus([cancelled, { ...success, agent_id: 'data' }]), 'cancelled');
assert.equal(timelineTicketStatus([cancelled, { ...success, status: 'queued' }]), 'pending');
assert.equal(timelineTicketStatus([cancelled, { ...success, status: 'running' }]), 'running');
assert.equal(timelineTicketStatus([cancelled, success]), 'succeeded');
assert.equal(timelineTicketStatus([success, { ...cancelled, created_at: '2026-10-08T06:00:00Z' }]), 'cancelled');

// Run 1520c6e3 iteration 3: cancelled checkpoint continuation was replaced
// by a successful base-model training attempt within the same iteration.
const oldTrain = { ...cancelled, agent_id: 'train', operation: 'train', iteration: 3, model_source: 'checkpoint' };
const replacementTrain = { ...oldTrain, created_at: success.created_at, model_source: 'base_model', status: 'succeeded' };
assert.equal(timelineTicketStatus([oldTrain, replacementTrain]), 'succeeded');
assert.equal(timelineTicketStatus([replacementTrain, oldTrain]), 'succeeded');
assert.equal(timelineTicketStatus([oldTrain, { ...replacementTrain, iteration: 4 }]), 'cancelled');
assert.equal(timelineTicketStatus([oldTrain, { ...replacementTrain, status: 'queued' }]), 'pending');
