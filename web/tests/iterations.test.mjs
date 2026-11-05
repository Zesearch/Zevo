import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
const { outputText } = ts.transpileModule(readFileSync(new URL('../src/lib/iterations.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { assignIterations, iterationOrder } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const baseline = { id: 'base4', agent_id: 'inference', iteration: 4, model_source: 'base_model' };
test('a new-model baseline at a later iteration is displayed as baseline', () => {
  const info = assignIterations([baseline, { id: 'eval4', agent_id: 'evaluation', iteration: 4 }]);
  assert.equal(info.get('base4').kind, 'baseline');
  assert.equal(info.get('eval4').kind, 'baseline');
  assert.equal(info.get('base4').label, 'Iteration 4 · baseline');
  assert.equal(info.get('base4').num, 4);
});
test('training from a base model is a trained iteration, not a baseline', () => {
  const train = { id: 'train4', agent_id: 'train', iteration: 4, model_source: 'base_model' };
  assert.equal(assignIterations([baseline, train]).get('train4').kind, 'iteration');
});
test('checkpoint inference and first baseline retain their roles', () => {
  const tickets = [{ ...baseline, id: 'initial', iteration: 0 }, { ...baseline, id: 'trained', model_source: 'checkpoint' }];
  const info = assignIterations(tickets);
  assert.equal(info.get('initial').label, 'Iteration 0 · baseline');
  assert.equal(info.get('trained').kind, 'iteration');
});

test('a supervisor decision for the next round does not create a phantom iteration', () => {
  const tickets = [
    { id: 'supervisor', agent_id: 'orchestrator', iteration: 3 },
    { id: 'data0', agent_id: 'data', iteration: 0 },
    { id: 'train1', agent_id: 'train', iteration: 1 },
    { id: 'train2', agent_id: 'train', iteration: 2 },
  ];
  assert.equal(assignIterations(tickets).get('supervisor').spansRun, true);
  assert.deepEqual(iterationOrder(tickets).map(g => g.num), [0, 1, 2]);
  assert.deepEqual(iterationOrder([tickets[0]]), []);
  // A real training ticket creates the round even when it reuses earlier Data.
  const rounds = iterationOrder([...tickets, { id: 'train3', agent_id: 'train', iteration: 3 }]);
  assert.deepEqual(rounds.map(g => g.num), [0, 1, 2, 3]);
  assert.deepEqual(rounds.at(-1).ticketIds, ['train3']);
});
