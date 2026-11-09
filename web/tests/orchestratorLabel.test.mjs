import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/lib/orchestratorLabel.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { withOrchestratorRepairLabel: label } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);

test('ordinary action text stays unchanged; repair keeps the actual action', () => {
  assert.equal(label({}, 'Start Evaluation'), 'Start Evaluation');
  assert.equal(label({repair_source_agent_id: 'train'}, 'Start iteration 3 execution'),
    'Repair · Train → Start iteration 3 execution');
  assert.equal(label({repair_source_agent_id: 'train'}, 'Wait for dependencies'),
    'Repair · Train → Wait for dependencies');
});

test('orchestrator self repair differs from reviewing a failed child', () => {
  assert.equal(label({activation_phase: 'repair', repair_source_agent_id: 'train'}, 'Orchestrator deciding…'),
    'Repair · Orchestrator → Orchestrator deciding…');
});
