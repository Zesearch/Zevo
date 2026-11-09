import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/stageLabels.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'stageLabels.ts', { loader: 'ts' });
const { stageAttemptLabel: label } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const normal = {operation: '', activation_phase: '', execution_phase: ''};

test('every role has a semantic name even for old records with no operation', () => {
  const expected = {
    infrastructure: 'Provision Compute', data: 'Prepare Data', train: 'Select config and train candidate',
    inference: 'Run Candidate Inference', evaluation: 'Evaluate Validation',
    registry: 'Save and Register Model', orchestrator: 'Plan Run',
  };
  for (const [agent_id, name] of Object.entries(expected)) {
    assert.equal(label({agent_id}, normal), name);
    assert.equal(label({agent_id}, {...normal, activation_phase:'repair'}), `${name} · Repair`);
  }
});

test('execution phase and current activity are preserved with exactly one repair suffix', () => {
  assert.equal(label({agent_id:'inference'}, {...normal, activation_phase:'repair', execution_phase:'collect'}),
    'Collect Candidate Inference Results · Repair');
  assert.equal(label({agent_id:'train'}, {...normal, activation_phase:'repair', execution_phase:'submit'}, 2, 'Training'),
    'Training · Repair');
  assert.equal(label({agent_id:'registry'}, {...normal, activation_phase:'repair', execution_phase:'repair'}),
    'Save and Register Model · Repair');
});

test('repeated execution alone is not mislabeled as repair and special stages remain visible', () => {
  assert.equal(label({agent_id:'orchestrator'}, normal, 2), 'Review Run');
  assert.equal(label({agent_id:'registry'}, {...normal, activation_phase:'instruction'}), 'Apply Instruction');
  assert.equal(label({agent_id:'inference'}, {...normal, activation_phase:'remeasure'}), 'Length Remeasurement');
  assert.equal(label({agent_id:'evaluation',lane:'held_out_test'}, {...normal,activation_phase:'repair'}, 7),
    'Evaluate Held-Out · Repair');
  assert.equal(label({agent_id:'infrastructure'}, {...normal,operation:'release',activation_phase:'repair'}),
    'Release resources · Repair');
});

test('checkpoint finalization is distinct from continuing optimizer steps', () => {
  const train = {agent_id: 'train'};
  const resume = {...normal, activation_phase: 'continue'};
  assert.equal(label(train, resume), 'Continue Candidate Training');
  assert.equal(label(train, {...resume, training_finalization: false}), 'Continue Candidate Training');
  const finalize = {...resume, training_finalization: true};
  assert.equal(label(train, finalize), 'Finalize Candidate Training');
  assert.equal(label(train, {...finalize, activation_phase: 'repair', execution_phase: 'submit'}),
    'Finalize Candidate Training · Repair');
  assert.equal(label(train, {...finalize, activation_phase: 'instruction'}), 'Apply Instruction');
  assert.equal(label(train, finalize, 1, 'Saving model'), 'Saving model');
  assert.equal(label({agent_id: 'inference'}, finalize), 'Continue Candidate Inference');
});
