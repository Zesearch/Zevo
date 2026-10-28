import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/predictionArtifacts.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'predictionArtifacts.ts', { loader: 'ts' });
const { currentPredictionArtifacts } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const artifact = (id, name, rows, second, meta = {}, path = `/suite/${name}.csv`) => ({
  id, role: 'predictions', path, created_at: `2026-10-28T11:27:${second}Z`,
  meta: {suite_member_name: name, n_rows: rows, ...meta},
});

test('iteration 1 counts 8082 unique predictions instead of 17104 historical rows', () => {
  const rows = [
    artifact('care1', 'CareQA', 5604, '30'), artifact('care2', 'CareQA', 5604, '31'),
    artifact('bio1', 'BioASQ', 1538, '30'), artifact('bio2', 'BioASQ', 1538, '31'),
    artifact('med1', 'MedExQA', 940, '30'),
    artifact('trial', 'MedExQA', 940, '40', {diagnostic_only: true}, '/attempts/trial.csv'),
    artifact('restored', 'MedExQA', 940, '41', {restored_original: true}),
  ];
  assert.equal(rows.reduce((n, a) => n + a.meta.n_rows, 0), 17104);
  const current = currentPredictionArtifacts(rows);
  assert.equal(current.reduce((n, a) => n + a.meta.n_rows, 0), 8082);
  assert.deepEqual(new Set(current.map(a => a.id)), new Set(['care2', 'bio2', 'restored']));
  assert.equal(current.length, 3); // Same list supplies the preview dropdown.
  assert.equal(rows.length, 7); // Historical artifacts are not deleted.
});

test('a rejected newest trial cannot replace the valid prediction; accepted replacements can', () => {
  const old = artifact('old', 'A', 10, '10');
  const rejected = artifact('rejected', 'A', 20, '30', {diagnostic_only: true}, '/trial.csv');
  assert.deepEqual(currentPredictionArtifacts([old, rejected]), [old]);
  const accepted = artifact('accepted', 'A', 12, '20', {}, '/accepted.csv');
  assert.deepEqual(currentPredictionArtifacts([rejected, old, accepted]), [accepted]);
});

test('legacy identities collapse duplicates without merging unrelated unnamed datasets', () => {
  const a = artifact('a', '', 10, '10', {test_set_name: 'validation:A'});
  const newer = artifact('newer', 'A', 10, '20');
  const b = artifact('b', '', 20, '10', {scoring_set: '/questions/b.csv'}, '/old-b.csv');
  const b2 = artifact('b2', '', 20, '20', {scoring_set: '/questions/b.csv'}, '/new-b.csv');
  const c = artifact('c', '', 30, '10', {}, '/c.csv');
  const c2 = artifact('c2', '', 30, '20', {}, '/c.csv');
  const d = artifact('d', '', 40, '10', {}, '/d.csv');
  assert.deepEqual(new Set(currentPredictionArtifacts([a, newer, b, b2, c, c2, d]).map(a => a.id)),
    new Set(['newer', 'b2', 'c2', 'd']));
});

test('validation and held-out are selected independently even with the same benchmark name', () => {
  const val = artifact('val', 'A', 10, '10');
  const heldout = artifact('test', 'A', 20, '20');
  assert.deepEqual(currentPredictionArtifacts([val]), [val]);
  assert.deepEqual(currentPredictionArtifacts([heldout]), [heldout]);
  assert.deepEqual(currentPredictionArtifacts([{...val, role: 'scoring_questions'}]), []);
});
