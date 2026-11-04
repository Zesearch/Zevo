import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/checkpointComparison.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'checkpointComparison.ts', { loader: 'ts' });
const { comparisonPoints, comparisonRows } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const data = { measurements: [
  { step: 100, score: 0.5, metrics: { Coding: 0.1, Math: 0.9 } },
  { step: 200, score: 0.6, metrics: { Coding: 0.3, Math: 0.9 } },
] };

test('baseline and each completed checkpoint are independently selectable', () => {
  const points = comparisonPoints(data, { score: 0.4, validation_scores: { Coding: 0, Math: 0.8 } });
  assert.deepEqual(points.map(p => p.id), ['baseline', 'step:100', 'step:200']);
  const baseline = comparisonRows(points[0], points[1]);
  assert.equal(baseline.find(r => r.name === 'Coding').left, 0);
  assert.equal(baseline.find(r => r.name === 'Coding').delta, 0.1);
  const forward = comparisonRows(points[1], points[2]);
  const reverse = comparisonRows(points[2], points[1]);
  forward.forEach((row, i) => assert.equal(row.delta + reverse[i].delta, 0));
  assert.equal(forward.find(r => r.name === 'Math').delta, 0);
});

test('missing results are not zero and baseline absence does not prevent checkpoint comparisons', () => {
  assert.equal(comparisonPoints(data).length, 2);
  const rows = comparisonRows({ score: 0, metrics: { A: 0, B: NaN } }, { score: 0.2, metrics: { C: 0.5, B: 0.1 } });
  assert.equal(rows[0].delta, 0.2);
  assert.deepEqual(rows.slice(1).map(r => r.delta), [null, null, null]);
  assert.deepEqual(comparisonPoints({ measurements: [] }), []);
});
