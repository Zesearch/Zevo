import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/scoringMetrics.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'scoringMetrics.ts', { loader: 'ts' });
const { BUILTIN_METRICS } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
test('task editing and launch choices support every backend built-in metric', () => {
  const backend = readFileSync(new URL('../../src/zevo/contracts/orchestrator.py', import.meta.url), 'utf8');
  const block = backend.match(/BUILTIN_METRICS = frozenset\(\{([\s\S]*?)\}\)/)[1].replace(/#.*$/gm, '');
  const supported = [...block.matchAll(/"([a-z0-9_]+)"/g)].map(match => match[1]);
  assert.deepEqual([...BUILTIN_METRICS].sort(), supported.sort());
  assert.ok(BUILTIN_METRICS.includes('pass_at_1'));
});
