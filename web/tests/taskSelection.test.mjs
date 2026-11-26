import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/taskSelection.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'taskSelection.ts', { loader: 'ts' });
const { resolveLaunchTask } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const tasks = [{ id: 'haoyan-task', name: 'med' }, { id: 'solha-task', name: 'med' }];
test('same-name tasks resolve to the explicitly chosen owner in either order', () => {
  for (const rows of [tasks, [...tasks].reverse()]) {
    assert.equal(resolveLaunchTask(rows, 'med', 'solha-task'), tasks[1]);
    assert.equal(resolveLaunchTask(rows, 'med', 'haoyan-task'), tasks[0]);
  }
});
test('ambiguous names and stale IDs never silently pick another task', () => {
  assert.equal(resolveLaunchTask(tasks, 'med'), undefined);
  assert.equal(resolveLaunchTask(tasks, 'med', 'deleted-task'), undefined);
  assert.equal(resolveLaunchTask(tasks, 'legal', 'haoyan-task'), undefined);
});
test('single-user and standalone name-only tasks remain compatible', () => {
  const standalone = [{ name: 'math' }];
  assert.equal(resolveLaunchTask(standalone, ' math '), standalone[0]);
  assert.equal(resolveLaunchTask([tasks[0]], 'med'), tasks[0]);
});
