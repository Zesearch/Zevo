import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { transformWithEsbuild } from 'vite';
const source = readFileSync(new URL('../src/lib/format.ts', import.meta.url), 'utf8');
const { code } = await transformWithEsbuild(source, 'format.ts', { loader: 'ts' });
const { fmtScore, isPercentageMetric, metricScoreRange } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('old and current OLMO validation/test protocols display consistent percentages', () => {
  for (const metric of [
    'answer_accuracy', 'choice_accuracy', 'instruction_following_accuracy',
    'pairwise_win_rate_vs_human_reference_model_judge', 'pairwise_win_rate_vs_reference_model_judge',
    'suite_average', 'pass_at_1', 'exact_match', 'puzzle_accuracy',
    'validation_score', 'exact_match_flex', 'prompt_level_strict_acc', 'prompt_level_loose_acc',
    'inst_level_strict_acc', 'inst_level_loose_acc', 'length_controlled_winrate', 'alias_substring_match',
  ]) {
    assert.equal(fmtScore(0.75, metric), '75.0%', metric);
    assert.equal(fmtScore(0, metric), '0.0%', metric);
    assert.equal(fmtScore(1, metric), '100.0%', metric);
    assert.equal(isPercentageMetric(metric), true, metric);
    assert.equal(metricScoreRange(metric), '0–1 (0–100%)', metric);
  }
});

test('loss, duration and unknown custom metrics retain their raw scale', () => {
  assert.equal(fmtScore(0.75, 'loss'), '0.750');
  assert.equal(fmtScore(250, 'latency_ms'), '250.000');
  assert.equal(fmtScore(0.75, 'custom_score'), '0.750');
  assert.equal(fmtScore(0.75, ''), '0.750');
  assert.equal(fmtScore(null, 'validation_score'), '—');
  assert.equal(fmtScore(NaN, 'validation_score'), '—');
});
