import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'training_response', Path(__file__).resolve().parents[1] / 'playbook/runners/training_response.py',
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_general_sft_preserves_responses_without_requiring_answer_extraction():
    for response in ['A poem about stars.', 'def solve():\n    return 42', 'An ambiguous discussion of A and B.']:
        assert helper.prepare_response(response) == response
        helper.validate_training_response(response, expected_response=response,
                                          input_ids=[10, 11], labels=[-100, 11])


def test_boxing_preserves_rationale_and_is_idempotent():
    source = 'Reasoning about A and B.\nFinal Answer: D. Explanation.'
    start = source.index('D.')
    result = helper.prepare_response(source, scope='task', output_format='boxed',
                                     requirement='Objective requires boxed answer',
                                     gold='D', answer_span=(start, start + 1))
    assert result == 'Reasoning about A and B.\nFinal Answer: \\boxed{D}. Explanation.'
    assert helper.prepare_response(result, scope='task', output_format='boxed',
                                   requirement='Objective requires boxed answer',
                                   gold='D', answer_span=(start, start + 9)) == result


@pytest.mark.parametrize('kwargs', [
    {},
    {'scope': 'task', 'requirement': 'boxed'},
    {'scope': 'task', 'requirement': 'boxed', 'gold': 'B', 'answer_span': (0, 1)},
    {'scope': 'task', 'requirement': 'boxed', 'gold': 'A', 'answer_span': (0, 99)},
])
def test_adaptation_rejects_missing_authority_or_unreliable_answer(kwargs):
    with pytest.raises(ValueError):
        helper.prepare_response('A', output_format='boxed', **kwargs)


def test_boxed_instruction_with_plain_target_is_rejected():
    with pytest.raises(ValueError, match='conflicts'):
        helper.validate_training_response('Final Answer: D', expected_response='Final Answer: D',
                                          input_ids=[1, 2], labels=[-100, 2],
                                          required_format='boxed', gold='D')


@pytest.mark.parametrize('labels,span,error', [
    ([-100, 2], (1, 3), 'truncated'),
    ([-100, 2], (0, 1), 'masked'),
    ([-100, -100], (1, 2), 'empty supervision'),
    ([-100, 3], (1, 2), 'disagree'),
])
def test_actual_token_checks(labels, span, error):
    with pytest.raises(ValueError, match=error):
        helper.validate_training_response('A', expected_response='A', input_ids=[1, 2],
                                          labels=labels, answer_token_span=span)


def test_content_loss_is_rejected():
    with pytest.raises(ValueError, match='changed source'):
        helper.validate_training_response('A', expected_response='Explanation. A',
                                          input_ids=[1, 2], labels=[-100, 2])


def test_answer_supervision_passes():
    helper.validate_training_response(r'Reasoning. \boxed{A}', expected_response=r'Reasoning. \boxed{A}',
                                      required_format='boxed', gold='A', input_ids=[1, 2, 3],
                                      labels=[-100, 2, 3], answer_token_span=(1, 3))


def test_legal_answer_line_alignment_preserves_explanation_and_gold():
    source = 'Legal Concept: duty.\nRule Application: Answer: A is not correct.\nConclusion: C follows.\nAnswer: C'
    aligned = helper.prepare_response(source, scope='matching_records', output_format='answer_line',
        requirement='Frozen inference prompt requires a final answer line', gold='C',
        answer_span=(len(source)-1, len(source)), answer_template='Final Answer: {answer}')
    assert aligned == source.removesuffix('Answer: C') + 'Final Answer: C'
    assert helper.prepare_response(aligned, scope='matching_records', output_format='answer_line',
        requirement='Frozen inference prompt', gold='C', answer_span=(len(aligned)-1, len(aligned)),
        answer_template='Final Answer: {answer}') == aligned
    helper.validate_training_response(aligned, expected_response=aligned,
        required_format='answer_line', answer_template='Final Answer: {answer}', gold='C',
        input_ids=[10, 20, 30], labels=[-100, 20, 30], answer_token_span=(2, 3))


def test_answer_only_alignment_does_not_invent_reasoning():
    assert helper.prepare_response('Answer: B', scope='task', output_format='answer_line',
        requirement='Final answer line', gold='B', answer_span=(8, 9),
        answer_template='Final Answer: {answer}') == 'Final Answer: B'


@pytest.mark.parametrize('source,gold,span', [
    ('Reasoning says C', 'C', (15, 16)),
    ('Answer: C\nMore explanation.', 'C', (8, 9)),
    ('Answer: B', 'C', (8, 9)),
    ('An organism name with no option.', 'C', None),
])
def test_line_alignment_does_not_guess_or_drop_reasoning(source, gold, span):
    with pytest.raises(ValueError):
        helper.prepare_response(source, scope='task', output_format='answer_line',
            requirement='Final answer line', gold=gold, answer_span=span,
            answer_template='Final Answer: {answer}')


@pytest.mark.parametrize('template', [None, 'Final Answer: C', '{answer}{answer}', '{other}',
                                       'Final Answer:\n{answer}', 'Final Answer: {answer} '])
def test_line_alignment_requires_explicit_safe_template(template):
    with pytest.raises(ValueError):
        helper.prepare_response('Answer: C', scope='task', output_format='answer_line',
            requirement='Final answer line', gold='C', answer_span=(8, 9), answer_template=template)


def test_preserving_old_legal_target_cannot_satisfy_final_answer_contract():
    with pytest.raises(ValueError, match='conflicts'):
        helper.validate_training_response('Answer: C', expected_response='Answer: C',
            required_format='answer_line', answer_template='Final Answer: {answer}', gold='C',
            input_ids=[1, 2], labels=[-100, 2], answer_token_span=(1, 2))
    with pytest.raises(ValueError, match='preserve'):
        helper.validate_training_response('Answer: C', expected_response='Answer: C',
            answer_template='Final Answer: {answer}', gold='C', input_ids=[1], labels=[1])
    with pytest.raises(ValueError, match='preserve'):
        helper.prepare_response('Answer: C', answer_template='Final Answer: {answer}')


@pytest.mark.parametrize('labels,span,error', [([-100, 2], None, 'requires'),
    ([-100, 2], (1, 3), 'truncated'), ([-100, 2], (0, 1), 'masked')])
def test_aligned_answer_still_requires_real_supervision(labels, span, error):
    with pytest.raises(ValueError, match=error):
        helper.validate_training_response('Final Answer: C', expected_response='Final Answer: C',
            required_format='answer_line', answer_template='Final Answer: {answer}', gold='C',
            input_ids=[1, 2], labels=labels, answer_token_span=span)


def test_choice_gold_uses_record_options_not_benchmark_range():
    source = 'Reasoning.\nAnswer: G'
    options = list('ABCDEFGHIJ')
    result = helper.prepare_response(source, scope='task', output_format='answer_line',
        requirement='Final answer line', gold='G', answer_span=(len(source)-1,len(source)),
        answer_template='Final Answer: {answer}')
    helper.validate_training_response(result, expected_response=result, gold='G',
        option_labels=options, required_format='answer_line', answer_template='Final Answer: {answer}',
        input_ids=[1,2], labels=[-100,2], answer_token_span=(1,2))
    with pytest.raises(ValueError, match='not among'):
        helper.validate_training_response(result,expected_response=result,gold='G',
            option_labels=list('ABCD'),input_ids=[1,2],labels=[-100,2])


@pytest.mark.parametrize('options,gold',[([], 'G'),(['A','A'],'A'),(['a','b'],'a'),(['A','C'],'B')])
def test_choice_gold_requires_explicit_valid_options(options,gold):
    with pytest.raises(ValueError): helper.validate_choice_gold(gold,option_labels=options)
