import json
from types import SimpleNamespace

from zevo.engine.run.conversation import build_conversation_context, CONVERSATION_CHAR_BUDGET
from zevo.engine.agent.drivers._prompt import _conversation_block


def message(i, author='train', body='progress'):
    return SimpleNamespace(id=str(i), author=author, body=body, created_at=f'2026-10-31 {i:04d}')


def test_short_history_keeps_more_than_twelve(tmp_path):
    rows = [message(i) for i in range(60)]
    context = build_conversation_context(rows, agent_id='train', history_path=tmp_path/'history.jsonl')
    assert [c['id'] for c in context[1:]] == [str(i) for i in range(60)]
    assert '0 omitted' in context[0]['body']


def test_long_history_preserves_key_context_and_full_originals(tmp_path):
    rows = [message(0, 'user', 'old request'), message(1, 'orchestrator', 'work order'),
            message(2, 'system', '__REPAIR__\nExact failure: fix this'), message(3, 'user', 'new request')]
    rows += [message(i, 'system', 'Running: ' + 'x'*5000) for i in range(4, 40)]
    rows.append(message(40, 'train', 'Latest verified artifacts'))
    path = tmp_path/'history.jsonl'
    context = build_conversation_context(rows, agent_id='train', history_path=path)
    ids = [c['id'] for c in context[1:]]
    assert {'1','2','3','40'} <= set(ids)
    assert '0' not in ids
    assert ids == sorted(ids, key=int)
    assert sum(len(json.dumps(c, ensure_ascii=False)) for c in context[1:]) <= CONVERSATION_CHAR_BUDGET
    full = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['body'] for r in full] == [r.body for r in rows]
    rendered = _conversation_block(context)
    assert str(path) in rendered
    assert 'id=2' in rendered
    assert 'not a list of active instructions' in rendered


def test_large_critical_messages_keep_tail_and_fit_budget(tmp_path):
    rows = [message(0,'user','"'*20000+'USER-END'),
            message(1,'orchestrator','\\'*20000+'ORDER-END'),
            message(2,'system','__REPAIR__'+'x'*20000+'EXACT-ERROR'),
            message(3,'train','x'*20000+'RESULT-END')]
    context = build_conversation_context(rows, agent_id='train', history_path=tmp_path/'history.jsonl')
    assert len(context) == 5
    assert [r['body'].splitlines()[-1][-9:] for r in context[1:]] == [r.body[-9:] for r in rows]
    assert sum(len(json.dumps(c, ensure_ascii=False)) for c in context[1:]) <= CONVERSATION_CHAR_BUDGET
    assert '4 shortened' in context[0]['body']


def test_empty_history_and_durable_instructions(tmp_path):
    assert build_conversation_context([], agent_id='train', history_path=tmp_path/'empty.jsonl') == []
    rendered = _conversation_block([{'kind':'run_instruction','id':'r1','status':'scheduled','body':'Finish then register'}])
    assert 'RUN USER INSTRUCTIONS' in rendered
    assert 'Finish then register' in rendered
