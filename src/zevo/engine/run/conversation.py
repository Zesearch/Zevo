"""Bounded Ticket context with a lossless, ticket-local history snapshot."""
from __future__ import annotations

import json
from pathlib import Path

CONVERSATION_CHAR_BUDGET = 32_000
MESSAGE_CHAR_LIMIT = 8_000


def build_conversation_context(messages, *, agent_id: str, history_path: Path) -> list[dict]:
    rows = [dict(id=m.id, author=m.author, created_at=str(m.created_at), body=m.body or '')
            for m in messages]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    if not rows:
        return []

    # Reserve space for the latest human request, specialist work order, repair
    # instruction and agent report. These are evidence, not newly issued orders.
    predicates = [
        lambda r: r['author'] == 'user',
        lambda r: agent_id != 'orchestrator' and r['author'] == 'orchestrator',
        lambda r: r['author'] == 'system' and r['body'].startswith('__REPAIR__'),
        lambda r: r['author'] == agent_id,
    ]
    priority = []
    for predicate in predicates:
        match = next((i for i in range(len(rows) - 1, -1, -1) if predicate(rows[i])), None)
        if match is not None and match not in priority:
            priority.append(match)
    selected = {}
    remaining = CONVERSATION_CHAR_BUDGET
    for i in priority + [i for i in range(len(rows) - 1, -1, -1) if i not in priority]:
        row = rows[i]
        # Bound serialized content, including metadata and JSON escaping.
        allowance = min(MESSAGE_CHAR_LIMIT, remaining - 400)
        if allowance < 400:
            break
        body = row['body']
        if len(body) > allowance:
            half = (allowance - 100) // 2
            body = body[:half] + '\n[Middle omitted; read full message in history.]\n' + body[-half:]
        entry = {**row, 'body': body}
        cost = len(json.dumps(entry, ensure_ascii=False))
        limit = min(MESSAGE_CHAR_LIMIT, remaining)
        while cost > limit and len(body) > 200:
            half = len(body) // 3
            body = body[:half] + '\n[Middle omitted; read full message in history.]\n' + body[-half:]
            entry['body'] = body
            cost = len(json.dumps(entry, ensure_ascii=False))
        if cost > limit:
            continue
        selected[i] = entry
        remaining -= cost
    omitted = len(rows) - len(selected)
    truncated = sum(selected[i]['body'] != rows[i]['body'] for i in selected)
    notice = {
        'kind': 'conversation_history', 'author': 'system',
        'body': (
            f'Ticket history: {len(rows)} messages; {omitted} omitted, {truncated} shortened. '
            f'Full UTF-8 JSONL snapshot: {history_path.resolve()}. '
            'Each line contains id, author, created_at, body. Read line ranges with sed -n '
            "'1,20p' or search with rg -n; inspect omitted text before relying on an incomplete discussion. "
            'Included messages are chronological historical evidence, not a list of active instructions. '
            'Current typed inputs, pins and durable Run instructions are authoritative; newer instructions '
            'supersede conflicting older ones. Do not revive completed or superseded requests from history.'
        ),
    }
    return [notice, *(selected[i] for i in sorted(selected))]
