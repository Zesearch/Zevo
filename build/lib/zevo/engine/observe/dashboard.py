"""Dashboard aggregates over the caller's visible Run query."""
from datetime import datetime, timedelta, timezone
import math

from sqlalchemy import func

from zevo.contracts.tickets import TERMINAL_RUN_STATUSES
from zevo.db import Run
from zevo.engine.observe.run_metrics import baseline_and_best_test


async def aggregate_runs(db, visible_runs, queue_wait_state):
    # The API supplies the visibility predicate. Keep it on every query.
    counts = dict((await db.execute(
        visible_runs.with_only_columns(Run.status, func.count()).order_by(None).group_by(Run.status)
    )).all())
    durations = {key: 0 for key in ("1d", "1w", "1m", "all")}
    now = datetime.now(timezone.utc)
    cutoffs = {key: now - timedelta(days=days) for key, days in (("1d", 1), ("1w", 7), ("1m", 30))}
    groups = {}
    # Bound memory and the queue-wait IN clause, but visit every visible Run.
    after = None
    while True:
        query = visible_runs.with_only_columns(
            Run.id, Run.status, Run.started_at, Run.finished_at,
            Run.task_name, Run.metric, Run.metric_direction,
            Run.validation_metric_direction, Run.registry_version_tag,
            Run.champion_test_score, Run.history,
        ).where(Run.status.in_(TERMINAL_RUN_STATUSES)).order_by(Run.id).limit(500)
        if after is not None:
            query = query.where(Run.id > after)
        rows = (await db.execute(query)).all()
        if not rows:
            break
        after = rows[-1].id
        waits, _ = await queue_wait_state(db, rows)
        for row in rows:
            if row.started_at and row.finished_at:
                start = row.started_at.replace(tzinfo=timezone.utc) if row.started_at.tzinfo is None else row.started_at
                end = row.finished_at.replace(tzinfo=timezone.utc) if row.finished_at.tzinfo is None else row.finished_at
                duration = max(0, int((end - start).total_seconds()) - waits.get(row.id, 0))
                durations["all"] += duration
                for key, cutoff in cutoffs.items():
                    if end >= cutoff:
                        durations[key] += duration
            if row.status not in {"success", "degraded"} or not row.registry_version_tag:
                continue
            baseline, _ = baseline_and_best_test(row.history, row.validation_metric_direction)
            evolved = row.champion_test_score
            if baseline is None or evolved is None or not all(math.isfinite(x) for x in (baseline, evolved)):
                continue
            key = '\0'.join((row.task_name, row.metric, row.metric_direction))
            group = groups.setdefault(key, dict(
                key=key, taskName=row.task_name, metric=row.metric,
                metricDirection=row.metric_direction, runCount=0,
                averageBaseline=0.0, averageEvolved=0.0, averageImprovement=0.0,
                bestScore=evolved,
            ))
            group['runCount'] += 1
            group['averageBaseline'] += baseline
            group['averageEvolved'] += evolved
            group['averageImprovement'] += baseline - evolved if row.metric_direction == 'min' else evolved - baseline
            choose = min if row.metric_direction == 'min' else max
            group['bestScore'] = choose(group['bestScore'], evolved)
    for group in groups.values():
        for key in ('averageBaseline', 'averageEvolved', 'averageImprovement'):
            group[key] /= group['runCount']
    return dict(
        total=sum(counts.values()),
        active=sum(count for status, count in counts.items() if status not in TERMINAL_RUN_STATUSES),
        succeeded=counts.get('success', 0),
        failed=counts.get('failed', 0) + counts.get('halted', 0),
        runtime_seconds=durations,
        improvements=sorted(groups.values(), key=lambda group: (group['taskName'], group['metric'], group['metricDirection'])),
    )
