"""Policy shared by the watchdog, ticket API and runner admission guard."""
from datetime import datetime, timezone

FINALIZATION_SECONDS = 900

def finalization(run):
    return dict((getattr(run, "lifecycle", None) or {}).get("finalization") or {})

def finalization_allows(agent_id: str, lane: str = "optimization") -> bool:
    return agent_id in {"orchestrator", "registry", "inference", "evaluation"} or (
        agent_id == "data" and lane == "held_out_test"
    )

def aware(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
