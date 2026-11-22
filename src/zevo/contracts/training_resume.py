"""Machine-readable training resume intent, with legacy config compatibility."""
from pathlib import PurePosixPath


def resume_mode(implementation):
    mode = implementation.get("resume_mode")
    path = implementation.get("resume_path")
    aliases = [implementation.get(k) for k in
               ("resume_from_checkpoint", "resume_checkpoint", "resume")]
    active = [v for v in aliases if v is not None and v is not False]
    if mode is not None:
        if mode not in ("none", "checkpoint"):
            raise ValueError("resume_mode must be none or checkpoint; resolve auto detection before submission")
        if active:
            raise ValueError("Use resume_mode/resume_path without enabled legacy resume aliases")
        if mode == "none" and path is not None:
            raise ValueError("resume_mode none requires resume_path null or absent")
        if mode == "checkpoint" and not (isinstance(path, str) and PurePosixPath(path).is_absolute()):
            raise ValueError("resume_mode checkpoint requires an absolute resume_path")
        return mode
    if path is not None:
        raise ValueError("resume_path requires explicit resume_mode")
    if not active:
        return "none"
    if all(v is True or isinstance(v, str) and PurePosixPath(v).is_absolute() for v in active):
        return "checkpoint"
    raise ValueError("Ambiguous legacy resume value: replace prose/auto with resume_mode none or checkpoint and resume_path")
