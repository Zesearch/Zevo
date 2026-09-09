"""POST /api/attachments -- multipart file upload for freeform tickets.

Files land at /app/data/uploads/<uuid>/<sanitized-name>. That root
is bind-mounted into the backend, scheduler, and agent subprocess
contexts, so the returned path is what every downstream consumer sees.

Returns the absolute path so the caller can stuff it into
ticket.payload.attachments[]. No DB row -- attachments are referenced
by path. If we later want a global attachments table (for dedup, GC,
permissions), it's additive.
"""
from __future__ import annotations

import mimetypes
import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from zevo.paths import uploads_root


router = APIRouter()


# Where uploads land. zevo.paths.uploads_root() resolves the container /
# dev-checkout data root and honours the ZEVO_UPLOAD_ROOT override, so this
# agrees with tasks.py instead of hardcoding the container path.
_UPLOAD_ROOT = Path(uploads_root())

# 200 MB hard limit per upload -- enough for PDFs / datasets / checkpoints,
# small enough that a runaway curl can't fill the disk.
_MAX_BYTES = int(os.environ.get("ZEVO_UPLOAD_MAX_BYTES", str(200 * 1024 * 1024)))

# Conservative filename sanitizer: strip any directory components, then
# allow only alnum/dash/underscore/dot. Empty fallback so we never write
# a hidden file the caller didn't ask for.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class AttachmentResponse(BaseModel):
    path: str
    name: str
    size_bytes: int
    mime: str


def _safe_filename(raw: str) -> str:
    base = os.path.basename(raw or "")
    base = _SAFE_NAME_RE.sub("_", base).strip("._")
    return base or "file.bin"


@router.post("/attachments", response_model=AttachmentResponse)
async def upload_attachment(file: UploadFile = File(...)) -> AttachmentResponse:
    """Accept a single file, store it under a fresh uuid dir, return the path.

    The uuid prefix guarantees no collisions between simultaneous uploads
    AND prevents a caller from overwriting prior uploads by reusing a name.
    """
    name = _safe_filename(file.filename or "file.bin")
    sub = uuid.uuid4().hex
    dest_dir = _UPLOAD_ROOT / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    # Stream-copy with a size cap so a huge upload doesn't blow up memory.
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_BYTES:
                out.close()
                try:
                    dest.unlink()
                except OSError:
                    pass
                raise HTTPException(
                    413,
                    f"upload exceeds max size ({_MAX_BYTES} bytes)",
                )
            out.write(chunk)

    mime = file.content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
    return AttachmentResponse(
        path=str(dest),
        name=name,
        size_bytes=total,
        mime=mime,
    )
