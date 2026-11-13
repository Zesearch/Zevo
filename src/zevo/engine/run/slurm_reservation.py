"""Preserve an optional site reservation through engine-owned allocations."""
from __future__ import annotations

import re
import shlex
from datetime import datetime


def script_reservations(body: str) -> list[str]:
    values = []
    for line in body.splitlines():
        if not line.lstrip().startswith('#SBATCH'):
            continue
        args = shlex.split(line.lstrip()[len('#SBATCH'):], comments=True)
        for i, arg in enumerate(args):
            if arg == '--reservation':
                values.append(args[i + 1] if i + 1 < len(args) else '')
            elif arg.startswith('--reservation='):
                values.append(arg.split('=', 1)[1])
    return values


def apply_reservation(body: str, reservation: str) -> str:
    if not reservation:
        return body
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', reservation):
        raise ValueError('invalid Slurm reservation name')
    existing = script_reservations(body)
    if existing:
        if any(value != reservation for value in existing):
            raise ValueError('stage script conflicts with configured Slurm reservation')
        return body
    lines = body.splitlines(keepends=True)
    index = 1 if lines and lines[0].startswith('#!') else 0
    lines.insert(index, f'#SBATCH --reservation={reservation}\n')
    return ''.join(lines)


def reservation_nodes(output: str, reservation: str) -> str:
    """Read scontrol's record and remote clock (same timezone as its dates)."""
    record, clock = output.strip().rsplit('\n__ZEVO_CLOCK__', 1)
    fields = dict(re.findall(r'(\w+)=(\S+)', record))
    if fields.get('ReservationName') != reservation:
        raise ValueError(f'Slurm reservation {reservation} was not returned')
    now = datetime.fromisoformat(clock.strip())
    for key in ('StartTime', 'EndTime'):
        try:
            value = datetime.fromisoformat(fields[key]).replace(tzinfo=now.tzinfo)
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Cannot verify reservation {key}') from exc
        if key == 'EndTime' and value <= now:
            raise ValueError(f'Slurm reservation {reservation} has expired')
        if key == 'StartTime' and value > now:
            raise ValueError(f'Slurm reservation {reservation} is not active yet')
    nodes = fields.get('Nodes', '')
    if not nodes or nodes in {'(null)', 'ALL'} or not re.fullmatch(r'[A-Za-z0-9_.,\[\]-]+', nodes):
        raise ValueError('Slurm reservation has no concrete node scope')
    return nodes
