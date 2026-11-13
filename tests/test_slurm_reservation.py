import asyncio

import pytest
from pydantic import ValidationError

from zevo.engine.run import slurm_capacity
from zevo.engine.run.gpu_controller import controller_script
from zevo.engine.run.slurm_reservation import apply_reservation, reservation_nodes
from tests.test_stage_resource_planning import _cluster_info, _partial_node_constraints
from tests.test_remote_evaluation import work, cluster_contract
from zevo.engine.remote_evaluation import launch_script


RECORD = ('ReservationName=reserved StartTime=2026-11-01T00:00:00 '
          'EndTime=2026-12-01T00:00:00 Nodes=gpu[01-02] Users=user Accounts=(null)'
          '\n__ZEVO_CLOCK__2026-11-13T16:00:00-05:00\n')


def test_optional_field_and_unsafe_name():
    info = _cluster_info(constraints=_partial_node_constraints())
    assert info.cluster.reservation == ''
    data = info.model_dump()
    data['cluster']['reservation'] = 'bad\n#SBATCH --exclusive'
    with pytest.raises(ValidationError):
        type(info).model_validate(data)


@pytest.mark.parametrize('stage', ['data', 'train', 'inference', 'evaluation'])
def test_allocation_header_preserves_reservation(stage):
    body = f'#!/bin/bash\n#SBATCH --job-name={stage}\n#SBATCH --gpus=1\necho run\n'
    scoped = apply_reservation(body, 'reserved')
    assert apply_reservation(body, '') == body
    assert apply_reservation(scoped, 'reserved') == scoped
    assert '#SBATCH --reservation=reserved\n' in controller_script(scoped, '/remote', single_workload=True)
    with pytest.raises(ValueError, match='conflicts'):
        apply_reservation(scoped, 'other')


def test_evaluation_script_contains_reservation(tmp_path):
    inp = work(tmp_path)
    inp.slurm_job = cluster_contract(tmp_path)
    info = _cluster_info(constraints=_partial_node_constraints())
    info.cluster.reservation = 'reserved'
    assert '#SBATCH --reservation=reserved' in launch_script(inp, info, '/remote')


@pytest.mark.parametrize('replace,message', [
    (('ReservationName=reserved', 'ReservationName=other'), 'not returned'),
    (('2026-12-01', '2026-11-02'), 'expired'),
    (('2026-11-01', '2026-11-20'), 'not active'),
    (('Nodes=gpu[01-02]', 'Nodes=(null)'), 'node scope'),
])
def test_invalid_reservation_is_not_ordinary_capacity(replace, message):
    with pytest.raises(ValueError, match=message):
        reservation_nodes(RECORD.replace(*replace), 'reserved')


def test_probe_scopes_nodes_and_keeps_busy_reservation(monkeypatch):
    info = _cluster_info(constraints=_partial_node_constraints())
    info.cluster.reservation = 'reserved'
    info.cluster.qos = ''
    info.resource_plan.slurm_qos = ''
    commands = []
    async def query(info, command, **kwargs):
        commands.append(command)
        if 'scontrol' in command:
            return RECORD
        return 'gpu01|gpu:8|gpu:8|mix\ngpu02|gpu:8|gpu:8|alloc\n'
    monkeypatch.setattr(slurm_capacity, '_query', query)
    snapshot = asyncio.run(slurm_capacity.probe_slurm_capacity(info))
    assert "-n 'gpu[01-02]'" in commands[1]
    assert snapshot.immediately_free_gpus == 0
    assert len(commands) == 2


def test_invalid_reservation_stops_before_partition_probe(monkeypatch):
    info = _cluster_info(constraints=_partial_node_constraints())
    info.cluster.reservation = 'reserved'
    commands = []
    async def query(info, command, **kwargs):
        commands.append(command)
        raise ValueError('Access denied')
    monkeypatch.setattr(slurm_capacity, '_query', query)
    with pytest.raises(ValueError, match='Access denied'):
        asyncio.run(slurm_capacity.probe_slurm_capacity(info))
    assert len(commands) == 1
