"""Validation size errors should be reported before asynchronous Run setup."""
from types import SimpleNamespace

from zevo.api.routers.ui.preflight import _check_validation_sets


def test_local_validation_size_matches_run_setup_minimum(tmp_path):
    path = tmp_path / "validation.csv"
    request = SimpleNamespace(validation_sets=[SimpleNamespace(name="medbullets-smoke", test_set=str(path))])

    path.write_text("id,answer\n" + "".join(f"{i},A\n" for i in range(50)))
    items = []
    _check_validation_sets(request, items)
    assert [item.code for item in items] == ["validation_set_1_too_small"]
    assert "50 row(s)" in items[0].message
    assert "at least 200" in items[0].message

    path.write_text("id,answer\n" + "".join(f"{i},A\n" for i in range(200)))
    items = []
    _check_validation_sets(request, items)
    assert not items
