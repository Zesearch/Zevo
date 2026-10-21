"""Independent Validation warns on small samples and rejects empty sets."""
from types import SimpleNamespace

from zevo.api.routers.ui.preflight import _check_validation_sets


def test_independent_validation_size_warning_and_empty_blocker(tmp_path):
    path = tmp_path / "validation.csv"
    request = SimpleNamespace(validation_sets=[SimpleNamespace(name="medbullets-smoke", test_set=str(path))])

    path.write_text("id,answer\n" + "".join(f"{i},A\n" for i in range(50)))
    items = []
    _check_validation_sets(request, items)
    assert [item.code for item in items] == ["validation_set_1_too_small"]
    assert "50 row(s)" in items[0].message
    assert items[0].severity == "risk"
    assert "allowed" in items[0].message

    path.write_text("id,answer\n")
    items = []
    _check_validation_sets(request, items)
    assert [item.code for item in items] == ["validation_set_1_empty"]
    assert items[0].severity == "blocker"

    path.write_text("id,answer\n" + "".join(f"{i},A\n" for i in range(200)))
    items = []
    _check_validation_sets(request, items)
    assert not items
