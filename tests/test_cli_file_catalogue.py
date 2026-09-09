from pathlib import Path

from zevo.cli.zevo import _dataset_files


def test_dataset_files_recurses_and_ignores_catalogue_metadata(tmp_path: Path) -> None:
    (tmp_path / "source.json").write_text("{}")
    (tmp_path / ".profile.json").write_text("{}")
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train.json").write_text("[]")
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "val.json").write_text("[]")
    (tmp_path / "test" / "__pycache__").mkdir(parents=True)
    (tmp_path / "test" / "test_eval.py").write_text("print('ok')")
    (tmp_path / "test" / "__pycache__" / "test_eval.pyc").write_bytes(b"cache")

    assert [path.relative_to(tmp_path).as_posix() for path in _dataset_files(tmp_path)] == [
        "test/test_eval.py",
        "train/train.json",
        "validation/val.json",
    ]
