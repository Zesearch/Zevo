from pathlib import Path

from zevo.holdout_storage import protect_asset, private_mirror, resolve_asset


def test_managed_test_asset_moves_to_private_root(tmp_path, monkeypatch):
    files = tmp_path / "data" / "files"
    private = tmp_path / "private"
    source = files / "capybara" / "test" / "test.json"
    source.parent.mkdir(parents=True)
    source.write_text('[{"prompt":"secret","response":"answer"}]')

    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_UPLOAD_ROOT", str(tmp_path / "data" / "uploads"))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(private))

    logical = protect_asset(str(source))
    mirror = private_mirror(logical)

    assert logical == str(source)
    assert not source.exists()
    assert mirror == private / "files" / "capybara" / "test" / "test.json"
    assert mirror.read_text() == '[{"prompt":"secret","response":"answer"}]'
    assert resolve_asset(logical) == str(mirror)


def test_protection_is_idempotent(tmp_path, monkeypatch):
    files = tmp_path / "files"
    private = tmp_path / "private"
    source = files / "bundle" / "test.csv"
    source.parent.mkdir(parents=True)
    source.write_text("question,answer\nq,a\n")
    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_UPLOAD_ROOT", str(tmp_path / "uploads"))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(private))

    assert protect_asset(str(source)) == str(source)
    assert protect_asset(str(source)) == str(source)
    assert Path(resolve_asset(str(source))).read_text() == "question,answer\nq,a\n"


def test_repo_relative_managed_path_is_protected(tmp_path, monkeypatch):
    files = tmp_path / "data" / "files"
    source = files / "bundle" / "test" / "test.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"question":"q","answer":"a"}\n')
    monkeypatch.setenv("ZEVO_FILES_DIR", str(files))
    monkeypatch.setenv("ZEVO_UPLOAD_ROOT", str(tmp_path / "data" / "uploads"))
    monkeypatch.setenv("ZEVO_HOLDOUT_ROOT", str(tmp_path / "private"))

    logical = "data/files/bundle/test/test.jsonl"
    assert protect_asset(logical) == logical
    assert not source.exists()
    assert Path(resolve_asset(logical)).read_text() == '{"question":"q","answer":"a"}\n'
