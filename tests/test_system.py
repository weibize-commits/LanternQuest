from pathlib import Path

from lanternquest.system import find_executable


def test_find_executable_uses_path(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "example.exe"
    executable.write_bytes(b"")
    monkeypatch.setenv("PATH", str(tmp_path))

    discovered = find_executable("example")
    assert discovered is not None
    assert Path(discovered) == executable
