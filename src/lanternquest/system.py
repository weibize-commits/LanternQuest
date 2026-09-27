import os
import shutil
from pathlib import Path


def find_executable(name: str) -> str | None:
    """Find an executable, including a freshly installed WinGet package."""
    discovered = shutil.which(name)
    if discovered:
        return discovered

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None

    winget_root = Path(local_app_data) / "Microsoft" / "WinGet"
    candidates = [
        winget_root / "Links" / f"{name}.exe",
        *sorted((winget_root / "Packages").glob(f"Gyan.FFmpeg*/**/{name}.exe")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None

