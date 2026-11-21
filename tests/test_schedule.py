"""The files that run `check` on a schedule (macOS launchd): checked without installing them."""

import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "scheduled-check.sh"
PLIST = ROOT / "scripts" / "com.finscope.check.plist"


def test_the_plist_runs_the_script_every_day_at_eight() -> None:
    job = plistlib.loads(PLIST.read_bytes())

    assert job["Label"] == "com.finscope.check"
    assert job["ProgramArguments"] == ["/bin/zsh", "__FINSCOPE_DIR__/scripts/scheduled-check.sh"]
    assert job["StartCalendarInterval"] == {"Hour": 8, "Minute": 0}


def test_the_script_is_executable_and_runs_check_from_the_project_folder() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert SCRIPT.stat().st_mode & 0o111  # executable
    assert 'cd "${0:A:h}/.."' in text  # the project folder, wherever launchd starts it
    assert "data/logs/check.log" in text
    assert text.rstrip().endswith("run --quiet python -m finscope check")


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed (CI on Linux)")
def test_the_script_is_valid_zsh() -> None:
    subprocess.run(["zsh", "-n", str(SCRIPT)], check=True)
