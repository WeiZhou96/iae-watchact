"""The released per-request results reproduce Tables 1-4 of the paper."""
import subprocess, sys
from pathlib import Path


def test_release_matches_paper():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'analysis' / 'check_release.py'
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
