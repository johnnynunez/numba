"""Capture actual imports of the untouched 0.64.0 source with real NumPy."""
import json
import subprocess
import sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
p = subprocess.run([sys.executable, '-c', 'import numpy; print("NUMPY="+numpy.__version__,flush=True); import numba'], cwd=source, capture_output=True, text=True)
r = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, source=str(source))
out.write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
if 'NUMPY=2.5.1' in p.stdout:
    assert p.returncode != 0
    assert 'Numba needs NumPy 2.4 or less. Got NumPy 2.5.' in p.stderr
else:
    # Not success: matching NumPy only removes the first gate.
    assert 'Numba needs NumPy 2.4 or less' not in p.stderr
