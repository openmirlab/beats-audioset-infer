"""Load verification-only guards without adding a production dependency.

The helper lives outside wheel data and controls environment-specific goldens.
Reads: tools/parity_support.py; never imports external model code automatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
