import os
import sys

# Make `src` importable as a package from anywhere under the project.
_ROOT = os.path.dirname(__file__)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
