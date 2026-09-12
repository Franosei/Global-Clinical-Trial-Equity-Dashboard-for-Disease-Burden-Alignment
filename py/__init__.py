"""Application package.

The historical package name ``py`` collides with pytest's compatibility shim.
Expose that shim when pytest is installed so repository-local test discovery
continues to work without renaming the public module paths.
"""

from __future__ import annotations

import sys

try:  # pragma: no cover - only needed while pytest imports its legacy shim
    import _pytest._py.error as error
    import _pytest._py.path as path

    sys.modules.setdefault("py.error", error)
    sys.modules.setdefault("py.path", path)
except ImportError:  # pytest is not a runtime dependency
    pass
