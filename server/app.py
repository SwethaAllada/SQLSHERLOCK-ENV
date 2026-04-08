# Re-export from sqlsherlock_env/server/app.py for openenv validate compatibility.
# openenv expects server/app.py at the repo root.

import sys
import os

# Ensure sqlsherlock_env is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sqlsherlock_env"))

from server.app import app, main  # noqa: F401, E402

if __name__ == "__main__":
    main()
