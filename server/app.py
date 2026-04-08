# Re-export from sqlsherlock_env/server/app.py for openenv validate compatibility.
# openenv expects server/app.py at the repo root.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sqlsherlock_env"))

from server.app import app  # noqa: F401, E402


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
