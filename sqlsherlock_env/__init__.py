# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SQLSherlock-Env — RL environment for AI data scientist agents."""

try:
    # When running from within sqlsherlock_env/ (PYTHONPATH=.)
    from client import SQLSherlockEnv
    from models import SQLSherlockAction, SQLSherlockObservation, SQLSherlockState
except ImportError:
    # When imported as a package (import sqlsherlock_env)
    from sqlsherlock_env.client import SQLSherlockEnv
    from sqlsherlock_env.models import SQLSherlockAction, SQLSherlockObservation, SQLSherlockState

__version__ = "1.0.0"

__all__ = [
    "SQLSherlockEnv",
    "SQLSherlockAction",
    "SQLSherlockObservation",
    "SQLSherlockState",
]
