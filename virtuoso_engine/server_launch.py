"""
server_launch.py — wrap EDA commands for the active server profile.

The Python orchestrator runs on the host and spawns Virtuoso and Spectre as
subprocesses.  On some servers those tools live inside a singularity container
that must be entered after loading environment modules.  A *server profile*
(the ``server:`` section of the merged config, selected via the ``RFIC_SERVER``
env var in config_loader.load_config) describes how to wrap each command:

    server:
      enabled: true
      pre_launch_commands:
        - "module purge"
        - "module load cadence/2025 keysight/2026 rfic-link/2026"
      container_command: >
        singularity exec --env PATH=$PATH ... /path/to/container.sif

``wrap_command`` turns a base argv such as ``["virtuoso", "-log", "x"]`` into

    ["bash", "-lc",
     "module purge && module load ... && singularity exec ... container.sif virtuoso -log x"]

so that the modules load and the tool runs inside the container.  When the
profile is disabled / empty (the default ``local`` server), the argv is
returned unchanged — byte-for-byte the original behaviour.

Env references like ``$PATH`` / ``$DISPLAY`` / ``$UID`` are left literal in the
YAML and expand at run time inside the ``bash -lc`` login shell (where the
``module`` function is also defined).
"""

from __future__ import annotations

import shlex
from typing import List, Optional


def wrap_command(argv: List[str], server: Optional[dict]) -> List[str]:
    """
    Wrap ``argv`` according to the server profile.

    Args:
        argv:   The base command, e.g. ``["spectre", "-64", netlist, ...]``.
        server: The ``server`` config dict (or None / {} for the local server).

    Returns:
        ``argv`` unchanged when the profile is disabled, otherwise a
        ``["bash", "-lc", <single command string>]`` list suitable for
        ``subprocess.Popen`` / ``subprocess.run``.
    """
    if not server or not server.get("enabled"):
        return argv

    parts: List[str] = []
    for cmd in server.get("pre_launch_commands", []) or []:
        cmd = str(cmd).strip()
        if cmd:
            parts.append(cmd)

    container = str(server.get("container_command", "") or "").strip()
    quoted = " ".join(shlex.quote(a) for a in argv)
    if container:
        parts.append(f"{container} {quoted}")
    else:
        # No container, only module/pre-launch setup: still run the command.
        parts.append(quoted)

    inner = " && ".join(parts)
    return ["bash", "-lc", inner]
