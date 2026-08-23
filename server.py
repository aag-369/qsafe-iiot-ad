"""
Unified ASGI entrypoint: the research dashboard and the live demo, one service.

    uvicorn server:app --host 0.0.0.0 --port 8000

Routing
-------
    /            the existing research dashboard (web/backend/app.py)
    /link        the Q-Safe Field Link live demo (qsafe_link/gateway.py)
    /link/console, /link/node, /link/monitor, /link/

Both are complete FastAPI applications. They are mounted rather than merged
because they have genuinely different lifecycles -- the dashboard answers
request-shaped questions, while the demo owns a background control loop --
and merging their routers would entangle those. Mounting also means neither
file needed restructuring to make this work.

Why one service rather than two
-------------------------------
Free hosting tiers give you one web service. Running the demo as a second
service would either cost money or mean the demo is simply absent from the
deployed site. Mounting both in one process also loads TensorFlow once
instead of twice, which matters on a 512 MB instance.

Idle behaviour
--------------
The demo's control loop executes real Qiskit circuits continuously. On a
laptop during a demonstration that is the point; on an always-on hosted
instance with nobody watching it is waste. So on this entrypoint the demo
starts with **no devices** and pauses its loop when no browser is connected
(`QSAFE_LINK_IDLE_PAUSE_S`). A visitor presses "Start a demonstration" on the
console and it wakes immediately.

Environment
-----------
    PORT                        bound by the platform (default 8000)
    QSAFE_LINK_MOUNT            mount prefix for the demo (default "/link")
    QSAFE_LINK_ENABLED          "0" to serve the dashboard alone
    QSAFE_LINK_DEVICES          devices to pre-create (default 0)
    QSAFE_LINK_RATE             QKD rounds/second per device (default 2.0)
    QSAFE_LINK_QUBITS           qubits per BB84 round (default 64)
    QSAFE_LINK_IDLE_PAUSE_S     pause the loop after N idle seconds (default 120)
    QSAFE_LINK_TYPE_TAGGER      "0" to skip the attack-type classifier
"""

from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI  # noqa: E402


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


MOUNT = os.environ.get("QSAFE_LINK_MOUNT", "/link").rstrip("/") or "/link"

# --- the research dashboard ------------------------------------------------
from web.backend.app import app as dashboard_app  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run the mounted applications' own lifespans.

    Starlette does NOT propagate lifespan events into mounted sub-apps. Left
    alone, the dashboard would never run its warm-up and — far worse — the
    demo's control-loop worker and WebSocket broadcaster would never start,
    so every page would load and then sit there dead. Entering each child's
    lifespan context explicitly is what makes mounting behave like running
    the app directly.
    """
    async with AsyncExitStack() as stack:
        for child in _mounted_apps():
            await stack.enter_async_context(child.router.lifespan_context(child))
        _announce()
        yield


def _mounted_apps() -> list[FastAPI]:
    children = []
    if link_app is not None:
        children.append(link_app)
    children.append(dashboard_app)
    return children


app = FastAPI(title="Q-Safe IIoT-AD", version="1.1.0", lifespan=lifespan)

# --- the live demo ---------------------------------------------------------
link_app = None
if _env_flag("QSAFE_LINK_ENABLED", True):
    from qsafe_link.gateway import create_app as create_link_app  # noqa: E402
    from qsafe_link.recorder import SessionRecorder  # noqa: E402
    from qsafe_link.runtime import LinkRuntime  # noqa: E402

    link_runtime = LinkRuntime(
        models_dir=str(REPO_ROOT / "models"),
        n_qubits_per_round=_env_int("QSAFE_LINK_QUBITS", 64),
        rounds_per_second=_env_float("QSAFE_LINK_RATE", 2.0),
        min_devices_for_alert=_env_int("QSAFE_LINK_MIN_DEVICES_ALERT", 3),
        enable_type_tagger=_env_flag("QSAFE_LINK_TYPE_TAGGER", True),
        idle_pause_after_s=_env_float("QSAFE_LINK_IDLE_PAUSE_S", 120.0),
    )
    # Constructing the runtime is cheap: models and liboqs load on the first
    # request that reaches the demo, not at import.
    link_recorder = SessionRecorder().attach(link_runtime)

    n_devices = _env_int("QSAFE_LINK_DEVICES", 0)
    if n_devices > 0:
        from qsafe_link.gateway import DEMO_DEVICE_NAMES

        for i, (device_id, display) in enumerate(DEMO_DEVICE_NAMES[:n_devices]):
            link_runtime.add_node(device_id, display, seed=1000 + i)

    link_app = create_link_app(
        link_runtime,
        recorder=link_recorder,
        port=_env_int("PORT", 8000),
    )
    # Mount the demo BEFORE the dashboard: the dashboard mounts StaticFiles at
    # its own root and would otherwise swallow every path.
    app.mount(MOUNT, link_app, name="qsafe-link")

app.mount("/", dashboard_app, name="dashboard")


def _announce() -> None:
    port = _env_int("PORT", 8000)
    print("=" * 68)
    print("  Q-Safe IIoT-AD — dashboard + live demo")
    print("=" * 68)
    print(f"  Research dashboard : http://localhost:{port}/")
    if link_app is not None:
        print(f"  Field Link console : http://localhost:{port}{MOUNT}/console")
        print(f"  Field sensor       : http://localhost:{port}{MOUNT}/node")
        print(f"  Control room       : http://localhost:{port}{MOUNT}/monitor")
        print(f"  Join / QR codes    : http://localhost:{port}{MOUNT}/")
        print(f"\n  Demo devices start: {_env_int('QSAFE_LINK_DEVICES', 0)} "
              f"(press 'Start a demonstration' on the console to bring some up)")
    else:
        print("  Field Link demo    : disabled (QSAFE_LINK_ENABLED=0)")
    print("=" * 68)
