"""
FastAPI gateway for the Q-Safe Field Link demo.

Serves three browser clients over the LAN and brokers the protected link
between them:

    /node     phone -- the field sensor's HMI; produces application telemetry
    /monitor  phone or smartwatch -- the control-room display
    /console  laptop/projector -- the operator console

The gateway is the *peer endpoint* of each device's protected link. Uplink
telemetry arriving from a sensor page is sealed through that device's
`QSafeSession` before it is relayed, and the monitor receives both the
decrypted payload and a preview of the ciphertext that actually crossed the
link -- so an observer can see, side by side, what the authorized endpoint
reads and what an interceptor would have captured.

Nothing in the control loop blocks on a network client: node stepping runs
on `LinkRuntime`'s worker thread and this module only drains its event
deque. A phone dropping off Wi-Fi cannot stall the physics.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qkd_sim.qber_stream_multiclass import ATTACK_TYPE_NAMES

from .recorder import SessionRecorder
from .runtime import LinkRuntime
from .scenarios import SCENARIOS, apply_scenario, resolve_attack_type

STATIC_DIR = Path(__file__).resolve().parent / "static"

BROADCAST_HZ = 8.0
SNAPSHOT_EVERY_N_BROADCASTS = 4  # full state snapshot ~2x/second

# Names used when a visitor starts a demonstration from the console's empty
# state. Kept here rather than in run.py so both entry points agree.
DEMO_DEVICE_NAMES = [
    ("plant-01", "Pump House A"),
    ("plant-02", "Valve Station B"),
    ("plant-03", "Substation C"),
    ("plant-04", "Water Intake D"),
    ("plant-05", "Compressor E"),
    ("plant-06", "Feeder Line F"),
]
# A phone that walks out of Wi-Fi range stays TCP-writable for a long time
# before it errors. Without a deadline, awaiting that one send blocks the
# fan-out to every other screen -- including the projector.
SEND_TIMEOUT_S = 2.0


def detect_lan_ip() -> str:
    """Best-effort LAN address for building the QR-code URLs."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        # No packet is actually sent for UDP connect; this just asks the OS
        # which interface would be used to reach the outside world.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


class ConnectionHub:
    """Tracks every connected WebSocket so broadcasts can fan out."""

    def __init__(self) -> None:
        self.consoles: set[WebSocket] = set()
        self.monitors: dict[str, set[WebSocket]] = {}
        self.sensors: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def add_console(self, ws: WebSocket) -> None:
        async with self._lock:
            self.consoles.add(ws)

    async def drop_console(self, ws: WebSocket) -> None:
        async with self._lock:
            self.consoles.discard(ws)

    async def add_monitor(self, device_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self.monitors.setdefault(device_id, set()).add(ws)

    async def drop_monitor(self, device_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self.monitors.get(device_id, set()).discard(ws)

    @staticmethod
    async def _send(ws: WebSocket, payload: str) -> bool:
        """Send with a deadline. Returns False if the socket should be dropped."""
        try:
            await asyncio.wait_for(ws.send_text(payload), timeout=SEND_TIMEOUT_S)
            return True
        except (asyncio.TimeoutError, Exception):
            return False

    async def broadcast_console(self, message: dict) -> None:
        payload = json.dumps(message)
        async with self._lock:
            targets = list(self.consoles)
        if not targets:
            return
        # Concurrently, so the slowest client sets the latency rather than
        # the sum of all of them.
        results = await asyncio.gather(
            *(self._send(ws, payload) for ws in targets), return_exceptions=True
        )
        for ws, ok in zip(targets, results):
            if ok is not True:
                await self.drop_console(ws)

    async def send_monitors(self, device_id: str, message: dict) -> None:
        payload = json.dumps(message)
        async with self._lock:
            targets = list(self.monitors.get(device_id, set())) + list(
                self.monitors.get("*", set())
            )
        if not targets:
            return
        results = await asyncio.gather(
            *(self._send(ws, payload) for ws in targets), return_exceptions=True
        )
        # Drop unresponsive monitors rather than retrying them every tick.
        for ws, ok in zip(targets, results):
            if ok is not True:
                await self.drop_monitor(device_id, ws)


class AttackRequest(BaseModel):
    device_id: str | None = None
    attack_type: str = Field("benign", description="benign | eavesdrop | jamming | pns")
    intensity: float = Field(0.5, ge=0.0, le=1.0)


class ScenarioRequest(BaseModel):
    key: str
    device_id: str | None = None


class DeviceRequest(BaseModel):
    device_id: str
    display_name: str = ""
    role: str = "field-sensor"


class CommandRequest(BaseModel):
    device_id: str
    command: str
    value: float | str | None = None


def create_app(
    runtime: LinkRuntime,
    recorder: SessionRecorder | None = None,
    port: int = 8000,
) -> FastAPI:
    app = FastAPI(title="Q-Safe Field Link", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    hub = ConnectionHub()
    app.state.runtime = runtime
    app.state.hub = hub
    app.state.recorder = recorder
    app.state.lan_ip = detect_lan_ip()
    app.state.port = port

    # --- background fan-out -------------------------------------------------
    async def broadcaster() -> None:
        """Drain the runtime's event deque and push to every console.

        Also relays each device's traffic and rekey events to that device's
        monitors, so a watch showing one device does not have to receive the
        whole fleet's telemetry.
        """
        counter = 0
        interval = 1.0 / BROADCAST_HZ
        while True:
            try:
                # An open console counts as someone watching, so the control
                # loop keeps stepping for as long as any client is attached
                # and pauses once they all leave.
                has_clients = bool(
                    hub.consoles
                    or any(hub.monitors.values())
                    or hub.sensors
                )
                if has_clients:
                    runtime.note_client_activity()
                else:
                    # Nobody to send to. Skip the whole cycle rather than
                    # building snapshots for an empty room -- `build_snapshot`
                    # reaches into `backend_state()`, which would load
                    # TensorFlow on an instance that has never served the demo.
                    await asyncio.sleep(interval)
                    continue

                events = []
                while runtime.events:
                    events.append(runtime.events.popleft())

                if events:
                    await hub.broadcast_console({"type": "events", "events": events})
                    for ev in events:
                        dev = ev.get("data", {}).get("device_id")
                        if dev and ev["kind"] in ("rekey", "frame", "type_change"):
                            await hub.send_monitors(dev, {"type": "event", "event": ev})

                counter += 1
                if counter % SNAPSHOT_EVERY_N_BROADCASTS == 0:
                    snapshot = build_snapshot(runtime)
                    await hub.broadcast_console({"type": "snapshot", **snapshot})
                    for node_state in snapshot["devices"]:
                        await hub.send_monitors(
                            node_state["device_id"],
                            {"type": "device_state", "device": node_state},
                        )
            except Exception as exc:  # a broadcast failure must not kill the loop
                print(f"[qsafe-link] broadcaster error: {exc}")
            await asyncio.sleep(interval)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.start()
        _app.state.broadcast_task = asyncio.create_task(broadcaster())
        _print_banner()
        try:
            yield
        finally:
            _app.state.broadcast_task.cancel()
            runtime.stop()

    app.router.lifespan_context = lifespan

    def _print_banner() -> None:
        url = f"http://{app.state.lan_ip}:{app.state.port}"
        print("\n" + "=" * 68)
        print("  Q-Safe Field Link is up")
        print("=" * 68)
        print(f"  Operator console : {url}/console")
        print(f"  Field sensor     : {url}/node      (phone)")
        print(f"  Control room     : {url}/monitor   (phone or watch)")
        print(f"  Join / QR codes  : {url}/")
        # Deliberately does not force the model load. Printing a banner is
        # not a reason to spend several seconds and a few hundred megabytes
        # importing TensorFlow on an instance that may never serve the demo.
        bs = runtime.backend_state(force_load=False)
        if bs["models_loaded"]:
            print(f"\n  KEM backend      : {bs['kem_backend']} "
                  f"({'REAL liboqs' if bs['using_real_liboqs'] else 'SIMULATED — labelled in UI'})")
            print(f"  Detector         : {bs['detector_backend']} "
                  f"(threshold {bs['detector_threshold']:.2f}, window {bs['detector_window']})")
            print(f"  Handshake cost   : BIKE-L1 {bs['bike_reference_ms']:.2f} ms  |  "
                  f"HQC-128 {bs['hqc_reference_ms']:.2f} ms  (measured on this host)")
        else:
            print("\n  Models + KEM backend load on the first request that "
                  "reaches the demo.")
            print(f"  liboqs available : {bs['liboqs_available']}")
        if bs["idle_pause_after_s"]:
            print(f"  Idle pause       : control loop stops after "
                  f"{bs['idle_pause_after_s']:.0f}s with no browser connected")
        print("=" * 68 + "\n")

    # --- pages --------------------------------------------------------------
    def page(name: str, base: str = "") -> HTMLResponse:
        path = STATIC_DIR / name
        if not path.exists():
            raise HTTPException(404, f"missing page {name}")
        # Pages are templated rather than served straight off disk so that
        # every asset, API call and WebSocket URL is rooted at whatever prefix
        # this app is mounted under. Served standalone the prefix is empty;
        # mounted beside the research dashboard it is "/link". Hard-coded
        # absolute paths would work in exactly one of those two cases.
        html = path.read_text(encoding="utf-8").replace("__BASE__", base)
        return HTMLResponse(html)

    def _base(request: Request) -> str:
        return (request.scope.get("root_path") or "").rstrip("/")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return page("index.html", _base(request))

    @app.get("/console", response_class=HTMLResponse)
    def console(request: Request):
        return page("console.html", _base(request))

    @app.get("/node", response_class=HTMLResponse)
    def node_page(request: Request):
        return page("node.html", _base(request))

    @app.get("/monitor", response_class=HTMLResponse)
    def monitor_page(request: Request):
        return page("monitor.html", _base(request))

    @app.get("/replay", response_class=HTMLResponse)
    def replay_page(request: Request):
        return page("replay.html", _base(request))

    # --- REST ---------------------------------------------------------------
    @app.get("/api/health")
    def health():
        # Deliberately does not force the model load: a hosting platform
        # polls this every few seconds, and waking TensorFlow for a health
        # check on an instance nobody is using is pure waste.
        backend = runtime.backend_state(force_load=False)
        return {
            "status": "ok" if backend.get("worker_alive", True) else "worker_stopped",
            "lan_ip": app.state.lan_ip,
            "port": app.state.port,
            **backend,
        }

    @app.get("/api/state")
    def state():
        return build_snapshot(runtime)

    @app.get("/api/scenarios")
    def scenarios():
        return {"scenarios": [s.as_dict() for s in SCENARIOS.values()]}

    @app.post("/api/devices")
    def add_device(req: DeviceRequest):
        node = runtime.add_node(req.device_id, req.display_name, req.role)
        return node.state()

    @app.post("/api/start-demo")
    def start_demo(n: int = 3):
        """Bring up a small simulated fleet on demand.

        A hosted deployment starts with no devices, so an unattended instance
        is not executing quantum circuits for nobody. This is what the
        console's empty state calls when a visitor asks for a demonstration.
        """
        n = max(1, min(n, len(DEMO_DEVICE_NAMES)))
        existing = {node.device_id for node in runtime.node_list()}
        created = []
        for i, (device_id, display) in enumerate(DEMO_DEVICE_NAMES[:n]):
            if device_id in existing:
                continue
            runtime.add_node(device_id, display, seed=1000 + i)
            created.append(device_id)
        runtime.note_client_activity()
        return {"created": created, "n_devices": len(runtime.node_list())}

    @app.delete("/api/devices/{device_id}")
    def remove_device(device_id: str):
        if not runtime.remove_node(device_id):
            raise HTTPException(404, f"no such device {device_id}")
        return {"removed": device_id}

    @app.post("/api/attack")
    def set_attack(req: AttackRequest):
        try:
            attack = resolve_attack_type(req.attack_type)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        targets = (
            [runtime.get(req.device_id)] if req.device_id else runtime.node_list()
        )
        targets = [t for t in targets if t is not None]
        if not targets:
            raise HTTPException(404, "no matching device")
        for node in targets:
            node.set_attack(attack, req.intensity)
        runtime.emit(
            "attack_set",
            {
                "devices": [n.device_id for n in targets],
                "attack_type": req.attack_type,
                "intensity": req.intensity,
            },
        )
        return {
            "applied_to": [n.device_id for n in targets],
            "attack_type": req.attack_type,
            "intensity": req.intensity,
        }

    @app.post("/api/scenario")
    def run_scenario(req: ScenarioRequest):
        try:
            return apply_scenario(runtime, req.key, req.device_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/command")
    async def send_command(req: CommandRequest):
        """Send an authenticated command down the protected link."""
        node = runtime.get(req.device_id)
        if node is None:
            raise HTTPException(404, f"no such device {req.device_id}")
        frame = node.send_downlink({"command": req.command, "value": req.value})
        runtime.emit("downlink", frame)
        sensor = hub.sensors.get(req.device_id)
        if sensor is not None:
            try:
                await sensor.send_text(json.dumps({"type": "command", "frame": frame}))
            except Exception:
                pass
        return frame

    @app.get("/api/report")
    def report():
        if recorder is None:
            raise HTTPException(404, "no recorder attached")
        return recorder.build_report(runtime)

    @app.get("/api/qr")
    def qr(request: Request, path: str = "/node"):
        """QR code for a client page, so people can join by camera."""
        url = join_urls(request)["base"] + path
        try:
            import io

            import qrcode
            import qrcode.image.svg

            img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage)
            buf = io.BytesIO()
            img.save(buf)
            return Response(content=buf.getvalue(), media_type="image/svg+xml")
        except Exception as exc:
            # QR generation is a convenience, never a hard dependency: the
            # join page always shows the URL as text as well.
            return JSONResponse({"url": url, "qr_error": str(exc)}, status_code=200)

    @app.get("/api/join-urls")
    def join_urls(request: Request):
        """Absolute URLs for the join page and its QR codes.

        Built from the request's own host and scheme rather than the detected
        LAN address whenever the app is reached through a hostname -- on a
        hosted deployment the container's private IP is useless to a phone.
        """
        prefix = _base(request)
        host = request.headers.get("host", "")
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
        if host and not host.startswith("0.0.0.0"):
            root = f"{scheme}://{host}{prefix}"
        else:
            root = f"http://{app.state.lan_ip}:{app.state.port}{prefix}"
        return {
            "base": root,
            "console": f"{root}/console",
            "node": f"{root}/node",
            "monitor": f"{root}/monitor",
        }

    # --- WebSockets ---------------------------------------------------------
    @app.websocket("/ws/console")
    async def ws_console(ws: WebSocket):
        await ws.accept()
        await hub.add_console(ws)
        runtime.note_client_activity()
        try:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "hello",
                        "backend": runtime.backend_state(),
                        "scenarios": [s.as_dict() for s in SCENARIOS.values()],
                        **build_snapshot(runtime),
                    }
                )
            )
            while True:
                await ws.receive_text()  # console is read-mostly; REST drives control
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await hub.drop_console(ws)

    @app.websocket("/ws/sensor/{device_id}")
    async def ws_sensor(ws: WebSocket, device_id: str):
        """A phone acting as the field sensor for `device_id`.

        The device is created on first connection, so a judge scanning the QR
        code and picking a name is enough to bring a new node online.
        """
        await ws.accept()
        runtime.note_client_activity()
        node = runtime.get(device_id) or runtime.add_node(device_id, device_id)
        node.connected_sensor = True
        hub.sensors[device_id] = ws
        try:
            await ws.send_text(
                json.dumps({"type": "welcome", "device": node.state(),
                            "backend": runtime.backend_state()})
            )
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "telemetry":
                    payload = msg.get("payload", {})
                    try:
                        frame = node.send_uplink(payload)
                    except Exception as exc:
                        # Never tear down a phone's socket over one frame:
                        # reconnecting takes ~500 ms, and the moment this is
                        # most likely is the instant of a rekey, which is
                        # exactly the beat the demo is built around.
                        node.metrics.record_rejected()
                        runtime.emit("frame_rejected",
                                     {"device_id": device_id, "error": str(exc)})
                        continue
                    runtime.emit("frame", frame)
                    await hub.send_monitors(device_id, {"type": "frame", "frame": frame})
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "ack",
                                "seq": frame["seq"],
                                "epoch": frame["epoch"],
                                "profile": frame["profile"],
                                "key_fingerprint": frame["key_fingerprint"],
                                "plaintext_bytes": frame["plaintext_bytes"],
                                "ciphertext_bytes": frame["ciphertext_bytes"],
                                "ciphertext_preview": frame["ciphertext_preview"],
                            }
                        )
                    )
                elif msg.get("type") == "rename":
                    node.display_name = str(msg.get("display_name", node.display_name))[:40]
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            node.connected_sensor = False
            hub.sensors.pop(device_id, None)

    @app.websocket("/ws/monitor/{device_id}")
    async def ws_monitor(ws: WebSocket, device_id: str):
        """A phone or watch displaying `device_id` (or "*" for the fleet)."""
        await ws.accept()
        await hub.add_monitor(device_id, ws)
        runtime.note_client_activity()
        node = runtime.get(device_id)
        if node is not None:
            node.connected_monitor = True
        try:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "welcome",
                        "device": node.state() if node else None,
                        "backend": runtime.backend_state(),
                    }
                )
            )
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await hub.drop_monitor(device_id, ws)
            if node is not None:
                node.connected_monitor = False

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def build_snapshot(runtime: LinkRuntime, history: int = 140) -> dict:
    devices = []
    for node in runtime.node_list():
        st = node.state()
        st["history"] = node.recent(history)
        devices.append(st)
    detections = []
    detection_summary = {}
    recorder = getattr(runtime, "recorder", None)
    if recorder is not None:
        # Measured on the server (attack switched on -> HQC-128 live), so the
        # figure is the same no matter which client triggered the attack or
        # when a console connected.
        detections = recorder.detections[-25:]
        detection_summary = {
            "n_episodes": len(recorder.detections),
            "n_spurious_escalations": recorder.spurious_escalations,
            "n_unmeasurable_episodes": recorder.unmeasurable_episodes,
            "n_re_acquisitions": recorder.re_acquisitions,
        }

    return {
        "ts": time.time(),
        "detections": detections,
        "detection_summary": detection_summary,
        "fleet": runtime.fleet_state(),
        "backend": runtime.backend_state(),
        "devices": devices,
        "attack_types": list(ATTACK_TYPE_NAMES.values()),
    }
