"""
One-command launcher for the Q-Safe Field Link demo.

    python -m qsafe_link.run                    # gateway + 1 device
    python -m qsafe_link.run --devices 6        # a fleet, for the correlator
    python -m qsafe_link.run --port 8000 --rate 3

Binds 0.0.0.0 by default so phones on the same Wi-Fi (or the laptop's
hotspot) can reach it, and prints the exact URLs and a QR page to open.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DEVICE_NAMES = [
    ("plant-01", "Pump House A"),
    ("plant-02", "Valve Station B"),
    ("plant-03", "Substation C"),
    ("plant-04", "Water Intake D"),
    ("plant-05", "Compressor E"),
    ("plant-06", "Feeder Line F"),
    ("plant-07", "Turbine G"),
    ("plant-08", "Tank Farm H"),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Q-Safe Field Link — live device-to-device demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 exposes to LAN)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--devices", type=int, default=1, help="pre-created simulated devices")
    p.add_argument("--rate", type=float, default=3.0, help="QKD rounds per second per device")
    p.add_argument(
        "--qubits",
        type=int,
        default=64,
        help="qubits per BB84 round (64 matches the training stream; lower values "
        "widen the benign QBER distribution and raise the false-escalation rate)",
    )
    p.add_argument(
        "--min-devices-alert",
        type=int,
        default=3,
        help="devices that must concur before a fleet campaign alert",
    )
    p.add_argument("--models", default=str(REPO_ROOT / "models"), help="model artifact directory")
    p.add_argument("--no-type-tagger", action="store_true", help="skip the attack-type classifier")
    p.add_argument(
        "--tls",
        action="store_true",
        help="serve HTTPS with a self-signed cert (enables phone accelerometer access)",
    )
    p.add_argument("--certfile", default="", help="TLS cert path (implies --tls)")
    p.add_argument("--keyfile", default="", help="TLS key path (implies --tls)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import uvicorn

    from qsafe_link.gateway import create_app
    from qsafe_link.recorder import SessionRecorder
    from qsafe_link.runtime import LinkRuntime

    print("[qsafe-link] loading models and resolving KEM backend...")
    runtime = LinkRuntime(
        models_dir=args.models,
        n_qubits_per_round=args.qubits,
        rounds_per_second=args.rate,
        min_devices_for_alert=args.min_devices_alert,
        enable_type_tagger=not args.no_type_tagger,
    )
    recorder = SessionRecorder().attach(runtime)

    n = max(0, min(args.devices, len(DEFAULT_DEVICE_NAMES)))
    for i in range(n):
        device_id, display = DEFAULT_DEVICE_NAMES[i]
        runtime.add_node(device_id, display, seed=1000 + i)
    if n:
        print(f"[qsafe-link] pre-created {n} device(s): "
              f"{', '.join(d for d, _ in DEFAULT_DEVICE_NAMES[:n])}")

    app = create_app(runtime, recorder=recorder, port=args.port)

    ssl_kwargs = {}
    if args.tls or args.certfile or args.keyfile:
        certfile, keyfile = args.certfile, args.keyfile
        if not (certfile and keyfile):
            certfile, keyfile = _ensure_self_signed_cert()
        ssl_kwargs = {"ssl_certfile": certfile, "ssl_keyfile": keyfile}
        print(f"[qsafe-link] TLS enabled ({certfile}). Phones will warn about the "
              "self-signed certificate — tap through it once.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl_kwargs)
    return 0


def _ensure_self_signed_cert() -> tuple[str, str]:
    """Generate a self-signed cert so DeviceMotion works on phones.

    Browsers gate the accelerometer behind a secure context, and a plain
    http:// LAN address is not one. The demo works without this -- the sensor
    page falls back to touch-driven telemetry -- so this stays optional.
    """
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    from qsafe_link.gateway import detect_lan_ip

    out_dir = REPO_ROOT / "demo_output" / "tls"
    out_dir.mkdir(parents=True, exist_ok=True)
    certfile, keyfile = out_dir / "cert.pem", out_dir / "key.pem"
    if certfile.exists() and keyfile.exists():
        return str(certfile), str(keyfile)

    ip = detect_lan_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Q-Safe Field Link")])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except ValueError:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    keyfile.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return str(certfile), str(keyfile)


if __name__ == "__main__":
    raise SystemExit(main())
