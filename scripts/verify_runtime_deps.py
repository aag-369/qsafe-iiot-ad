"""
Check that the deployed image's dependency list covers everything the
deployed entrypoint actually imports.

Why this exists: CI installs the *developer* requirements, so it will happily
pass while the Docker image is missing a package. The failure then appears
only as the container exiting on boot, several minutes into a deploy, with the
traceback buried in a hosting dashboard. That is exactly how `cryptography`
went missing -- `secure_channel/session.py` needs it for HKDF, it was listed in
demo/requirements.txt, and the image installed a hand-picked subset that left
it out.

This walks the import graph of `server.py` statically (no imports executed, so
it runs anywhere) and compares it against the union of the requirements files
the Dockerfile installs.

    python scripts/verify_runtime_deps.py

Exits non-zero if the image would be missing something.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Requirements files installed by web/Dockerfile.
INSTALLED_REQUIREMENTS = [
    "requirements.txt",
    "web/requirements.txt",
    "demo/requirements-runtime.txt",
]

# First-party packages -- not dependencies.
LOCAL = {
    "server", "web", "qsafe_link", "secure_channel", "demo", "scripts",
    "crypto_agility", "qkd_sim", "ai_detector", "orchestrator", "fleet", "tests",
}

# Distribution name -> the module name it actually provides.
DISTRIBUTION_TO_MODULE = {
    "tensorflow-cpu": "tensorflow",
    "tensorflow": "tensorflow",
    "scikit-learn": "sklearn",
    "liboqs-python": "oqs",
    "qiskit-aer": "qiskit_aer",
    "uvicorn[standard]": "uvicorn",
    "pillow": "PIL",
}

# Imports guarded by try/except with a working fallback, so their absence
# degrades behaviour rather than breaking the service. Each needs a reason.
OPTIONAL = {
    # qsafe_link/detector.py falls back to tf.lite.Interpreter, which ships
    # with TensorFlow.
    "ai_edge_litert": "falls back to tf.lite.Interpreter",
    # qsafe_link/gateway.py renders plain URLs when QR generation is absent.
    "qrcode": "join page degrades to plain URLs",
    # crypto_agility/kem_backend.py falls back to the simulated backend.
    "oqs": "falls back to the simulated KEM backend",
}


def declared_modules() -> set[str]:
    mods: set[str] = set()
    for rel in INSTALLED_REQUIREMENTS:
        path = REPO_ROOT / rel
        if not path.exists():
            raise SystemExit(f"FAIL: {rel} is referenced by the Dockerfile but missing")
        for raw in path.read_text().splitlines():
            line = raw.split("#")[0].strip()
            if not line or line.startswith("-r"):
                continue
            name = re.split(r"[<>=!~;\s]", line)[0].strip().lower()
            if not name:
                continue
            mods.add(DISTRIBUTION_TO_MODULE.get(name, name.replace("-", "_")))
    # Pulled in transitively by fastapi/uvicorn, and imported directly.
    mods |= {"starlette", "pydantic", "websockets"}
    return mods


def imported_modules() -> dict[str, set[str]]:
    """Top-level third-party module -> the files that import it."""
    targets = [REPO_ROOT / "server.py", REPO_ROOT / "web" / "backend" / "app.py"]
    for pkg in ("qsafe_link", "secure_channel", "crypto_agility", "qkd_sim",
                "ai_detector", "orchestrator", "fleet"):
        targets += sorted((REPO_ROOT / pkg).rglob("*.py"))

    found: dict[str, set[str]] = {}
    for path in targets:
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise SystemExit(f"FAIL: cannot parse {path}: {exc}")
        rel = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in LOCAL or n in sys.stdlib_module_names:
                    continue
                found.setdefault(n, set()).add(rel)
    return found


def main() -> int:
    declared = declared_modules()
    imported = imported_modules()

    print("Deployed entrypoint imports:", ", ".join(sorted(imported)))
    print("Image provides:            ", ", ".join(sorted(declared)))

    missing = {m: f for m, f in imported.items() if m not in declared}
    hard = {m: f for m, f in missing.items() if m not in OPTIONAL}
    soft = {m: f for m, f in missing.items() if m in OPTIONAL}

    if soft:
        print("\nOptional, absent by design:")
        for m, files in sorted(soft.items()):
            print(f"  {m}: {OPTIONAL[m]}")

    if hard:
        print("\nFAIL: the deployed image would be missing these packages:\n")
        for m, files in sorted(hard.items()):
            shown = ", ".join(sorted(files)[:3])
            print(f"  {m}  <- imported by {shown}")
        print(
            "\nAdd them to demo/requirements-runtime.txt (installed by "
            "web/Dockerfile).\nCI passes without them because it installs the "
            "full developer requirements;\nthe container would exit on boot."
        )
        return 1

    print("\nOK — every hard dependency of the deployed entrypoint is installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
