# Committing and deploying this change

Everything is already written into your working tree, so GitHub Desktop should
show it as pending changes with nothing left to move by hand.

---

## 1. What GitHub Desktop will show you

**New files** (44):

```
server.py                        unified entrypoint: dashboard at /, demo at /link
scripts/verify_kem_ordering.py   build-time gate on the KEM cost ordering

secure_channel/                  KEM secret -> HKDF -> AES-256-GCM, re-keyed on switch
  __init__.py  aead.py  session.py  metrics.py

qsafe_link/                      the live demo runtime
  __init__.py  channel.py  detector.py  node.py  runtime.py
  gateway.py   scenarios.py  recorder.py  run.py
  static/      console.html  node.html  monitor.html  index.html
               replay.html   shared.css  qsafe.js

demo/                            tooling and documentation
  README.md  RUNBOOK.md  ARCHITECTURE_MAPPING.md  COMMIT_GUIDE.md
  preflight.py  record_demo.py  requirements.txt  Dockerfile  __init__.py

demo_output/                     the offline evidence pack
  demo_replay.html  demo_report.json  charts/*.png

tests/test_secure_channel.py     23 tests
tests/test_qsafe_link.py         45 tests
```

**Modified files** (6):

| File | Change |
|---|---|
| `crypto_agility/kem_backend.py` | Recalibrated the simulated fallback's reference timings, which had BIKE-L1 and HQC-128 **inverted**. See § 4. |
| `render.yaml` | Points at `server:app`; adds the demo's environment variables. |
| `web/Dockerfile` | Installs `qrcode`, runs the KEM ordering gate, starts `server:app`. |
| `web/frontend/index.html` | Nav entry, hero button, and a Field Link section linking to `/link/console`. |
| `.github/workflows/ci.yml` | Runs the new tests, the KEM gate, and a smoke test of the deployed entrypoint. |
| `README.md`, `.gitignore` | Documents the demo; keeps the evidence pack, ignores the bulky event trace. |

`demo_output/demo_session.json` (~1 MB) is gitignored — it is fully
regenerable with `python -m demo.record_demo` and adds nothing to review.

### Two housekeeping notes

Your default branch is **`master`**, so the CI workflow triggers on both
`main` and `master` — otherwise it would have sat there never running.

`QSFIN/` and `qsafe-iiot-ad/` are empty nested clones sitting inside the
working tree (each contains only `.git/` and `.gitattributes`). Committing
them would add broken gitlink entries rather than files, so `.gitignore` now
excludes them and they will disappear from GitHub Desktop's list. Delete them
locally if you do not need them. There is also a zero-byte `.write_test` file
left over from a permission check — it is ignored, and safe to delete.

---

## 2. Before you commit

```bash
python -m pytest tests/ -q          # expect 121 passed
python scripts/verify_kem_ordering.py
```

Both should pass on your machine. If `verify_kem_ordering.py` reports the
simulated backend rather than `LiboqsKEMBackend`, that is only because liboqs
is not built locally — it will be built inside the Docker image.

---

## 3. Committing in GitHub Desktop

One commit is fine; the change is coherent. Suggested message:

> **Add Q-Safe Field Link: live device-to-device demonstration**
>
> Adds `secure_channel/` (KEM shared secret -> HKDF-SHA256 -> AES-256-GCM,
> re-keyed on every crypto-agility profile switch) and `qsafe_link/` (the
> per-node control loop closed in real time against phones over Wi-Fi, with
> phone / smartwatch / projector clients).
>
> Deploys as one service: dashboard at /, demo at /link.
>
> Also fixes inverted reference timings in the simulated KEM fallback, which
> would have reported a negative CPU saving on any host without liboqs.
>
> 68 new tests; 121 total.

Then **Push origin**.

If you would rather split it: commit `crypto_agility/kem_backend.py` and
`scripts/verify_kem_ordering.py` first as a standalone fix — it stands on its
own and is the one change that touches previously published behaviour.

---

## 4. The one change to existing behaviour

`crypto_agility/kem_backend.py` carried reference timings for its
*simulated* fallback with the two profiles inverted — BIKE-L1 at 1.85 ms and
HQC-128 at 0.47 ms, i.e. the low-overhead baseline priced at roughly four
times the hardened profile.

This never affected your published results: `models/benchmark_report.json`
records `liboqs_available: true`, and back-solving it gives ≈0.78 ms for
BIKE-L1 and ≈7.34 ms for HQC-128, consistent with the real library. But on any
machine without liboqs built, the live benchmark would have reported a
**negative** CPU saving — inverting the framework's central claim, on screen.

The table is now calibrated to this project's own measured liboqs values, with
the provenance recorded at the definition site, and
`scripts/verify_kem_ordering.py` fails the build if it ever inverts again.

No committed metric, model artifact, or reported number changes.

---

## 5. After Render redeploys

Render picks up the push automatically. The first build is slow (TensorFlow
plus liboqs from source); later deploys reuse layer caching.

Watch the build log for:

```
KEM backend: LiboqsKEMBackend (liboqs available: True)
  BIKE-L1      0.7xx ms / handshake
  HQC-128      8.xxx ms / handshake
OK — baseline is ~11x cheaper than hardened.
```

If that step fails, the build stops — deliberately. An image whose baseline
profile costs more than its hardened one would invert every saving figure.

Then check, in order:

1. `https://<your-app>.onrender.com/` — the dashboard, with a **Field Link**
   entry in the nav and an "Open the Field Link demo" button in the hero.
2. `https://<your-app>.onrender.com/link/api/health` — should show
   `"worker_alive": true` and `"models_loaded": false`.
3. `https://<your-app>.onrender.com/link/console` — the console. It starts
   empty; press **Start a demonstration**. First load is slow (models are
   loading); after that, scenario buttons should work.

### If something looks wrong

| Symptom | Cause |
|---|---|
| `/link/console` is 404 | `QSAFE_LINK_ENABLED` is `0`, or Render did not pick up `render.yaml` — re-sync the blueprint. |
| Page loads but nothing moves | `worker_alive: false`. Check the deploy log for an exception at startup. |
| Amber "SIMULATED KEM" pill | The liboqs build failed. The ordering gate should have caught it — check the build log. |
| Very slow, choppy updates | Expected on the free tier. Lower `QSAFE_LINK_RATE` to `1`. |
| Instance runs out of memory | Set `QSAFE_LINK_TYPE_TAGGER=0`. The core escalate/de-escalate path never uses it. |

**Do not present from the deployed instance.** It is for sharing a link. The
booth demo runs on your laptop — see `RUNBOOK.md` § 2.
