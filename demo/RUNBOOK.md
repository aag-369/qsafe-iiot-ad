# Q-Safe Field Link — demo runbook

Everything needed to run the live demonstration, in the order you will need
it. Read § 1–3 before the day; keep § 4–7 open during it.

---

## 1. Setup, once, before you leave

Run these from the repository root (`qsafe-iiot-ad/`).

### Option A — Docker (recommended: real liboqs, nothing to install)

```bash
docker build -t qsafe-field-link -f demo/Dockerfile .
docker run --rm -p 8000:8000 qsafe-field-link
```

`--network host` on Linux, or keep `-p 8000:8000` on Docker Desktop for
Windows/macOS. Phones reach it at your laptop's LAN address.

### Option B — local Python

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r demo/requirements.txt

# Real BIKE-L1 / HQC-128 (WSL or Git Bash on Windows; skip to use the
# labelled simulated fallback)
bash scripts/setup_liboqs.sh
```

### Then always run the preflight

```bash
python -m demo.preflight
```

It checks the interpreter, every package, every model artifact, that both
KEM profiles round-trip correctly **and that the baseline is genuinely
cheaper than the hardened profile**, that live detector decisions match the
batch pipeline, BB84 throughput on this machine, and the LAN address phones
will need. It prints one verdict. Do not skip it.

---

## 2. Start it

```bash
python -m qsafe_link.run --devices 3
```

It prints the URLs and binds `0.0.0.0`, so anything on the same Wi-Fi can
reach it:

```
Operator console : http://192.168.1.24:8000/console     ← the projector
Field sensor     : http://192.168.1.24:8000/node        ← phone 1
Control room     : http://192.168.1.24:8000/monitor     ← phone 2 / watch
Join / QR codes  : http://192.168.1.24:8000/            ← hand this around
```

**Everyone must be on one network.** If venue Wi-Fi is unreliable, turn on
your laptop's hotspot and join the phones to that. No internet is needed —
nothing loads from a CDN.

Useful flags:

| Flag | Why |
|---|---|
| `--devices 6` | Enough nodes for the fleet correlator to be interesting. |
| `--rate 3` | QKD rounds/second per device. Raise for a faster demo, lower if the laptop is slow. |
| `--qubits 96` | Near-zero spurious escalations (see § 6). Deviates from the training distribution — say so if asked. |
| `--tls` | Self-signed HTTPS, which unlocks the phone accelerometer. Phones warn once; tap through. |
| `--min-devices-alert 2` | Makes a fleet campaign alert easier to trigger with few phones. |

### Devices

Each phone that opens `/node` and picks a name **becomes a node** — no
setup. `--devices N` also pre-creates simulated nodes so the fleet view
works even with one phone. On a smartwatch, open `/monitor?compact=1`.

---

## 3. Two-minute room check

1. `python -m demo.preflight` → **READY**
2. Start the server, open `/console` on the projector
3. Phone 1 → `/node`, name it "Pump House A", drag the pad, confirm frames flow
4. Phone 2 (or watch) → `/monitor`, confirm the pressure matches phone 1
5. Tap **Active eavesdropper** on the console → both phones should flip to
   amber HQC-128 within a second
6. Tap **Normal operation** → back to green within ~2 seconds

If step 5 or 6 fails, go to § 7.

---

## 4. The five-minute demo

Times are a guide. The bracketed lines are what to actually say.

### 0:00 — The problem (30s, no clicking)

> "Everything you see on this screen is an industrial edge node — a pump
> station, a substation. It's protected by post-quantum cryptography,
> because RSA and elliptic curve are broken the day a quantum computer
> arrives, and the traffic being harvested *today* is decrypted then. That's
> Harvest Now, Decrypt Later, and it makes this a present-tense problem."
>
> "But the hardened post-quantum profile is expensive. On this machine one
> hardened handshake costs **9.2 milliseconds** and the lightweight one
> costs **0.78**." *(point at the header pills)* "Twelve times. On a
> battery-powered Cortex-M4 running a real-time control loop, you cannot
> afford to pay that all the time."

### 0:30 — What's actually running (45s)

> "So we don't. We watch the physics instead."
>
> *(point at the QBER chart)* "That's the quantum bit error rate of a BB84
> channel — real Qiskit circuits, one per round. Below it is a GRU with
> four thousand parameters, quantized to INT8, eighty kilobytes — small
> enough to flash onto the microcontroller. It's watching a twenty-round
> window of that error rate."
>
> *(point at the green strip)* "Green means the link is on the cheap
> profile. It's been green this whole time, because nobody's attacking."

### 1:15 — Show the honest thing first (45s)

Hand a phone to someone. Have them drag the setpoint pad.

> "This phone is the node. Drag that — you're changing a pump setpoint.
> Watch the other screen."

*(the monitor/watch tracks it)*

> "That telemetry is AES-256-GCM encrypted, under a key derived from a real
> BIKE-L1 handshake. Those matching sixteen hex digits" *(point at the key
> fingerprint on both phones)* "are the session key. Remember them."

### 2:00 — The stealthy attack (90s) — **the important beat**

Tap **HNDL reconnaissance**.

> "Now I'm turning on an eavesdropper. Not a loud one — this one intercepts
> about fifteen percent of qubits, because a real harvest-now attacker
> doesn't want to be caught. Watch the error rate."

*(QBER creeps from ~0.014 to ~0.05)*

> "It went from one and a half percent to five. If you'd set a static
> threshold — and everybody sets a static threshold — you'd have put it at
> eight or ten percent, and you would see **nothing**."

*(confidence climbs; the strip turns amber)*

> "But the model isn't looking at the number, it's looking at the *shape*
> over twenty rounds. There it goes."

Point at both phones.

> "And look — the session key changed. On both devices. That wasn't a
> colour change on a dashboard, that was a real HQC-128 handshake, and the
> rekey log says exactly what it cost." *(point at the log: ~9–13 ms)*

### 3:30 — Backing off (30s)

Tap **Normal operation**.

> "Attack's gone. It doesn't drop back instantly — there's a cooldown, so
> one noisy round can't make it flap. Four rounds of quiet, and…"

*(strip returns to green, key changes again)*

> "Back on the cheap profile. That's the whole idea: pay for the hardened
> crypto only while there's evidence you need it."

### 4:00 — The number (30s)

Point at the CPU-saved tile.

> "Seventy-something percent less KEM cost than running the hardened
> profile all the time — computed live, priced with handshake costs
> measured on this laptop. The paper reports 73.2% on a two-thousand-round
> benchmark. That's the same methodology, running in front of you."

### 4:30 — Fleet (30s, if you have 3+ nodes)

Tap **Coordinated fleet campaign**.

> "One device escalating is that device doing its job. Several devices
> escalating at the same time, *and agreeing on what kind of attack it is* —
> that's one adversary sweeping the fleet. Type agreement is what separates
> a campaign from a coincidence."

*(the red campaign banner appears)*

---

## 5. Letting judges drive it

Hand them the console and say: **"Pick any scenario. Watch the green bar."**

Each button carries its own subtitle and, once tapped, the panel underneath
tells them what to watch for. It is safe to mash them — the switch
controller re-evaluates every round independently and has no notion of
"already handling an attack", so nothing gets into a bad state.

Good things to invite:
- *"Try Photon-number splitting — that's the one we're worst at."* (Honesty
  reads well, and it's a real result: F1 0.40 on that class.)
- *"Drag the pad on the phone while an attack is running."* (Shows the link
  keeps carrying traffic through a rekey — no dropped frames.)
- *"Watch the key fingerprint, not the colour."* (The colour is UI; the
  fingerprint is cryptography.)

---

## 6. Hard questions, prepared answers

**"Is the QBER real, or are you just drawing a line?"**
Real. The scenario buttons set `eve_intercept_prob` and
`channel_error_prob` on an actual Qiskit BB84 circuit — state preparation,
intercept-resend Eve conditioned on a mid-circuit measurement, depolarizing
noise, sifting. QBER is computed from the sifted key. No QBER value in this
demo is synthesised directly. `qsafe_link/channel.py`.

**"Is that real post-quantum cryptography?"**
Yes, when the header pill is green: real `liboqs`, BIKE-L1 and HQC-1.
Keygen, encapsulate, decapsulate, with the shared secret checked for
agreement before any traffic key is installed. If the pill is amber, the
library isn't built and it's a timing-calibrated stand-in — labelled
everywhere, including in the exported report.

**"So the phone is doing post-quantum crypto?"**
No, and I won't claim it. There's no browser-side BIKE-L1 — only HQC-128
exists in WASM. The phone is the node's sensor and HMI; the PQC-protected
hop is node ↔ gateway, which is how an industrial gateway actually
terminates crypto for field devices. `ARCHITECTURE_MAPPING.md` § 3.

**"It just escalated and nothing was attacking."**
Correct, and expected. Our operational precision is 0.796 — about one in
five escalations is a false alarm, and we report that in the paper rather
than tuning it away. Measured here: 0.07% of benign rounds, roughly one per
eight minutes per device, 0.45% of time on the hardened profile. A false
escalation costs 9 milliseconds and fails toward *more* security. A false
*negative* would be the serious one, and recall is 0.951.

**"Why does the stealthy one take longer to catch?"**
Because it's genuinely harder. It lifts QBER by about three and a half
points above a noise floor that itself has a standard deviation of two.
There's no single round that's diagnostic — the model needs several rounds
of the window to fill with the new shape. That's the argument for a
temporal model over a threshold.

**"What if the attacker knows the threshold and stays under it?"**
Then they're intercepting so little that the harvest is small, and they
still have to stay under it *for the whole window*, not per round. It's a
real limitation and it's in the paper's Limitations section. The honest
answer is that this raises the cost of stealth, it doesn't eliminate it.

**"Does the attack-type classifier drive the crypto decision?"**
No — deliberately. It's additive. A misclassified attack type can never
weaken the active cryptographic posture. That's an architectural property,
not a configuration: `SwitchController` only ever sees the binary
detector's confidence.

**"Would this run on a real Cortex-M4?"**
The detector would — it's INT8 TFLite, 80.6 KB, and
`firmware_notes/CORTEX_M4_DEPLOYMENT.md` covers the port. The absolute
millisecond figures here are host-measured and would change. The portable
claim is the *ratio* between profiles, which is what the adaptive policy
exploits.

**"Why rekey on switch rather than per round?"**
Production links don't re-handshake every packet. Rekeying on profile
change is realistic, and it's what makes the switch visible — the session
key fingerprint changes because a real handshake just ran.

---

## 7. Troubleshooting

**Phones can't reach the laptop.**
Same network? Try the laptop's hotspot. Windows Firewall will prompt on
first run — allow **private networks**. Confirm the address with
`python -c "from qsafe_link.gateway import detect_lan_ip; print(detect_lan_ip())"`.
If it prints `127.0.0.1`, Wi-Fi is down.

**Header pill says "SIMULATED KEM".**
`liboqs` isn't built. Run `bash scripts/setup_liboqs.sh` (WSL or Git Bash on
Windows), or use the Docker image. The demo still works — it's labelled.

**Everything is slow / rounds lag.**
Lower `--rate` or `--devices`. Preflight prints this machine's BB84
throughput; total load is `devices × rate` rounds/second. Under attack a
64-qubit round costs ~20 ms, so ~12 devices-rounds/second is a practical
ceiling on a modest laptop.

**A device escalates and won't come back.**
Confirm the attack is actually cleared — a per-device scenario stays on that
device. Tap **Normal operation** (fleet-scoped, clears everything). De-escalation
needs 4 consecutive rounds below 0.50.

**Accelerometer button says unavailable.**
Expected on `http://`. Browsers gate motion sensors behind a secure context.
Restart with `--tls` and tap through the certificate warning. The touch pad
works either way — the accelerometer is a bonus, not a requirement.

**Console shows an empty rekey log.**
Only if no profile change has happened yet. The log is rebuilt from device
state on every snapshot, so opening the console mid-session shows full
history.

---

## 8. Deploying it (Render) — and why that is not the booth demo

The repository deploys as **one** service that serves both surfaces:

```
https://<your-app>.onrender.com/               research dashboard
https://<your-app>.onrender.com/link/console   Field Link demo
```

`server.py` mounts both apps in one process. `render.yaml` already points at
it, so after you push, Render redeploys and the demo appears — no dashboard
changes needed on your side.

**Do not present from the deployed instance.** A free instance gets a fraction
of a CPU, and every BB84 round is a real quantum-circuit simulation. The
control loop will run visibly slower than on your laptop, and the free tier
also sleeps after 15 minutes idle (30-60s to wake). Deploy it so you can *send
someone a link*; run the booth demo locally.

Three behaviours differ on the deployed instance, all deliberate:

| Behaviour | Why |
|---|---|
| Starts with **no devices** | An unattended instance should not execute quantum circuits for nobody. Press **Start a demonstration** on the console. |
| Control loop **pauses** after 120s with no browser connected | Same reason. It resumes the instant someone connects. |
| Models load on the **first** request to `/link` | So the platform's health check does not wake TensorFlow on an instance nobody is using. The first console load is therefore slower. |

Everything is an environment variable in `render.yaml` — `QSAFE_LINK_RATE`,
`QSAFE_LINK_QUBITS`, `QSAFE_LINK_DEVICES`, `QSAFE_LINK_IDLE_PAUSE_S`,
`QSAFE_LINK_TYPE_TAGGER`, and `QSAFE_LINK_ENABLED=0` to serve the dashboard
alone. If the instance runs out of memory, set `QSAFE_LINK_TYPE_TAGGER=0`
first: the attack-type classifier is the largest single item and the core
escalate/de-escalate path never consults it.

**Checking a deploy worked.** `/link/api/health` should return
`worker_alive: true` and `models_loaded: false` before anyone opens the demo.
If `kem_backend` says anything other than `LiboqsKEMBackend`, the liboqs build
failed during the image build and the UI will label every result as simulated.

## 9. If the Wi-Fi dies — the fallback

Recorded before you left:

```bash
python -m demo.record_demo --devices 4 --rate 6
```

This produces, in `demo_output/`:

| File | What it is |
|---|---|
| `demo_replay.html` | **Open this.** A standalone page — the whole session embedded, works from a USB stick with no server, no network, no Python. |
| `demo_report.json` | The evidence pack: detection latencies, rekey log with real handshake costs, per-profile round counts, provenance. |
| `charts/*.png` | Publication-quality figures — QBER, confidence, and profile strip aligned against ground-truth attack windows. |
| `demo_session.json` | Full event trace. |

Open `demo_replay.html` and run § 4 against it, past tense. Every number in
it was measured by the same code, so nothing you say has to change.

**Re-record it the morning of.** It takes about ninety seconds and the
timestamp in the report is visible on the page.
