"""
Scripted, headless run of the full demonstration — the fallback for the day
the venue Wi-Fi does not work.

Runs the exact same `LinkRuntime` the live gateway runs, drives it through a
fixed sequence of scenarios, and writes:

    demo_output/demo_report.json    measured results (the evidence pack)
    demo_output/demo_session.json   full event trace
    demo_output/demo_replay.html    standalone replay -- opens with no server
    demo_output/charts/*.png        publication-quality figures (if matplotlib)

Nothing about the pipeline is stubbed: real Qiskit BB84 rounds, the committed
INT8 detector, the published thresholds, and real liboqs KEM operations where
the library is available.

    python -m demo.record_demo --devices 4 --rate 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# (scenario key, seconds to hold, target: "focus" one device or None for fleet)
DEFAULT_SCRIPT = [
    ("calm", 12, None),
    ("hndl", 18, "focus"),
    ("calm", 10, None),
    ("eavesdrop", 14, "focus"),
    ("calm", 10, None),
    ("jamming", 12, "focus"),
    ("calm", 8, None),
    ("pns", 16, "focus"),
    ("calm", 8, None),
    ("campaign", 18, None),
    ("calm", 12, None),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record a scripted Q-Safe Field Link session")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--rate", type=float, default=6.0, help="rounds/s (higher = faster recording)")
    p.add_argument("--qubits", type=int, default=64)
    p.add_argument("--out", default=str(REPO_ROOT / "demo_output"))
    p.add_argument("--models", default=str(REPO_ROOT / "models"))
    p.add_argument("--speed", type=float, default=1.0, help="scale every hold duration")
    p.add_argument("--no-charts", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from qsafe_link.recorder import SessionRecorder
    from qsafe_link.runtime import LinkRuntime
    from qsafe_link.scenarios import apply_scenario

    print("[record] loading models and resolving KEM backend...")
    runtime = LinkRuntime(
        models_dir=args.models,
        n_qubits_per_round=args.qubits,
        rounds_per_second=args.rate,
        min_devices_for_alert=3,
    )
    recorder = SessionRecorder().attach(runtime)

    names = [
        ("plant-01", "Pump House A"),
        ("plant-02", "Valve Station B"),
        ("plant-03", "Substation C"),
        ("plant-04", "Water Intake D"),
        ("plant-05", "Compressor E"),
        ("plant-06", "Feeder Line F"),
    ]
    n_devices = max(1, min(args.devices, len(names)))
    if n_devices != args.devices:
        print(f"[record] --devices {args.devices} clamped to {n_devices}")
    for i in range(n_devices):
        runtime.add_node(names[i][0], names[i][1], seed=2000 + i)
    focus = names[0][0]

    bs = runtime.backend_state()
    print(f"[record] KEM: {bs['kem_backend']} (real liboqs: {bs['using_real_liboqs']})")
    print(f"[record] detector: {bs['detector_backend']}, threshold {bs['detector_threshold']:.2f}")
    print(f"[record] handshake cost — BIKE-L1 {bs['bike_reference_ms']:.2f} ms, "
          f"HQC-128 {bs['hqc_reference_ms']:.2f} ms\n")

    runtime.start()
    # Let the 20-round detector window fill before the first scenario, so the
    # opening state is a warm link rather than a cold-start artifact.
    warmup = max(4.0, 24 / args.rate)
    print(f"[record] warming up for {warmup:.0f}s...")
    time.sleep(warmup)

    total = sum(d for _, d, _ in DEFAULT_SCRIPT) * args.speed
    print(f"[record] running {len(DEFAULT_SCRIPT)} steps, ~{total:.0f}s\n")
    t_start = time.time()

    for key, hold, target in DEFAULT_SCRIPT:
        dev = focus if target == "focus" else None
        apply_scenario(runtime, key, dev)
        node = runtime.get(focus)
        label = f"{key} -> {dev or 'fleet'}"
        profile = node.session.profile.value if node and node.session.profile else "?"
        print(f"[record] +{time.time()-t_start:6.1f}s  {label:<28} (profile now {profile})")
        # Send application traffic throughout so the AEAD counters and the
        # ciphertext previews in the report are real, not zero.
        deadline = time.time() + hold * args.speed
        i = 0
        while time.time() < deadline:
            for n in runtime.node_list():
                try:
                    n.send_uplink(
                        {
                            "setpoint_pct": 40 + (i % 20),
                            "pressure_bar": round(1.2 + (40 + i % 20) * 0.062, 3),
                            "temp_c": round(42 + (i % 7) * 0.3, 2),
                            "src": "scripted",
                        }
                    )
                except Exception as exc:
                    # This script is the fallback for when the venue Wi-Fi
                    # fails. Losing the whole evidence pack to one dropped
                    # frame would defeat its purpose.
                    print(f"[record] frame dropped on {n.device_id}: {exc}")
            i += 1
            time.sleep(0.5)

    runtime.stop()
    print("\n[record] writing evidence pack...")

    out = Path(args.out)
    try:
        paths = recorder.save(runtime, out)
    except Exception as exc:
        print(f"[record] FAILED to write the evidence pack: {exc}")
        return 1
    report = json.loads(paths["report"].read_text())

    write_replay(out / "demo_replay.html", recorder, runtime, report)
    if not args.no_charts:
        try:
            write_charts(out / "charts", recorder, runtime)
        except Exception as exc:
            print(f"[record] charts skipped ({exc})")

    print_summary(report)
    print(f"\n[record] wrote:\n  {paths['report']}\n  {paths['trace']}\n  {out/'demo_replay.html'}")
    return 0


def print_summary(report: dict) -> None:
    d = report["detection"]
    fleet = report["fleet"]
    cmp_ = report["comparison_to_paper"]
    print("\n" + "=" * 66)
    print("  RECORDED SESSION SUMMARY")
    print("=" * 66)
    print(f"  KEM provenance          {report['provenance']['kem']}")
    print(f"  Attack episodes         {d['n_episodes']}")
    if d["median_latency_s"] is not None:
        print(f"  Detection latency       median {d['median_latency_s']}s  "
              f"(min {d['min_latency_s']}s, max {d['max_latency_s']}s)")
    if d.get("n_spurious_escalations"):
        print(f"  False positives         {d['n_spurious_escalations']} "
              f"(escalations with no adversary active)")
    if d.get("n_unmeasurable_episodes"):
        print(f"  Unmeasurable episodes   {d['n_unmeasurable_episodes']} "
              f"(device was already hardened — no escalation to time)")
    print(f"  Escalations             {report['crypto_agility']['n_escalations']}")
    print(f"  De-escalations          {report['crypto_agility']['n_de_escalations']}")
    for profile, v in report["crypto_agility"]["handshake_ms_by_profile"].items():
        print(f"    {profile:<10} {v['n']:3d} handshakes, median {v['median_ms']:.2f} ms")
    print(f"  Rounds on baseline      {cmp_['live_baseline_round_fraction']*100:.1f}%")
    print(f"  CPU saved (paper method){cmp_['live_paper_equivalent_saved_pct']:.1f}%  "
          f"(paper reports {cmp_['paper_cpu_reduction_pct']}%)")
    print(f"  Fleet campaign alerts   {len(fleet['alerts'])}")
    print("=" * 66)


def write_replay(path: Path, recorder, runtime, report: dict) -> None:
    """Standalone replay page: the whole session embedded, no server needed."""
    rounds_by_device: dict[str, list] = {}
    for ev in recorder.events:
        if ev["kind"] != "round":
            continue
        d = ev["data"]
        rounds_by_device.setdefault(d["device_id"], []).append(
            [round(d["qber"], 5), round(d["confidence"], 5), d["profile"],
             1 if d["ground_truth_attack"] else 0]
        )
    scenarios = [
        {"ts": ev["ts"], "key": ev["data"]["scenario"]["key"],
         "title": ev["data"]["scenario"]["title"]}
        for ev in recorder.events if ev["kind"] == "scenario"
    ]
    payload = {
        "report": report,
        "rounds": rounds_by_device,
        "scenarios": scenarios,
        "threshold": runtime.detector.threshold,
        "names": {n.device_id: n.display_name for n in runtime.node_list()},
    }
    html = _REPLAY_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_charts(out_dir: Path, recorder, runtime) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#0d0d0d", "axes.facecolor": "#1a1a19",
        "savefig.facecolor": "#0d0d0d", "text.color": "#ffffff",
        "axes.labelcolor": "#c3c2b7", "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.edgecolor": "#383835",
        "grid.color": "#2c2c2a", "font.size": 9,
    })

    for node in runtime.node_list():
        hist = node.recent(100000)
        if len(hist) < 20:
            continue
        t = [h["t"] for h in hist]
        qber = [h["qber"] for h in hist]
        conf = [h["confidence"] for h in hist]
        truth = [h["ground_truth_attack"] for h in hist]

        fig, axes = plt.subplots(
            3, 1, figsize=(11, 5.4), sharex=True,
            gridspec_kw={"height_ratios": [3, 3, 0.55], "hspace": 0.16},
        )

        # Ground-truth attack windows as shading on both value panels.
        for ax in axes[:2]:
            start = None
            for i, a in enumerate(truth):
                if a and start is None:
                    start = i
                elif not a and start is not None:
                    ax.axvspan(t[start], t[i], color="#d03b3b", alpha=0.13, lw=0)
                    start = None
            if start is not None:
                ax.axvspan(t[start], t[-1], color="#d03b3b", alpha=0.13, lw=0)
            ax.grid(True, alpha=0.35, lw=0.6)

        axes[0].plot(t, qber, color="#3987e5", lw=1.4)
        axes[0].set_ylabel("QBER")
        axes[0].set_title(f"{node.display_name} — live BB84 channel, detector, and KEM profile",
                          color="#ffffff", loc="left", fontsize=11)

        axes[1].plot(t, conf, color="#d95926", lw=1.4)
        axes[1].axhline(runtime.detector.threshold, color="#898781", ls="--", lw=1)
        axes[1].set_ylabel("detector confidence")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].text(t[0], runtime.detector.threshold + 0.03,
                     f"τ_up = {runtime.detector.threshold:.2f}", color="#898781", fontsize=8)

        for i, h in enumerate(hist):
            axes[2].axvspan(t[i], t[i] + 1,
                            color="#0ca30c" if h["profile"] == "BIKE-L1" else "#fab219",
                            alpha=0.9, lw=0)
        axes[2].set_yticks([])
        axes[2].set_ylabel("KEM\nprofile", rotation=0, ha="right", va="center", fontsize=8.5)
        axes[2].set_xlabel("QKD round")
        axes[2].set_xlim(t[0], t[-1])

        from matplotlib.patches import Patch
        # Legend on the figure, under the axes: keeping it inside axes[2]
        # forced a large empty band between the confidence plot and the
        # profile strip, which read as a rendering fault rather than spacing.
        fig.legend(
            handles=[
                Patch(color="#0ca30c", label="BIKE-L1 baseline"),
                Patch(color="#fab219", label="HQC-128 hardened"),
                Patch(color="#d03b3b", alpha=0.35, label="adversary active (ground truth)"),
            ],
            loc="lower center", ncol=3, frameon=False, fontsize=8.5,
            bbox_to_anchor=(0.5, -0.005),
        )
        # subplots_adjust rather than tight_layout: the figure-level legend
        # is not an Axes, so tight_layout warns and can mis-measure.
        fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.13, hspace=0.16)
        fig.savefig(out_dir / f"{node.device_id}_timeline.png", dpi=170)
        plt.close(fig)

    # Cost comparison across the fleet.
    nodes = runtime.node_list()
    if nodes:
        adaptive = sum(n.metrics.round_adaptive_ms for n in nodes)
        static = sum(n.metrics.round_static_ms for n in nodes)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        bars = ax.barh(["AI-gated adaptive", "Always-on HQC-128"], [adaptive, static],
                       color=["#0ca30c", "#fab219"], height=0.55)
        for b, v in zip(bars, [adaptive, static]):
            ax.text(v * 1.01, b.get_y() + b.get_height() / 2, f"{v:,.0f} ms",
                    va="center", color="#c3c2b7", fontsize=9)
        saved = (1 - adaptive / static) * 100 if static else 0
        ax.set_title(f"Cumulative KEM cost — {saved:.1f}% saved by AI gating",
                     color="#ffffff", loc="left", fontsize=11)
        ax.set_xlabel("milliseconds")
        ax.set_xlim(0, static * 1.18)
        ax.grid(True, axis="x", alpha=0.3, lw=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / "cost_comparison.png", dpi=170)
        plt.close(fig)
    print(f"[record] charts -> {out_dir}")


_REPLAY_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>Q-Safe Field Link — Recorded Session</title>
<style>
:root{color-scheme:dark;--s0:#0d0d0d;--s1:#1a1a19;--s2:#232322;--ink1:#fff;--ink2:#c3c2b7;--ink3:#898781;
--line:rgba(255,255,255,.1);--grid:#2c2c2a;--qber:#3987e5;--conf:#d95926;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--s0);color:var(--ink1);font-family:var(--sans);padding:22px}
.wrap{max-width:1180px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--ink3);font-size:12.5px;margin-bottom:18px}
.card{background:var(--s1);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.ct{font-size:11px;font-weight:650;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);margin-bottom:10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.tile{background:var(--s1);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .l{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3)}
.tile .v{font-size:27px;font-weight:680;margin-top:4px}
.tile .v.g{color:var(--good)}.tile .v.w{color:var(--warn)}
canvas{display:block;width:100%}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:inherit;font-size:12.5px;font-weight:600;color:var(--ink1);background:var(--s2);
border:1px solid var(--line);border-radius:8px;padding:7px 12px;cursor:pointer}
button[aria-pressed=true]{background:color-mix(in srgb,var(--qber) 26%,var(--s2));border-color:var(--qber)}
.legend{display:flex;gap:14px;font-size:11px;color:var(--ink2);margin-top:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th.n{text-align:right}
th{color:var(--ink3);font-weight:600;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.note{font-size:12px;color:var(--ink2);line-height:1.6}
.badge{display:inline-block;padding:3px 9px;border-radius:6px;font-size:11px;font-weight:650}
</style></head><body><div class="wrap">
<h1>Q-Safe Field Link — recorded session</h1>
<div class="sub" id="sub"></div>
<div class="tiles" id="tiles"></div>
<div class="card">
  <div class="ct">Device</div>
  <div class="row" id="devBtns"></div>
</div>
<div class="card">
  <div class="ct">Quantum bit error rate</div>
  <canvas id="c1" height="150"></canvas>
  <div class="legend"><span><i style="background:var(--qber)"></i>QBER</span>
  <span><i style="background:var(--crit);opacity:.35"></i>adversary active (ground truth)</span></div>
</div>
<div class="card">
  <div class="ct">GRU detector confidence</div>
  <canvas id="c2" height="140"></canvas>
  <div class="legend"><span><i style="background:var(--conf)"></i>confidence</span>
  <span style="color:var(--ink3)">— — escalate threshold</span></div>
</div>
<div class="card">
  <div class="ct">Active KEM profile</div>
  <canvas id="c3" height="26"></canvas>
  <div class="legend"><span><i style="background:var(--good)"></i>BIKE-L1 baseline</span>
  <span><i style="background:var(--warn)"></i>HQC-128 hardened</span></div>
</div>
<div class="card"><div class="ct">Detection episodes — measured attack onset to hardened profile live</div>
  <table id="epTable"></table></div>
<div class="card"><div class="ct">Provenance</div><div class="note" id="prov"></div></div>
</div>
<script>
const D = __PAYLOAD__;
const $ = (i)=>document.getElementById(i);
let dev = Object.keys(D.rounds)[0];

$('sub').textContent = `Recorded ${D.report.generated_at_iso} · ${D.report.session_duration_s}s · `
  + `${D.report.backend.kem_backend} · detector ${D.report.backend.detector_backend}`;

const c = D.report.comparison_to_paper, det = D.report.detection;
$('tiles').innerHTML = `
 <div class="tile"><div class="l">CPU saved vs always-on HQC-128</div><div class="v g">${c.live_paper_equivalent_saved_pct.toFixed(1)}%</div></div>
 <div class="tile"><div class="l">Rounds on baseline</div><div class="v">${(c.live_baseline_round_fraction*100).toFixed(0)}%</div></div>
 <div class="tile"><div class="l">Attack episodes</div><div class="v">${det.n_episodes}</div></div>
 <div class="tile"><div class="l">Median detection</div><div class="v w">${det.median_latency_s ?? '—'}s</div></div>
 <div class="tile"><div class="l">Escalations</div><div class="v w">${D.report.crypto_agility.n_escalations}</div></div>
 <div class="tile"><div class="l">Fleet alerts</div><div class="v">${D.report.fleet.alerts.length}</div></div>`;

$('devBtns').innerHTML = Object.keys(D.rounds).map((k)=>
  `<button data-k="${k}" aria-pressed="${k===dev}">${D.names[k]||k}</button>`).join('');
$('devBtns').querySelectorAll('button').forEach(b=>b.onclick=()=>{
  dev=b.dataset.k;
  $('devBtns').querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.k===dev)));
  draw();
});

function setup(cv){const d=Math.min(devicePixelRatio||1,2);const w=cv.clientWidth;const h=+cv.getAttribute('height');
 cv.width=w*d;cv.height=h*d;const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);x.clearRect(0,0,w,h);return{x,w,h};}

function line(cv,vals,truth,color,yMax,thr){
 const{x,w,h}=setup(cv);const pL=44,pR=8,pT=8,pB=14;const pw=w-pL-pR,ph=h-pT-pB;
 const Y=v=>pT+ph*(1-v/yMax),X=i=>pL+(vals.length<2?pw:pw*i/(vals.length-1));
 let s=null;for(let i=0;i<truth.length;i++){if(truth[i]&&s===null)s=i;
  else if(!truth[i]&&s!==null){x.fillStyle='#d03b3b';x.globalAlpha=.14;x.fillRect(X(s),pT,Math.max(1,X(i)-X(s)),ph);x.globalAlpha=1;s=null;}}
 if(s!==null){x.fillStyle='#d03b3b';x.globalAlpha=.14;x.fillRect(X(s),pT,Math.max(1,X(truth.length-1)-X(s)),ph);x.globalAlpha=1;}
 x.strokeStyle='#2c2c2a';x.fillStyle='#898781';x.font='10px system-ui';x.textAlign='right';x.textBaseline='middle';
 for(let i=0;i<=3;i++){const v=yMax*i/3,y=Math.round(Y(v))+.5;x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.fillText(v.toFixed(2),pL-7,y);}
 if(thr!=null){const y=Math.round(Y(thr))+.5;x.save();x.strokeStyle='#898781';x.setLineDash([4,4]);
  x.beginPath();x.moveTo(pL,y);x.lineTo(w-pR,y);x.stroke();x.restore();}
 x.strokeStyle=color;x.lineWidth=1.8;x.lineJoin='round';x.beginPath();
 vals.forEach((v,i)=>{const px=X(i),py=Y(v);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();}

function strip(cv,profiles){const{x,w,h}=setup(cv);const pL=44,pR=8;const st=(w-pL-pR)/profiles.length;
 // Draw contiguous runs as single rects: per-round rects land on sub-pixel
 // boundaries and anti-alias into stripes, which reads as profile flapping
 // that did not happen.
 let i=0;while(i<profiles.length){let j=i;while(j<profiles.length&&profiles[j]===profiles[i])j++;
  const x0=Math.round(pL+i*st),x1=Math.round(pL+j*st);
  x.fillStyle=profiles[i]==='BIKE-L1'?'#0ca30c':'#fab219';
  x.fillRect(x0,4,Math.max(1,x1-x0),h-8);i=j;}}

function draw(){const r=D.rounds[dev]||[];
 const q=r.map(a=>a[0]),cf=r.map(a=>a[1]),pr=r.map(a=>a[2]),tr=r.map(a=>a[3]);
 line($('c1'),q,tr,'#3987e5',Math.max(.25,Math.ceil((Math.max(...q,0)+.05)*20)/20),null);
 line($('c2'),cf,tr,'#d95926',1,D.threshold);
 strip($('c3'),pr);}

$('epTable').innerHTML = '<tr><th>Device</th><th>Scenario</th><th class="n">Detection latency</th><th class="n">Handshake</th><th>Profile</th></tr>'
 + det.episodes.map(e=>`<tr><td>${D.names[e.device_id]||e.device_id}</td><td>${e.scenario}</td>
   <td class="n">${e.latency_s.toFixed(2)} s</td><td class="n">${e.handshake_ms.toFixed(2)} ms</td>
   <td><span class="badge" style="background:#fab21922;color:#fab219">${e.profile}</span></td></tr>`).join('');

const p = D.report.provenance;
$('prov').innerHTML = `<b>KEM:</b> ${p.kem}<br><b>Detector:</b> ${p.detector}<br><b>QKD:</b> ${p.qkd}<br><br>${p.note}`;

draw(); addEventListener('resize', draw);
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
