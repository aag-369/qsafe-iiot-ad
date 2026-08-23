"""
One-tap adversary scenarios for the demo.

Each scenario is a named configuration of the *real* BB84 parameters from
`qkd_sim.qber_stream_multiclass.ATTACK_PROFILES` -- nothing here fabricates
telemetry. `intensity` interpolates within that attack type's published
parameter range, so "stealthy" and "aggressive" are different points on the
same physical model rather than different code paths.

The scenarios exist because a person who has never seen this system has
about fifteen seconds of patience. "Tap HNDL Reconnaissance, watch the amber
light come on" is a demo; "adjust the intercept probability slider to 0.22"
is a lab session.
"""

from __future__ import annotations

from dataclasses import dataclass

from qkd_sim.qber_stream_multiclass import AttackType


@dataclass
class Scenario:
    key: str
    title: str
    subtitle: str
    attack_type: AttackType | None
    intensity: float
    scope: str  # "single" | "fleet"
    expect: str
    detail: str

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "subtitle": self.subtitle,
            "attack_type": self.attack_type.name.lower() if self.attack_type else "benign",
            "intensity": self.intensity,
            "scope": self.scope,
            "expect": self.expect,
            "detail": self.detail,
        }


SCENARIOS: dict[str, Scenario] = {
    "calm": Scenario(
        key="calm",
        title="Normal operation",
        subtitle="No adversary present",
        attack_type=None,
        intensity=0.0,
        scope="fleet",
        expect="Link settles on BIKE-L1. This is the profile the device runs 82% of the time.",
        detail=(
            "Benign depolarizing channel noise only. QBER sits near the noise "
            "floor, the detector stays well under threshold, and the link runs "
            "the low-overhead baseline profile."
        ),
    ),
    "hndl": Scenario(
        key="hndl",
        title="HNDL reconnaissance",
        subtitle="Stealthy intercept-resend — the hard case",
        attack_type=AttackType.EAVESDROP,
        # Calibrated: ~14.8% interception lifts mean QBER from ~1.4% to ~5.2%.
        # Low enough that a static threshold set anywhere sensible misses it,
        # high enough that the detector converges in ~4 rounds and then holds
        # (95% of attack rounds on HQC-128, no flapping) over a 4-seed sweep.
        intensity=0.12,
        scope="single",
        expect="QBER rises only slightly. A static threshold would miss this; the GRU should not.",
        detail=(
            "A low-rate intercept-resend attack, the 'harvest now, decrypt "
            "later' reconnaissance pattern: the adversary samples a small "
            "fraction of qubits to avoid tripping a naive QBER threshold, "
            "banking ciphertext against a future quantum computer. This is the "
            "scenario the temporal model exists for."
        ),
    ),
    "eavesdrop": Scenario(
        key="eavesdrop",
        title="Active eavesdropper",
        subtitle="Aggressive intercept-resend",
        attack_type=AttackType.EAVESDROP,
        intensity=0.85,
        scope="single",
        expect="Fast escalation to HQC-128, usually within a round or two of onset.",
        detail=(
            "Textbook BB84 interception at high rate. Intercept-resend forces "
            "a ~25% error rate on the intercepted fraction, so QBER climbs "
            "sharply and unmistakably."
        ),
    ),
    "jamming": Scenario(
        key="jamming",
        title="Channel jamming",
        subtitle="Denial-of-service, no interception",
        attack_type=AttackType.JAMMING,
        intensity=0.6,
        scope="single",
        expect="Loud QBER spike; escalation, and the type classifier should tag it 'jamming'.",
        detail=(
            "Direct elevation of channel noise with no interception at all. "
            "A loud, easy signature -- but a different one, which is why the "
            "additive attack-type classifier is worth having."
        ),
    ),
    "pns": Scenario(
        key="pns",
        title="Photon-number splitting",
        subtitle="Sustained low-level bias — the stealthiest",
        attack_type=AttackType.PNS,
        # The stealthiest class by construction. 0.85 keeps it genuinely
        # subtle (QBER ~5.3%, no interception signature) while holding the
        # hardened profile ~89% of attack rounds rather than flapping.
        intensity=0.85,
        scope="single",
        expect="A small, sustained bias above the noise floor. The hardest class to attribute.",
        detail=(
            "A PNS-style attack exploits multi-photon pulses without the full "
            "disturbance of intercept-resend, so it sits just above the benign "
            "noise floor. Reported honestly: this is the class the attack-type "
            "classifier does worst on (F1 0.40)."
        ),
    ),
    "campaign": Scenario(
        key="campaign",
        title="Coordinated fleet campaign",
        subtitle="One adversary sweeping every device at once",
        attack_type=AttackType.EAVESDROP,
        intensity=0.7,
        scope="fleet",
        expect="Every device escalates AND agrees on attack type — the fleet correlator fires.",
        detail=(
            "The same attack type hits the whole fleet in the same window. "
            "Simultaneous escalation alone is not enough to alert -- unrelated "
            "devices escalating at the same moment for different reasons is a "
            "coincidence. Requiring type agreement is what separates a campaign "
            "from noise."
        ),
    ),
}

_ATTACK_BY_NAME = {
    "benign": None,
    "eavesdrop": AttackType.EAVESDROP,
    "jamming": AttackType.JAMMING,
    "pns": AttackType.PNS,
}


def resolve_attack_type(name: str) -> AttackType | None:
    key = (name or "benign").strip().lower()
    if key not in _ATTACK_BY_NAME:
        raise ValueError(f"unknown attack type {name!r}; expected one of {list(_ATTACK_BY_NAME)}")
    return _ATTACK_BY_NAME[key]


def apply_scenario(runtime, key: str, device_id: str | None = None) -> dict:
    """Apply a named scenario to one device or to the whole fleet."""
    if key not in SCENARIOS:
        raise ValueError(f"unknown scenario {key!r}; expected one of {list(SCENARIOS)}")
    scenario = SCENARIOS[key]

    if scenario.scope == "fleet" or device_id is None:
        targets = runtime.node_list()
    else:
        node = runtime.get(device_id)
        targets = [node] if node else []

    for node in targets:
        node.set_attack(scenario.attack_type, scenario.intensity, label=scenario.title)

    runtime.active_scenario = key if scenario.attack_type else None
    runtime.emit(
        "scenario",
        {
            "scenario": scenario.as_dict(),
            "devices": [n.device_id for n in targets],
        },
    )
    return {
        "scenario": scenario.as_dict(),
        "applied_to": [n.device_id for n in targets],
    }
