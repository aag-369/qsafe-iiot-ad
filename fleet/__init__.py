"""Fleet-scale simulation: runs several independent IIoT devices at once and
correlates their detections to distinguish a coordinated, multi-device
attack campaign from ordinary, unrelated per-device noise.

Each device still runs the exact same, unmodified core pipeline used
everywhere else in this project (qkd_sim -> ai_detector's binary GRU ->
crypto_agility's switch controller + real liboqs KEM ops) — nothing about
the tested single-device security decision changes. What's new here is
purely additive: the attack-type classifier tags each device's rounds, and
FleetCorrelator looks *across* devices for coordinated timing + matching
attack types.
"""
