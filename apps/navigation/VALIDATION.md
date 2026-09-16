# Navigation bridge validation

Validated in WSL Ubuntu with one headless Microduck in `apartment_flat`. This record covers
the observation/action foundation, not autonomous kitchen navigation or physical hardware.

## Automated checks

- `uv run pytest`: 42 passed, plus 9 subtests. Includes stale/replayed observations, malformed
  sensors, floor/obstacle classification, failed transport, cancellation, concurrent stop,
  frozen odometry, gaze HOME-offset conversion at the client boundary, and stdin EOF.
- `uv run ruff check duck_nav tests scripts`: passed.
- `cargo test -p robotd -p duck-ipc-proto`: 217 passed; one existing recurrent-memory test
  ignored because it requires ONNX Runtime >= 1.23. Includes a camera-direction regression
  for the `robot.look` HOME-offset bug and resending returned head commands.
- The flat-scene regression checks 18 points over the stairwell: the new scene has level
  floor there and the original scene still has descending steps.
- Existing `spaces/policy-shop/lan.py` selfcheck passed with real local WebRTC peers, RGB
  frames, signalling, DTLS and JSON-RPC, without the Reachy SDK installed.
- Shell syntax and both repositories' `git diff --check` passed.

## Live observations

The baseline simulator reports 50 Hz control, healthy bus/IMU, approximately 0.117 m trunk
height and 1.00x real time. The bridge receives camera video, all 64 ToF zones and timestamped
robot state over WebRTC. Camera recordings have the mounting rotation corrected.

A bounded 0.08 m/s, 0.5-second move completed. An explicit stop cancelled a subsequent move
after approximately 0.16 seconds. The checks require exactly zero requested velocity and
applied velocity below 0.001 within two seconds; daemon velocity smoothing makes exact
floating-point zero an inappropriate settling test. `runs/*/smoke.json` holds measured values.

Live validation exposed and fixed a pre-existing gaze bug: absolute IK angles were being
sent as HOME-relative policy commands. IPC API 29 separates the returned command offsets
from absolute joint targets. The bridge also preserves measured neck posture while aiming.
After rebuilding API 29, `look_at 1 0 0` returned `gaze_settled` in 0.53 seconds with a fresh
camera frame. The final forward/cancel smoke also passed on that build, with final health
reporting 50 Hz and 1.00x real time.

Local evidence from this validation run (recordings are intentionally ignored by git):

- Gaze: `runs/20260916T032640Z-4b4fe967/events.jsonl`
- Forward/cancel: `runs/20260916T032712Z-ecb5ed01/smoke.json`

## Remaining limits

- The current gait under-tracks conservative short turns. One 15-degree request produced
  approximately 1.4 degrees before the no-progress guard stopped it. The bridge reports an
  incomplete result and measured angle; it does not repeatedly increase command duration.
  Calibrate turn tracking before building a dependable room-navigation loop.
- A movement duration is a command budget, not a distance guarantee. The short forward test
  produced only millimetres of odometry displacement during its active command window.
- Stop acknowledgement confirms accepted zero intent. It does not prove physical settling.
- The forward depth sensor cannot certify clearance around the feet, behind the body, or
  across a drop-off. The test scene covers the stairwell specifically for this first milestone.
- There is no vision-model adapter, kitchen recognition, voice interface, persistent room map,
  or autonomous retry logic in this milestone.
