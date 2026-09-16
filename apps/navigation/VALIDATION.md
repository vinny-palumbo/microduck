# Navigation and visual-agent validation

Validated in WSL Ubuntu with one headless Microduck in `apartment_flat`. This record covers
the observation/action foundation, scripted agent loop and one live visual inspection.
It does not establish autonomous kitchen navigation or physical hardware readiness.

## Automated checks

- `uv run pytest`: 92 passed, plus 9 subtests. Includes stale/replayed observations, malformed
  sensors, floor/obstacle classification, failed transport, cancellation, concurrent stop,
  frozen odometry, gaze HOME-offset conversion at the client boundary, stdin EOF, and calibration
  scoring (yaw wrap, transient false positives, residual commands, drift, odometry disagreement,
  both-direction probe cutoff and cancellation). Agent regressions also cover serial JPEG/result
  context, no simulator-truth input, unverified completion, blocked/stuck detection across gaze
  actions, invalid/multiple/truncated tool calls, stale observations, timeouts and cancellation.
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

## Turn calibration — 2026-09-16

**Result: 0/12 passed.** The scored battery requests ±5°, ±15°, ±30° twice, sequentially,
with the ordinary 0.25 rad/s guard configuration. The full scoring rules and repeatable command
are in [the README](README.md#measure-turn-accuracy). Actual settled motion was:

| Request | True settled rotation, range over two trials | Result |
|---|---|---|
| +5° | −0.14° to +0.13° | no progress |
| −5° | −0.41° to −0.36° | no progress |
| +15° | +0.05° to +0.07° | no progress |
| −15° | −0.58° to −0.54° | no progress |
| +30° | +0.09° to +0.11° | no progress |
| −30° | −0.68° to −0.59° | no progress |

Odometry and true settled yaw agreed within 0.012° in this battery. Every action requested zero
on stop, the final applied command was below 0.001 rad/s, and the process exited 1 as expected
for failed accuracy. The simulator remained upright and healthy at 50 Hz, approximately 1.00x.
This rules out a heading-estimator discrepancy in these runs; it does not establish accurate
turning merely because the requested command was delivered.

An earlier session retained a nonzero head command and was close to the head-alignment boundary.
After restarting the flat simulator, a fresh-posture 12-trial battery reproduced the failure.
Subsequent two-second rate probes retained the sensor/depth/health guards and had a 25° measured
excursion cutoff. At +0.6, −0.6, +1.0, and −1.0 rad/s, true settled rotation was respectively
+0.28°, −0.96°, +0.44°, and −0.96°. The −1.0 probe briefly twisted 11.03° but sprang back after
stopping. Its full two-second duration completed; that is not a successful angle-controlled turn.
The scored battery above followed those probes, with the default head command retained.

Configured policy: `velstand.onnx`, SHA-256
`1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`.
Runtime revision: `3abb4dd`; simulator revision: `6bf2d69`. The benchmark itself was uncommitted
during measurement. Raw recordings remain local and ignored by git:

- Fresh-posture battery: `runs/20260916T163129Z-fc94df1d/turns.json`
- Higher-rate probes: `runs/20260916T163323Z-ecd4825a/turns.json`
- Scored battery: `runs/20260916T163725Z-a21c7e92/turns.json`

### Actuator comparison and next engineering step

Baseline inspection found an actuator-fidelity gap. The archived experiment below tested it;
the current runtime retains the original XML actuators.
An additional offline comparison used the existing `microduck_rl/scripts/infer_policy.py`
controller, the same policy and apartment scene, fresh HOME placement for each trial, zero
head/body commands, 50 Hz inference, 0.2 command smoothing, two seconds at rest, two seconds of
yaw command, then two seconds at zero. It compared XML position actuators with BAM M6 at
7.4 V, firmware gain 200, no current limit or voltage sag:

| Yaw command | XML settled rotation | BAM settled rotation |
|---|---|---|
| +0.25 rad/s | +0.60° | +1.37° |
| −0.25 rad/s | −0.76° | −0.58° |
| +1.00 rad/s | +0.95° | +2.39° |
| −1.00 rad/s | −0.88° | −3.56° |

All eight remained upright (final trunk height 0.116–0.117 m). The scratch experiment and its
results are `runs/actuator_probe.py` and `runs/actuator_probe.json`. This is a single trial per
condition, an offline policy rehearsal rather than a daemon/WebRTC test; it suggests actuator
fidelity alone will not recover short-turn tracking. It is not a calibrated controller gain.

No runtime guard limits or policy weights were changed in this initial calibration milestone.

## Archived BAM experiment — 2026-09-16

This experiment was reverted after it failed to fix locomotion. A restorable patch, including
the diagnostic script and tests, is saved locally at
`/home/vin_p/Pollen/backups/20260916-bam-experiment.patch`. It was checked with `git apply --check`
against the restored RL checkout. Measurements below are historical evidence, not the current
runtime. The simulator was restarted with its original XML actuators.

During the experiment, the simulator shared CPU BAM setup with the deployment rehearsal.
Regression tests compared joint trajectories and motor torques to 1e-10 for
identical target sequences, with and without battery sag. Additional tests cover torque-off,
stale-target clearing on enable, independent ducks, legacy diagnostics and training constants.
The experimental simulator/rehearsal tests passed (9 tests); navigation tests at that point
were 55 passed with 9 subtests. The BAM-specific tests are now in the archived patch.

After restarting with BAM, the daemon remained upright at 50 Hz and approximately 1.00x real
time. Forward/cancel smoke passed: bounded move 0.505 s, explicit cancellation 0.154 s, final
applied commands below 0.001. Gaze also returned `gaze_settled` in 0.437 s. The twelve ordinary turn requests still all returned
`turn_no_progress` and failed the settled-angle benchmark.

Live two-second rate probes, retaining the existing diagnostic bounds, produced:

| Command | True settled rotation |
|---|---|
| +0.6 rad/s | +1.52° |
| −0.6 rad/s | −2.67° |
| +1.0 rad/s | +3.30° |
| −1.0 rad/s | −4.90° |

To distinguish startup delay from persistent under-tracking, the now-archived offline script
`microduck_rl/scripts/measure_turn_response.py` resets the scene for each rate, uses the same
BAM setup and policy, runs a command for ten seconds, then observes four seconds at zero.
It writes the complete heading/tilt/command trace and stops the command early at 90° excursion
or 45° tilt. This isolated rehearsal never connects to the daemon or alters navigation limits.

The script is no longer present in the restored checkout; recover it from the patch only if
a future locomotion investigation needs this experiment.

The ten-second ±1.0 rad/s trials settled at +2.59°/−3.18°; their last-second angular rates
were below 0.001 rad/s in magnitude. All six rates (±0.25, ±0.6, ±1.0) remained upright, with
maximum tilt under 2.2°. Running the same test with the bundled `alpha_walking.onnx` also
under-tracked (+3.38°/−4.20° at ±1.0), so switching to that policy is not a demonstrated fix.
These are single deterministic offline trials per condition, not hardware validation or a
claim that every training-scene setting now matches.

Local evidence:

- Ordinary turn battery: `runs/20260916T171147Z-6d66b714/turns.json`
- Forward/cancel smoke: `runs/20260916T171515Z-b31941c9/smoke.json`
- Live response probes: `runs/20260916T171523Z-22fe27ea/turns.json`
- Gaze: `runs/20260916T171829Z-f0d5a3b8/events.jsonl`
- Offline traces: `runs/bam-turn-response.json`, `runs/bam-walk-turn-response.json`

**Decision:** pause actuator/contact/policy investigation. No policy weights were changed,
and these experiments do not establish that retraining is required. Preserve the calibration
and measurements; keep the current work focused on camera-driven decisions and truthful failure.

## Minimal agent loop — 2026-09-16

The implemented loop is typed goal -> fresh image -> one model decision -> guarded tool ->
result/new image -> reassessment. The adapter is implemented against the Gemini Robotics ER 2
`generateContent` interface. Initial validation used scripted decisions; the subsequent live
inspection below exercised provider access, images and model-selected gaze actions. Automated
adapter tests separately check request construction and response rejection.

Two scripted runs exercised the actual WebRTC simulator after the BAM rollback:

- `runs/20260916T174442Z-525e1e21/mission.json`: gaze settled in 0.43 seconds, a new observation
  was taken, and the fixture explicitly finished blocked after three decisions.
- `runs/20260916T174458Z-19adc3a5/mission.json`: two 0.05 m/s, 0.5-second moves returned
  `duration_elapsed`, but net displacement at the following observations was only 0.23 mm and
  0.04 mm. A look between them did not reset the failure count. The loop ended `blocked` after
  three decisions and 2.39 seconds, with stop acknowledged and `goal_verified: false`.

Both recordings identify `model: scripted-fixture`. They prove plumbing and failure handling,
not perception or kitchen navigation.

### First live Gemini inspection

`runs/20260916T182245Z-0da9db33/mission.json` records a live
`gemini-robotics-er-2-preview` run. The instruction was to look around, describe the scene and
identify a possible way forward, using gaze/observe/finish only. The model made two gaze calls
that returned `gaze_settled` (0.44 and 0.57 seconds), then finished after three decisions and
11.99 seconds. Final stop was acknowledged; no body movement tool was called.

The model described an opening between light and dark grey walls, an orange/pink checkered
floor and a blue-grey wall beyond. Review of `frame-0005.jpg` supports those visible features.
Its proposed path through the opening is a visual hypothesis: clearance, traversability and
room identity were not established. `goal_observed` reflects completion of this inspection
request, with `goal_verified: false`; it is not a navigation success.

The key was passed privately to this process at runtime and was not stored in project files.
This single run established an end-to-end live camera/model/gaze loop. The later movement
checks below extend that evidence; reliable recognition across scenes remains unvalidated.

### Windows credential storage

The key is now stored in the current user's Windows Credential Manager under
`Pollen/Microduck/Gemini`. Write/read-back verification passed. Computer Use was stopped by
the user during Notepad cleanup, so discarding the plaintext draft is not confirmed. The setup
and status commands never print the key.

`runs/20260916T183239Z-91f99970/mission.json` verifies automatic loading with both key environment
variables explicitly removed. The real model described the camera view and finished in one
decision (2.95 seconds), without a body or head movement command; final stop was acknowledged.
This validates the saved credential path, not additional navigation capability.

Credential tests cover environment precedence, captured pipes, missing credentials/interop,
read-back mismatch, generic errors without secret output, timeout redaction and hidden-input
setup. Unit tests use fake credentials and never access the real Windows vault.

### Live movement and failure handling

The first movement test exposed a model decision failure:
`runs/20260916T184118Z-391f1607/mission.json`. Gemini requested a turn despite an obstacle
refusal, then requested forward motion after the refused turn. Both actions were blocked by
the existing guards. The local two-failure cutoff ended the mission; the model itself did
not explain the failure. This run does not count as successful model reassessment.

The system instruction now explicitly requires recentering before the first body action,
respecting `ready: false`, and finishing blocked after either `completed: false` or
`progress.negligible: true`. These are model instructions, not a replacement for the runtime
guards or a guarantee of future compliance. The existing two-failure cutoff remains in place.
No motion bounds, gait parameters, policy weights or actuator code changed.

Three subsequent live checks passed their bounded decision criteria:

| Run | Model decisions | Measured result |
|---|---|---|
| `20260916T184205Z-ae3b8621` | recenter, finish blocked | Guard reported obstacle at about 0.18 m; no body action requested. |
| `20260916T184257Z-4ce3b505` | recenter, turn +25 degrees, finish blocked | `turn_no_progress`; net odometry rotation at the next observation was 0.137 degrees. |
| `20260916T184327Z-164cd509` | recenter, move 0.05 m/s for 0.5 s, finish blocked | Tool returned `completed: true` / `duration_elapsed`; net displacement was only 0.055 mm. Model cited that negligible progress and stopped. |

The simulator was restarted in the same flat scene before the turn test to remove accumulated
idle drift. The forward test followed it. Each run is recorded under `runs/<run>/` with camera
frames, decision reasons, tool outcomes and `mission.json`. Checks against the records confirmed
recenter-first behavior, model-selected `finish(blocked)`, zero final requested commands,
applied commands below 0.001 and acknowledged stop. Every result retained `goal_verified: false`.
The latter two missions took 8.51 and 8.06 seconds respectively.

This is evidence that the revised loop can report both guard refusals and ineffective movement,
including an accepted command that did not meaningfully move the robot. It is not a reliability
benchmark, proof of obstacle identity, physical stillness, or successful navigation.

## Locomotion diagnosis and gaze correction (API 30)

The failed low-speed commands do not establish that the policy cannot walk. CPU MuJoCo
probes of the same deployed `velstand.onnx` (SHA-256
`1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`) found a sharp response
change: two-second commands from 0.10 through 0.25 m/s moved less than 1 mm after settling;
0.30 m/s moved about 104 mm with nearly 8 degrees of heading drift. Shorter pulses also
showed a startup delay. These are individual trials, not calibrated or repeatability results.

Changing collision settings did not remove the low-command failure. Restoring BAM was not
necessary to demonstrate motion. Runtime action scaling/filtering differs from training,
but isolated tests with and without it also moved at 0.30 m/s, so this investigation does not
justify changing those settings.

A separate causal fault was found in gaze: each look used measured neck pitch as the next
absolute neck target. The policy has a steady neck tracking bias, so repeated looks could
progressively lower the commanded posture. In an isolated 1.5-second 0.30 m/s comparison
using runtime scaling/filtering, neutral head commands moved about 105 mm; the lowered
neck command moved about 1.14 mm. Scratch experiment summaries are in ignored
`runs/contact_audit.json`, `speed_audit.json`, `short_pulse_audit.json`,
`duration_audit.json`, `filter_audit.json` and `head_audit.json`.

Two live WebRTC probes support this finding:

- `20260916T190418Z-9d10045b/forward.json`: after the old recenter call, a 0.30 m/s,
  1.5-second command produced about 1.18 mm settled simulator-truth displacement.
- `20260916T190847Z-7ca0bcc6/forward.json`: neutral head commands produced 65.42 mm
  displacement and -2.47 degrees of heading change. An odometry monitor requested stop at
  50 mm, about 1.19 seconds into the command. The settled displacement overshot that cutoff
  by about 15 mm. Final requested commands were zero.

These diagnostic probes retained sensor/health/depth guards and imposed a 1.5-second cap,
50 mm odometry cutoff and 10-degree heading cutoff. They used a diagnostic 0.30 m/s command
outside the model tool's limit. Simulator truth was used for evaluation only. The production
forward limit remains 0.10 m/s; no gait, actuator, policy weights or training changes were made.

API 30 publishes HOME joint angles. The bridge now preserves the applied neck command
instead of measured tracking error. It also requires the measured camera optical ray to be
within 0.10 radians of the requested point, alongside head-joint alignment and a subsequent
fresh camera frame. Exact neck tracking is not required when the camera is correctly aimed.

`20260916T191525Z-957ed74d/events.jsonl` records three successive live recenter calls after
rebuilding the daemons. All preserved a zero neck offset and acknowledged stop, but all
returned `look_timeout`: the measured camera did not meet the new alignment criterion.
This exposes remaining gaze tracking error; it is not a successful gaze or navigation test.
Earlier `gaze_settled` results used only the older joint checks and must not be interpreted as
passing this stronger optical alignment check. The inspection fixture may now end blocked
on gaze timeout. Fixing this requires measured gaze feedback or a demonstrated tracking fix,
not relaxing the criterion or retrying body movement.

Regression validation: 95 Python tests plus 9 subtests passed; Ruff passed. Rust tests for
`robotd` and `duck-ipc-proto`: 217 passed, 1 ignored. Tests cover repeated looks despite neck
tracking bias, preserving nonzero posture, missing contract fields and rejecting a misaligned
camera even when joint checks pass.

The next experiment should close and validate the gaze feedback loop while preserving neck
posture. Only then calibrate a short forward primitive across repeated trials, including
heading drift and stopping overshoot, before exposing an effective speed to the visual agent.

## Remaining limits

- The turn calibration above fails. The bridge reports incomplete turns and does not
  automatically retry them. Dependable room navigation still requires effective locomotion;
  precision turning is not a prerequisite for testing perception and guarded decisions.
- A movement duration is a command budget, not a distance guarantee. The short forward test
  produced only millimetres of odometry displacement during its active command window.
- Stop acknowledgement confirms accepted zero intent. It does not prove physical settling.
- The forward depth sensor cannot certify clearance around the feet, behind the body, or
  across a drop-off. The test scene covers the stairwell specifically for this first milestone.
- Live inspection and three bounded failure-handling checks passed after the instruction fix.
  Reliability across repeated trials/scenes, kitchen recognition and arrival remain unvalidated.
  Voice, persistent maps and locomotion retraining remain deferred.
