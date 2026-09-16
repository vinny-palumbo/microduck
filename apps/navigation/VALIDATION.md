# Navigation and visual-agent validation

## Voice branch integration — 2026-09-16

The `voice-guided-navigation` branch starts at fork default `main` (`fead66b`),
then selectively ports the tested observation/gaze foundation. The matching RL branch starts
at fork default `develop` (`cb70b79`). Historical results below remain the record of the earlier
prototype; they do not establish completion of the current voice-navigation objective.

Current integration evidence:

- 460 navigation Python tests (plus 9 subtests), Ruff, and 217 Rust tests passed. The checks include
  audio resampling/staleness/disconnection, live session tool sequencing, voice cancellation,
  guard failures during motion/settling, and action-lock cleanup after broken telemetry.
- The actual `gemini-robotics-er-2-streaming-preview` endpoint accepted the declared tools and
  persistent Live API configuration using the saved credential.
- Real WebRTC/daemon/MuJoCo walking arcs agreed with independent truth: requested 0.10 m/+20°
  produced 0.0981 m/+14.47°; 0.10 m/−20° produced 0.1021 m/−25.03°; 0.15 m/0° produced
  0.1488 m/−11.56°. Odometry and truth differed by less than 0.6 mm/0.08°. All settled, but
  the third action correctly reported `target_reached: false`. Raw calibration records:
  `runs/20260916T215014Z-22087ab5`, `20260916T215045Z-4207ca43`, `20260916T215056Z-5a08dcc4`.
- Live audio run `runs/20260916T215552Z-7042f8bf` transcribed a synthesized WAV saying
  “Go to the kitchen,” accepted the goal, looked forward, and executed two walking actions.
  It terminated blocked at an obstacle; simulator truth confirms no arrival. An offline
  contact audit found a cube beside the selected starting foot position, confounding its
  unexpected steering response. That placement must not be used to select controller gains.
- Run `runs/20260916T220002Z-71bcf3a4` transcribed synthesized “Stop” audio and cancelled with
  zero actions and an acknowledged stop. This validates the actual model audio path, not a
  human microphone or physical-robot test.
- Clear-start audio run `runs/20260916T220548Z-75954af4` travelled a net 1.149 m before an
  obstacle stop. Exact-interval scoring of `voice-mission-002.truth.jsonl` found complete
  coverage, no obstacle contacts, and no falls; the robot stopped outside the kitchen.
  The model turned before confirming a doorway. Directional depth sectors, explicit doorway
  inspection, and bounded refusal recovery were added afterward; the 0.35 m obstacle guard
  remains unchanged. Recovery tests cover inspection, changed commands, readiness, retry
  limits, and fatal sensor/health/stop failures.
- Run `runs/20260916T221426Z-afde477e` travelled 0.802 m with no recorded collisions or falls,
  then stopped when a wide head sweep was corrected prematurely. Measured optical error was
  still falling quickly. Feedback now waits while the head approaches its target, preserving
  the 2 s timeout, 0.10 rad alignment tolerance, and correction limit. All seven subsequent
  real-daemon gaze checks passed in `runs/20260916T221950Z-0eba319c`, including left-to-right
  and right-to-left 90° sweeps; final errors were 0.063–0.086 rad in 1.35–1.61 s.
- Run `runs/20260916T222047Z-c5e37668` accepted the spoken kitchen instruction, scanned both
  sides, and travelled a net 1.755 m without collisions or falls. The streaming model then
  incorrectly claimed arrival while the final image showed a plain wall. Exact-run scoring
  of `voice-mission-004.truth.jsonl` rejected arrival; `goal_verified` remained false. This
  establishes that a model completion claim alone is insufficient evidence.
- The stateless standard ER 2 arrival reviewer rejected that plain-wall image and accepted
  three offline-rendered interior views showing an oven, faucet, and refrigerator. The
  latter is a perception fixture, explicitly not navigation evidence. In live run
  `runs/20260916T222830Z-7f5703b9`, fresh four-view scans rejected three premature arrival
  claims and allowed exploration between them. The mission then stopped blocked as designed.
  Exact scoring of `voice-mission-005.truth.jsonl` found 1.377 m net travel, complete coverage,
  no collisions or falls, and no arrival. The streaming navigator's visual interpretation
  remains the limiting issue; the independent review prevents a false success report.
- On the same recorded doorway image, the standard ER 2 endpoint selected a rightward camera
  inspection instead of claiming arrival. This single comparison motivated a separate
  `--visual-planner standard` mode: streaming retains speech and cancellation, while each
  visual action comes from a stateless standard-model request and the same guarded executor.
  Tests cover goal retention, tool restrictions, cancellation during planning, arrival-review
  feedback, and rejection of raw simulator truth. This does not itself establish a completed
  navigation route.
- Delegated run `runs/20260916T223657Z-36e766ed` travelled 1.613 m net and stored a grounded
  observation of the kitchen's counter and stove. After 44 tool calls the streaming session
  stopped requesting the next delegated step, causing the 45 s model timeout. Exact scoring
  of `voice-mission-006.truth.jsonl` found no arrival, collisions, or falls; camera age stayed
  below 79 ms and the control loop stayed at 49.65–50.36 Hz. This exposed a coordination
  dependency rather than a sensor or locomotion failure. The standard planner now runs its own
  serial loop after goal acceptance. Regression tests cover a silent voice session, a pending
  goal response, repeated goal confirmation, spoken/tool cancellation during planning and body
  movement, and the final allowed action settling before budget exhaustion.
- Autonomous run `runs/20260916T225107Z-fc9d519a` continued without further voice-model
  requests and stopped blocked after 42 calls. Exact scoring of `voice-mission-007.truth.jsonl`
  found 0.865 m net movement, complete coverage, no collisions or falls, and no arrival.
  Turn directions matched their requests; visual descriptions alternated between left and right
  after scans. The final obstacle refusal was followed by roughly 132 seconds of repeated
  camera scans. This motivates retaining labelled views from the same body position.
- The planner now receives measured optical yaw/pitch with its current image and up to two
  recent side scans. Scans expire after 30 seconds, 2.5 cm of body displacement, 5° of yaw,
  or a dispatched walking action. Integration tests verify that both side views survive
  recentering, the current guard remains authoritative, and exact selected JPEG bytes are
  recorded with view IDs. Invalid or unsynchronized camera orientation discards cached scans.
- The active-motion spoken-stop diagnostic in `runs/20260916T225938Z-323e2a10` failed:
  a valid synthesized “Stop” WAV began during an actual model-selected walking command,
  but no stop transcription arrived before the 120 s diagnostic deadline. The initial kitchen
  instruction was transcribed correctly. Active voice cancellation is therefore not yet
  validated; the earlier stationary-stop result does not establish interruption during motion.
  `scripts/validate_voice_stop.py` records the audio trigger, read-only simulator motion,
  transcription, terminal result, and final commands separately, with 22 offline regression tests.
- An offline 24-trial pure-yaw screen used the unchanged policy and navigation scene at the
  clear starting pose, commands of ±0.5/±0.8 rad/s for 1/2/3 seconds, and two repetitions.
  Active yaw excursions of 3.6–9.1° returned to within 0.63° after settling; final translation
  stayed below 0.48 mm, with no falls or obstacle contacts. This does not provide a usable pivot
  primitive. Results remain in the RL checkout's `artifacts/navigation/pivot-screening.json`.

Full visual exploration and kitchen arrival are still under development. All runtime navigation
decisions use robot camera/depth/odometry only. Simulator truth stays in post-run scoring.

## Earlier guarded prototype

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

The follow-up below tests gaze feedback and short forward response. Effective speed remains
unavailable to the visual agent pending repeatable heading and stopping control.

## Bounded gaze feedback and forward response

Gaze now adjusts a virtual IK target using half the measured optical direction error every
0.25 seconds. It preserves the neck command, caps target displacement at 35% of original
target distance, and retains a two-second budget. Success requires measured aim within the
unchanged 0.10-radian tolerance continuously for 0.15 seconds plus a fresh camera frame.
The original requested point is the success criterion, not the adjusted IK point. Joint-target
tracking is no longer a separate success condition: the daemon's measured-joint camera FK
already determines whether the camera is aligned. This is kinematic feedback, not independent
image-based calibration. Returned results include `aim_error_rad` and correction count.

`20260916T192130Z-be03911f/gaze.json`: all ten head-only tests passed (two passes through
forward, left, right, upward and forward targets). Calls took 1.13-1.35 seconds, used three or
four corrections, and finished with roughly 0.065-0.073 radians of error. All preserved zero
neck offset and acknowledged stop. `scripts/calibrate_gaze.py` reproduces this battery.

A preliminary sequential forward run (`20260916T192157Z-2cb6b181/forward.json`) produced
63.0 and 64.0 mm settled displacement at a 50 mm odometry stop cutoff, with -4.4 and -10.8
degrees heading change. A third attempt stopped on the obstacle guard and settled at 44.6 mm.
This run exposed heading overshoot as well as distance overshoot.

After resetting the flat simulator, `scripts/calibrate_forward.py` recorded
`20260916T192402Z-7d19aa2c/forward-summary.json`. The first probe exhausted its 1.5-second
budget without reaching the odometry cutoff; settled truth displacement was 53.5 mm with
-5.3 degrees heading change. The second reached the cutoff but settled at 62.3 mm with
-14.5 degrees heading change, so the battery aborted and exited 1. Both acknowledged stop,
with zero requested velocity and applied velocity below 0.001 after the settling wait.

The diagnostic's provisional pass criterion is three distance-cutoff stops, settled travel
50 +/- 20 mm, heading change at most 10 degrees, and acknowledged/zero-command stops. It
failed. Runtime observation and settled truth differ; a stop cutoff is not a guarantee on
final travel or heading. No model speed limit, policy, actuator, or training changes were made.

Live Gemini inspection `20260916T192433Z-66c64fad/mission.json` also completed: the model
looked left, recentered, then finished after three decisions in 10.58 seconds. Both gaze
actions settled, stop was acknowledged, and no body movement was requested. Its scene
description remains a model assessment (`goal_verified: false`), not an arrival result.

Validation: 97 Python tests plus 9 subtests passed, including correction convergence without
neck changes, bounded correction refusal and measured-camera completion despite joint bias.
Ruff passed. Rust code and the API 30 wire contract are unchanged in this follow-up.

Next locomotion work should diagnose heading drift and command-to-settled-motion response
using this reproducible benchmark before enabling a faster model action. Successful gaze
alone does not establish reliable room navigation.

## Heading drift phase traces

Six independent probes used the same flat scene and deployed policy, restarting the simulator
before **every** trial. Order was neutral, recentered, recentered, neutral, neutral, recentered.
Each probe retained ordinary guards, a 0.30 m/s command, 1.5-second budget, 50 mm odometry
cutoff and 10-degree heading cutoff. `scripts/trace_forward.py` records the full time series
and scores simulator truth separately from the controller. The policy SHA-256 was unchanged.

Heading changes in degrees (positive/negative are opposite yaw directions):

| Run under `runs/` | Posture | First 0.5 s | Rest of action | After action | Settled total |
|---|---|---:|---:|---:|---:|
| `20260916T193359Z-1b39eb86` | neutral | -1.4 | +4.3 | -9.5 | -6.6 |
| `20260916T193421Z-67ebb72a` | recentered | -1.7 | -5.4 | +6.3 | -0.8 |
| `20260916T193445Z-d4a8665e` | recentered | -1.6 | -3.0 | -8.3 | -12.9 |
| `20260916T193506Z-8d7063b9` | neutral | -1.7 | -0.9 | +5.1 | +2.5 |
| `20260916T193527Z-4adf2e59` | neutral | -1.5 | +6.4 | -3.9 | +1.0 |
| `20260916T193547Z-50fc1e9c` | recentered | -1.9 | -8.4 | -4.7 | -15.0 |

The first 0.5 seconds contributed a consistent small negative rotation (about 1.4-1.9 degrees).
Larger rotation developed later and through stopping. Post-action heading change ranged from
-9.5 to +6.3 degrees and sometimes reversed the earlier drift. Thus the drift is not merely
a fixed steering offset during forward travel. Both postures exhibit it; recentered trials
were worse overall in this small battery, so posture may still influence it.

All requested and applied yaw-rate samples were exactly zero. Forward applied velocity
continued decaying after stop: in the first four traces it was about 0.021-0.026 m/s after
0.25 seconds and 0.001-0.002 m/s after 0.5 seconds. Most post-action rotation occurred in the
first second. This is evidence of substantial gait/settling dynamics around the stop transition,
not proof that command smoothing alone is the cause. Dependence on gait phase is a plausible
explanation, not isolated by these measurements. There is no outer heading-hold correction
in the forward primitive.

Final requested commands were zero, applied commands were below 0.001, and stop was acknowledged
in every trial. Final 0.3-second heading variation was below 0.10 degrees. Maximum sample gaps
were below 0.030 seconds; simulation ran approximately in real time. Travel ranged from 51.5 to
66.9 mm. The last recentered trial hit the heading cutoff yet settled near -15 degrees, again
showing that a cutoff does not bound the final rotation. Guarded cutoffs changed action duration
(about 1.11-1.53 seconds), and recentering changes preparation timing as well as posture, so this
battery cannot assign causality to a particular joint, foot contact or filter setting.

The immediate control target is the full move-and-settle behavior: a heading controller that
only watches the active command would miss a substantial source of error. Before choosing a
controller or changing training, characterize whether bounded yaw corrections remain effective
through deceleration; the existing turn benchmark already shows weak low-rate response.
No policy, actuator, gait setting or navigation speed limit changed in this investigation.

Validation: 99 Python tests plus 9 subtests passed; Ruff passed. New scoring regressions verify
phase attribution across the +/- pi wrap boundary and reject traces without a baseline.
Raw samples and camera snapshots are retained in each run's `events.jsonl`; `trace.json`
contains the phase summary. These ignored recordings are local evidence; the script, tests
and this numeric summary are committed for reproducibility.

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
