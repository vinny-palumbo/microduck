# Guarded visual-agent prototype

This prototype accepts a typed goal, sends a fresh camera image to one model, executes one
guarded action through WebRTC, and reassesses with the result and a new image. It tests visual
decisions and honest blocked/stuck reporting. Reliable room navigation is not demonstrated;
the default bounded commands produce little settled movement. Higher-speed simulator probes
show movement, but are not calibrated navigation actions.

Run from this checkout: the transport reuses `spaces/policy-shop/lan.py` and
`spaces/shared/control.py`. This is not a standalone wheel for distribution.

## Start the simulator

Follow [the simulator setup](../../docs/robot/simulation.md). Start one duck, headless, with its
camera in the flat apartment variant:

```sh
cd ~/Pollen/microduck
DUCK_SIM_SCENE=apartment_flat DUCK_SIM_CAMERAS=a DUCK_SIM_VIEWER=0 scripts/duck-sim
scripts/duck-sim status
scripts/duck-sim realtime
```

If your WebRTC GStreamer plugin was built locally, set `GST_PLUGIN_PATH` to its build directory
before starting. On the development machine used for this milestone:

```sh
export GST_PLUGIN_PATH="$HOME/.cache/duck-sim/gst-build/release${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"
```

The flat scene covers the original apartment's stairwell. It is a controlled, level-floor test
environment; the bridge has no reliable drop-off detector. Keep other gamepad/browser clients
from driving during a run: the robot's existing command interface is last-writer-wins, and this
client cannot acquire exclusive control.

Use the daemons built from this checkout (IPC API 30). The gaze conversion fix changes
`robot.look` to return resendable policy offsets in `head` and absolute measured-joint targets
in `joint_targets`; the bridge requires both, plus `robot.model.joint_home` to preserve commanded neck posture. The launcher rebuilds the local daemons. A
physical robot would need the matching daemon release installed before using this client.

## Use the tools

In a second terminal (WSL/Linux on Windows):

```sh
cd ~/Pollen/microduck/apps/navigation
uv sync --locked
uv run duck-nav observe
uv run duck-nav move_for --speed 0.05 --seconds 0.5
uv run duck-nav turn_by 15
uv run duck-nav look_at 1 0 0
uv run duck-nav stop
```

The default endpoint is the local simulator at `127.0.0.1:8443`. `--host` and `--port` precede the
command. The bridge never enables motors or changes policies. Initialize those through the
simulator launcher. `observe` reports why movement is unavailable, and an action can return
`completed: false` with its stop reason instead of moving.

`duck-nav tools` emits the tool descriptions and parameter schemas without connecting. To keep
one session open, use `uv run duck-nav serve` and send one JSON object per line:

```json
{"id":1,"tool":"observe"}
{"id":2,"tool":"move_for","arguments":{"speed_m_s":0.05,"duration_s":0.5}}
{"id":3,"tool":"turn_by","arguments":{"angle_deg":15}}
{"id":4,"tool":"stop"}
```

Wait for each result before sending the next action. A `stop` request is accepted immediately
while an action is running; other overlapping requests are refused. Stop an in-progress CLI
action with Ctrl-C; the async Python API also supports `await robot.stop()` from another task.
No command is repeated while waiting for the next input line. Closing stdin cancels the active
action, ends the session and requests a stop. Replies carry the request's `id` because a stop
reply may precede the cancelled action's reply.

Every invocation saves an `events.jsonl` and upright camera JPEGs under `runs/<timestamp-id>/`.
Each finished action records the result and current state/depth/camera metadata. These are
sensor observations, not simulator ground-truth coordinates or room labels. Use `--runs-dir`
to choose another location. Recordings are ignored by git.

## Run a visual mission

The first adapter uses [Gemini Robotics ER 2](https://ai.google.dev/gemini-api/docs/generate-content/robotics-overview)
through the [generateContent API](https://ai.google.dev/api/generate-content#FunctionDeclaration).
Each decision sends the typed goal, one JPEG, robot odometry,
depth summary and up to eight recent actions/results to Google. No simulator ground-truth
coordinates or room labels enter the planner. Each request is independent and returns exactly
one function call. Parallel calls, unknown tools, malformed arguments and truncated responses
end the mission with a stop request.

On Windows/WSL, save the key once in Windows Credential Manager from an interactive terminal:

```sh
uv run duck-agent-key save
uv run duck-agent-key status
uv run duck-agent "Look around and describe what prevents you from moving toward the kitchen"
```

`save` prompts with hidden input, stores a generic credential named `Pollen/Microduck/Gemini`
for the current Windows user on this computer, and verifies it by reading it back. Repeating
`save` updates that same entry. `status` reports presence only. The agent automatically loads
this entry when neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` is set; those environment variables
remain supported and take precedence in that order. Other Linux/macOS installations use the
environment variables. Scripted runs never read credentials.

The WSL bridge requires Windows interop and `powershell.exe` on PATH. It uses Windows
[CredWrite/CredRead](https://learn.microsoft.com/en-us/windows/win32/api/wincred/nf-wincred-credwritew)
with local-machine persistence scoped to your Windows user. The key crosses captured process
pipes, not command-line arguments, and is not written to source files, recordings or shell
profiles. Windows Credential Manager protects it at rest; programs running as your Windows
user can retrieve it. This is not isolation from other programs running as you.

To manage or remove the saved entry, open **Credential Manager -> Windows Credentials ->
Generic Credentials -> Pollen/Microduck/Gemini**. Removing it prevents automatic loading;
an environment variable, if set, still overrides the vault.

For an environment-only session instead:

```sh
read -rsp 'Gemini API key: ' GEMINI_API_KEY; printf '\n'
export GEMINI_API_KEY
uv run duck-agent "Describe the current view and finish without moving"
unset GEMINI_API_KEY
```

Use `--model` to select a compatible model, `--host`/`--port` for the WebRTC endpoint, and
`--runs-dir` for recordings. Defaults are 12 decisions and 120 seconds; `--max-steps` and
`--max-seconds` allow at most 30 decisions and 300 seconds. Each model request has a 30-second
timeout. Ctrl-C cancels the mission and requests stop. Existing tool bounds remain unchanged;
the adapter cannot enable motors or bypass guards.

A fresh frame with zero requested and near-zero applied commands is required before each
model decision. This checks command settling, not physical stillness. After movement, the
agent measures net odometry displacement/rotation. Two consecutive movement attempts that
fail or produce less than 5 mm forward displacement / 1 degree of turn end the run as `blocked`.
The model is instructed to recenter before its first body action and finish blocked after the
first failed/negligible body action; the two-attempt cutoff is a local fallback if it does not.
Intervening looks or observations do not reset that count. These conservative prototype
thresholds detect ineffective commands; they are not a calibrated distance controller.

Recordings include images, observations, decisions/reasons, action results, estimated progress
and `mission.json`. `goal_observed` is explicitly the model's assessment, with
`goal_verified: false`; it is not independently scored arrival. Exit 0 means that assessment,
2 means blocked/budget/mission failure, 1 means setup failure, and 130 means operator interruption.

Without a key, exercise the same loop using clearly labelled fixtures:

```sh
uv run duck-agent "Inspect the room" --scripted examples/inspect.json
uv run duck-agent "Try a short forward move and report if stuck" --scripted examples/stuck.json
```

The first checks gaze/observation/finish plumbing;
the second exercises negligible-progress
handling on the current simulator gait. Expected exit is 2 for these blocked outcomes.
`model: scripted-fixture` in the recording identifies these as plumbing tests, not evidence
of visual understanding. The [validation record](VALIDATION.md) distinguishes them from live
model testing.

## Guard contract

- Forward motion is limited to 0.10 m/s and 2 seconds per call. Reverse and sideways movement
  are unavailable because the forward sensor does not observe those directions.
- Turn requests are limited to 30 degrees per call and use odometry with a timeout and a
  no-progress check. Command duration is capped by the requested angle and yaw rate. An
  under-tracking gait returns an incomplete result with the measured angle; it is not retried
  automatically.
- Gaze preserves commanded neck posture, distinguishes policy offsets from absolute joint
  targets, and uses bounded feedback to correct measured camera direction. Aim must remain
  within 0.10 radians for 0.15 seconds, with a subsequent fresh camera frame. Neck tracking
  bias cannot accumulate through repeated looks. Each action has a two-second budget;
  correction displacement is capped at 35% of the original target distance.
- Camera, robot state, depth, and health must be fresh. Repeated source timestamps do not count
  as new sensor observations. Missing or malformed data refuses movement.
- Obstacle checks use the published ToF beam directions and current sensor pose, distinguish
  floor returns using gravity and trunk height, and reject untrusted near-field returns.
  Turning the head away from the travel direction prevents forward movement.
- Commands are refreshed only for the bounded action. A blocked, cancelled, expired, or failed
  action requests `robot.stop`. If the transport is lost, the daemon's command timeout remains
  the fallback; this app cannot acknowledge a stop over a disconnected channel.

This guard is a simulator prototype, not complete physical collision protection. A forward
8×8 sensor has blind spots around the body and feet and cannot establish clearance for every
part of a turn. Drop-offs, transparent obstacles, moving objects outside its view, calibration
error and simultaneous controllers need further work before autonomous use on hardware.

## Verify

```sh
uv run pytest
uv run ruff check duck_nav tests scripts
```

Tests exercise refusal paths, cancellation, connection failures and stale observations without
needing a model key or robot. A live smoke run must additionally establish camera/depth/state
arrival, a bounded movement followed by zero applied velocity, and a healthy simulator.

With the flat simulator freshly started and the path ahead clear, run the live check (it moves
the simulated duck and saves `smoke.json` alongside the frame/event recording):

```sh
uv run python scripts/smoke.py
```

The daemon slews applied velocity, so the live check requires exactly zero requested velocity
and applied velocity below 0.001 within two seconds. A tool's stop acknowledgement alone means
the daemon accepted the intent; it does not establish physical settling.

## Measure turn accuracy

Restart the flat simulator using the startup command above to reset placement and persistent
head commands. Then run:

```sh
uv run python scripts/calibrate_turns.py --label baseline --repeats 2
```

The benchmark requests left/right 5°, 15°, and 30° turns through the ordinary guarded action.
It reads ground truth from the local MuJoCo body endpoint (port 7801) for evaluation only;
the controller still sees WebRTC observations. It records the configured policy's SHA-256,
guard settings, initial posture, before/after frames, timestamped commands, odometry and true
heading in `runs/*/events.jsonl`, with a summary in `turns.json`. Repeats run sequentially;
they do not teleport/reset the duck between trials.

The score compares 0.3-second mean headings before the action and after a two-second wait.
A trial passes only if the action completed, stop was acknowledged, final heading error is
within 2° (1.25° for a 5° request), heading variation in the last window is at most 0.5°,
odometry agrees with truth within 0.5°, and the final requested/applied yaw command is zero
/ below 0.001 rad/s. These are benchmark acceptance criteria, not hardware safety guarantees.
Exit status is 0 for a passing battery, 1 for failed accuracy, and 2 for a guard/stop abort.
Connection or recording errors also exit nonzero. Partial summaries set `complete: false`.

For gait diagnosis, simulator-only rate probes are also available:

```sh
uv run python scripts/calibrate_turns.py --label response --probe-rates 0.6 -0.6 1 -1
```

Probe mode retains sensor, depth, health and deadman checks, but replaces the normal turn's
angle-command budget and no-progress cutoff with a two-second duration cap and a stop request
at 25° measured excursion. Rates are capped at 1 rad/s; stopping can still overshoot. Probe
results have no target-angle pass/fail score, and a successful exit only means the diagnostic
finished. These rates are not exposed by `turn_by`. Any guard abort ends the battery.

The [validation record](VALIDATION.md) fails the turn acceptance criteria. Keep this benchmark
as a separate locomotion diagnostic; precision turns do not gate testing the visual-agent loop.
Improve locomotion only through a separately scoped investigation. Do not enlarge guard limits
or retrain a policy simply to make a visual-agent demo appear successful. Reliable arrival will
need demonstrable movement and independent scoring when that milestone is attempted.

## Measure gaze and forward response

With the local flat simulator running and no other controller:

```sh
uv run python scripts/calibrate_gaze.py
```

This requests ten head-only looks across forward, left, right and upward targets. Each result
records optical error, correction count, preserved neck command and camera frames; `gaze.json`
reports whether every target completed. Feedback adjusts a virtual target passed to the
daemon's IK every 0.25 seconds using half the measured direction error. The original target
is always the success criterion; daemon travel limits still apply. Correction-limit, clamp,
stale-data and timeout failures request stop. This uses measured joint-derived camera poses,
not visual feature matching, so it does not independently validate physical camera calibration.

Restart the simulator to reset placement before the separate forward diagnostic:

```sh
uv run python scripts/calibrate_forward.py
```

This simulator-only script runs up to three sequential 0.30 m/s probes after recentering, with
ordinary sensor guards, a 1.5-second command cap, and stop requests at 50 mm odometry or 10°
heading excursion. It reads MuJoCo truth for evaluation only. The model cannot request this
speed. Stops can overshoot; these are diagnostic cutoffs, not movement guarantees.
A battery passes only if all three reach the distance cutoff, settled travel is within
50 ± 20 mm, settled heading change is at most 10°, and stop/zero-command checks pass.
It aborts after a guard refusal or excessive settled heading change; exit 1 means failure.
`forward-summary.json` and frame/event recordings retain the evidence. These initial
acceptance thresholds are diagnostic, not sufficient certification for navigation.

## Trace where heading drift develops

Restart the flat simulator **before every trial**, then run one of:

```sh
uv run python scripts/trace_forward.py --posture neutral
uv run python scripts/trace_forward.py --posture recentered
```

Use at least three trials per posture, alternating order. Each invocation runs one forward
probe with the same bounds as the forward diagnostic and saves `trace.json` plus roughly
50 Hz samples in `events.jsonl`. Samples contain simulator yaw/position, odometry,
requested/applied velocity, head commands, measured joints and IMU angular velocity.
Simulator truth is used only for scoring. The recording includes the policy hash and guard
configuration; it requires the local body endpoint before issuing commands.

The summary splits net heading change into the first 0.5 seconds, the remainder of the
action, and post-action settling (mean of the final 0.3 seconds of a two-second wait).
Action end is the tool's return after requesting stop; it is not the instant physical motion
ends. Angles are unwrapped before phase attribution. The split is a diagnostic convention,
not a detected gait-phase boundary. Exit 0 means the trace was recorded, not accurate walking.
Inspect cutoff, guard outcome, sample gaps and final zero commands alongside the heading values.
Different cutoff times and the extra recentering operation limit causal comparisons between
postures; these small batteries locate drift but do not isolate individual gait parameters.
