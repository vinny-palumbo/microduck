# Guarded navigation bridge

This is the first navigation milestone: observe a simulated Microduck and execute short,
checked actions through the same WebRTC session used by a robot. It does not yet choose a route,
recognize a kitchen, or call a language model. The JSON tools are the boundary for that next step.

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

Use the daemons built from this checkout (IPC API 29). The gaze conversion fix changes
`robot.look` to return resendable policy offsets in `head` and absolute measured-joint targets
in `joint_targets`; the bridge requires both. The launcher rebuilds the local daemons. A
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

## Guard contract

- Forward motion is limited to 0.10 m/s and 2 seconds per call. Reverse and sideways movement
  are unavailable because the forward sensor does not observe those directions.
- Turn requests are limited to 30 degrees per call and use odometry with a timeout and a
  no-progress check. Command duration is capped by the requested angle and yaw rate. An
  under-tracking gait returns an incomplete result with the measured angle; it is not retried
  automatically.
- Gaze preserves measured neck posture, distinguishes policy offsets from absolute joint
  targets, and waits for measured alignment and a subsequent camera frame within a timeout.
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
uv run ruff check duck_nav tests
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

The next milestone is a model adapter that receives `observe`, calls these tools, and records
its decisions until it identifies and enters the kitchen. First calibrate the short-turn
tracking limitation documented in [the validation record](VALIDATION.md). Then start with one model and score
arrival independently using simulator ground truth; voice and persistent room memory can then
build on that measured loop.
