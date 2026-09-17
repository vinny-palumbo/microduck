# Voice-guided navigation

`duck-voice` listens for an instruction such as “go to the kitchen,” sends the duck's camera
and microphone to Gemini Robotics ER 2, and keeps choosing short guarded actions until it
reports arrival, becomes blocked, or is stopped. This page owns the
live voice workflow. The [navigation bridge README](../../apps/navigation/README.md#guard-contract)
owns the sensor, gaze, and stop guards; [the simulator guide](simulation.md) owns simulator setup.

This is work in progress. Live model runs have accepted a prerecorded spoken kitchen instruction
and travelled up to 1.93 m from the start without collisions or falls. The best clean run reached
the doorway, but full entry has not yet been demonstrated; independent scoring rejected
premature model arrival claims. Trial 011 explored the wrong room and contacted furniture
outside the forward depth sensor's view, so it failed the collision criterion. These
recordings test speech input through the API, not a person speaking into a live microphone.
Physical-robot navigation has not been validated.

## Prepare the two checkouts

Use the `voice-guided-navigation` branches in both `microduck` and `microduck_rl`. They start from
the repositories' default branches. The simulator's `navigation` scene, start-pose flags, and
evaluation recorder require the matching `microduck_rl` branch; an older body server does not
have those flags. Follow [simulator setup](simulation.md#what-you-need) for the daemons, policies,
MuJoCo environment, and camera plugin.

Run the bridge from its checkout in WSL/Linux:

```sh
cd ~/Pollen/microduck/apps/navigation
uv sync --locked
uv run duck-agent-key save
uv run duck-agent-key status
```

On Windows/WSL, `save` accepts the key through a hidden interactive prompt. Existing saved keys
also work with `duck-voice`; supported environment variables and credential storage are described
in [the bridge's credential instructions](../../apps/navigation/README.md#run-a-visual-mission).
The account must have access to `gemini-robotics-er-2-streaming-preview` and the standard
`gemini-robotics-er-2-preview` endpoint used for a separate visual arrival assessment.

The [robotics streaming endpoint](https://ai.google.dev/gemini-api/docs/robotics-streaming) accepts
camera, speech, and text input and returns text and tool calls. Audio replies use a separate local
speech program, described below.

The default `--visual-planner standard` uses the streaming session for spoken instructions and
cancellation, then runs an autonomous visual loop with the standard ER 2 endpoint. Each
decision receives the current image, up to two recent side views, guarded sensor context,
actual recent actions, and remembered observations. Images carry measured camera angles in the
body frame so a sideways view cannot be mistaken for the body's forward direction. Earlier
views expire after 30 seconds, 2.5 cm of body displacement, 5° of body rotation, or a dispatched
walking action. Older scans are presented first and the current image last. Current sensor
guards remain authoritative. Once a goal is accepted, the loop
keeps taking guarded steps while the speech
session listens for cancellation; continuing does not require another spoken or model request.
Use `--visual-planner streaming` to compare the original mode, where the streaming model chooses
the physical actions itself. Both modes use the same movement guards and arrival reviewer.

## Start a navigation simulator

In the first terminal, stop any previous simulator before changing its scene or initial pose.
Create a new evidence directory for each run; the simulator deliberately refuses to overwrite an
existing evaluation log.

```sh
cd ~/Pollen/microduck
scripts/duck-sim down
mkdir -p "$HOME/Pollen/navigation-evidence"
evaluation_dir=$(mktemp -d "$HOME/Pollen/navigation-evidence/run.XXXXXX")
DUCK_SIM_SCENE=navigation \
DUCK_SIM_CAMERAS=a \
DUCK_SIM_VIEWER=0 \
DUCK_SIM_START_X=0 \
DUCK_SIM_START_Y=0 \
DUCK_SIM_START_YAW_DEG=90 \
DUCK_SIM_EVALUATION_LOG="$evaluation_dir/truth.jsonl" \
scripts/duck-sim up
scripts/duck-sim status
scripts/duck-sim realtime
```

`navigation` selects `scene_navigation.xml`, an apartment with visible kitchen details, including
a stove, sink, cabinets, and refrigerator. Initial x/y values are world metres; yaw is degrees,
with +90 facing world +y. These are simulator placement settings, not instructions or coordinates
given to the model. The example starts at the scene origin and does not precompute a route.

`DUCK_SIM_EVALUATION_LOG` is optional. It writes simulator truth for scoring after the run; it is
never part of WebRTC observations, model context, or tool results. Keep that file out of the
planner. For camera plugin configuration and real-time requirements, use
[the simulator guide](simulation.md#a-room-to-look-at-and-eyes-to-look-with).

## Speak a destination

In a second terminal:

```sh
cd ~/Pollen/microduck/apps/navigation
uv run duck-voice --audio mic
```

Wait for the `listening` event, then say “go to the kitchen.” The simulator supplies camera,
depth, and robot state; this command takes speech from the bridge computer's microphone. Under
WSLg, PulseAudio exposes the Windows microphone. Capture uses `parec` or `parecord` when present,
with FFmpeg as a fallback. To select a PulseAudio source, add `--mic-device SOURCE`.

Say “stop” or “cancel the mission” to end the run. Ctrl-C also cancels and requests a stop.
Standard mode sends explicit audio activity boundaries to preserve short follow-up commands:
it retains 200 ms of preceding PCM and ends an utterance after 500 ms of quiet audio, with a
15 s segment limit. This energy detector does not recognize words; cloud transcription still
determines spoken cancellation. Loud background noise can form a segment, and very quiet
speech can be missed. Microphone and motor-noise performance still need hardware validation.
Run one controller at a time: another browser or gamepad can overwrite robot commands; the bridge
cannot acquire exclusive control. A finished or blocked session exits. Start another invocation
for a new mission.

Use a recorded instruction to repeat the audio path:

```sh
uv run duck-voice --audio-wav /absolute/path/go-to-the-kitchen.wav
```

The file must be uncompressed 16-bit PCM WAV, mono or stereo. It is resampled to 16 kHz mono and
sent at playback speed. `--audio-wav` replaces the live microphone. A typed instruction is also
available for separating navigation debugging from speech recognition:

```sh
uv run duck-voice --audio none --goal "go to the kitchen"
```

The default audio source is the robot's WebRTC microphone. For a physical duck whose matching
daemons are installed and whose control state is already ready:

```sh
uv run duck-voice --host duck.local --audio robot
```

Replace `duck.local` with the duck's reachable host name or address; use `--port` if its WebRTC
endpoint differs from 8443. Connection details belong to [WebRTC](../design/remote-webrtc.md).
The bridge does not enable motors or select a policy. The physical sensor limitations are listed
in the [guard contract](../../apps/navigation/README.md#guard-contract).

To hear model updates on the bridge computer, install `espeak-ng` or `espeak` and add `--tts local`.
The default is `--tts none`, which still displays text. Local TTS plays on the computer, not the
duck's speaker; the microphone remains live so spoken cancellation can interrupt playback.

## What the model can do

Every tool is blocking and runs one at a time. The movement layer exposes walking arcs because
the current gait steers more effectively while walking than while attempting to pivot in place.
In standard planning mode, the visual loop owns physical actions; the speech session accepts
goals and cancellation. Both use the same stop mechanism.

| Model tool | Bridge operation |
|---|---|
| `start_navigation(goal)` | Accept the spoken task; preserve the original goal if it is already active. |
| `observe()` | Read current camera/sensor readiness, depth summary, and robot odometry. |
| `look_at(x, y, z)` | Aim the camera toward a trunk-frame point in metres: x forward, y left, z up. |
| `advance(distance_m, heading_deg=0)` | Walk a short arc, stop, and check settling. Zero holds the requested course; nonzero changes it relative to the current body heading, positive left. |
| `remember_place(name, observation, explored)` | Retain visual observations and explored places in the current mission. |
| `say(message)` | Display a brief update and optionally play local TTS. |
| `stop()` | Cancel the mission and request that the duck stand still. |
| `finish(status, reason)` | End with `goal_observed` or `blocked`, citing visual evidence or the obstruction. |

`advance` currently accepts 0.05–0.20 m and headings within ±30°. Its result reports actual measured
movement; a completed, settled action does not promise an exact target pose. The next decision
uses the resulting camera image and odometry. Its walking command calibration is separate from
the legacy `duck-nav move_for` and `duck-nav turn_by` limits documented in the bridge README.

Zero heading continues the last requested heading in the robot's odometry frame, so it can
curve to correct drift left by an earlier step. A nonzero heading sets a new target relative
to the current body direction. Observations expose the course target and current error;
results distinguish the supplied heading from the effective correction. A stop, failed action,
reinitialization, or unexpected movement clears the retained target. A correction beyond ±30°
is refused. The planner still needs room for the whole corrective arc; holding a heading does
not guarantee a straight path or an exact final pose.

Depth observations include left, center, and right sectors in the robot's trunk frame, plus the
sensor's current yaw. Side scans report what the sensor sees while keeping forward movement
disabled until the head is recentered. Missing obstacle returns do not certify clear space.
Close obstacles observed in side scans remain latched after recentering. A fresh scan of the
same area must positively establish clearance before body motion can resume; waiting alone
does not clear the warning. Changed body pose or a full hazard store stops the mission for
operator inspection. Restart only after clearance has been independently confirmed. This
retains observed hazards but cannot detect obstacles that the sensor has never seen.

Camera captures are limited to one per second. In standard mode, the local navigation loop
requests each next decision after the previous action completes. In streaming mode, a heartbeat
asks the model to continue only after its preceding turn completes. Recent action history and
remembered places support exploration without a supplied floor plan. Place memory is saved in
the recording, but is not automatically loaded into a later run.

Before accepting an arrival claim, the bridge stops and collects fresh forward and side views.
A separate, stateless visual reviewer receives only those images and the destination. It checks
room identity and full-body entry separately, including near-floor transitions and doorjambs in
side views. If it cannot establish that the duck has crossed the entrance, its visual evidence
is returned to the navigator so exploration can continue. Three rejected claims end the mission
as blocked. This second model assessment can still be wrong; simulator truth remains the
independent arrival check.

For kitchen goals, both visual stages require an identifiable oven, cooktop, sink with faucet,
or refrigerator. Tables, boxes, cabinets, and floor colors alone cannot identify the destination.

The model is instructed to enter the destination, not merely see it through a doorway. An
acknowledged obstacle, gaze, or depth-quality stop permits bounded replanning: inspect the route,
recenter the camera, and advance only after the guard is ready. Repeating a failed command also
requires a cleared obstruction; otherwise the action must change. Three consecutive refusals
end the mission. Negligible movement permits the same recovery only after physical settling is
verified. Stale sensors, health failures, and unacknowledged stops end the mission immediately.
Standard navigation also stops after 12 visual observation or memory decisions without completed,
measured walking progress. Head movements do not reset this budget. The planner receives the
remaining budget so repeated scans of an unchanged obstruction end promptly.
Tool cancellation, lost transport, model failure, exhausted budgets, and operator cancellation use the existing
[stop guard](../../apps/navigation/README.md#guard-contract).

| Option | Default |
|---|---|
| `--max-actions` | 600 total tool calls, including observation, memory, and speech calls |
| `--max-seconds` | 1800 seconds for the mission |
| `--model-timeout` | 45 seconds without model progress while a response is expected |
| `--instruction-timeout` | 120 seconds to receive the initial spoken task |
| `--runs-dir` | `runs` beneath the current working directory |

These are upper bounds, not a promise that the robot will navigate for the full duration or reach
the room. Each physical action also has its own guard and timeout.

## Inspect the result

Each run saves `events.jsonl`, camera JPEGs, and `mission.json` in its printed recording directory.
Events include speech transcriptions, model tool requests, action results, measured progress, and
failures. `mission.json` includes the goal, terminal status, stop acknowledgement, elapsed time,
and remembered places. `goal_observed` is the model's assessment and always carries
`goal_verified: false`.
The recording also identifies the visual planning model so runs from the two modes can be compared.

Exit 0 means that visual assessment; exit 2 means a blocked, cancelled, or failed mission; exit 1
means setup failure; and Ctrl-C exits 130. Interpret stop acknowledgements using the
[guard contract](../../apps/navigation/README.md#guard-contract).

After the mission ends, score the independent simulator log from the first terminal:

```sh
scripts/duck-sim down
cd ~/Pollen/microduck_rl
uv run python -m mjlab_microduck.sim.mission_eval "$evaluation_dir/truth.jsonl" \
  /absolute/path/to/the/printed/navigation/run
```

Keep the score, model recording, and initial simulator placement together. The scorer verifies
the duck's position and settled arrival within the exact mission interval using simulator truth.
It also checks recorded obstacle contacts, falls, stop acknowledgement, and the model's claim.
The model does not receive that
information. A successful replayed speech test, a successful live-microphone test, and successful
physical-robot navigation are distinct evidence and should be reported separately.

To repeat the visual arrival checks without connecting to a robot:

```sh
cd ~/Pollen/microduck/apps/navigation
uv run python scripts/validate_arrival.py
```

This makes eight provider requests against versioned doorway, interior, wall, and non-kitchen
images, verifies their checksums, and saves a JSON report under `runs/`. Expected verdicts and source provenance
stay in the local scorer; the model receives only the destination and camera images.

To test cancellation during motion, start a fresh local simulator and run:

```sh
cd ~/Pollen/microduck/apps/navigation
uv run python scripts/validate_voice_stop.py \
  --goal-wav /absolute/path/go-to-the-kitchen.wav \
  --stop-wav /absolute/path/stop.wav
```

The diagnostic sends the second WAV only after the model selects a walking action and fresh
telemetry shows an active nonzero command. It separately checks physical displacement using
the local simulator's read-only body protocol; those samples never enter model inputs. The
recording includes `voice-stop-validation.json`. Passing requires a stop transcription before
termination, cancellation, an acknowledged stop, and final zero commands. This does not certify
physical settling or live-microphone performance. See the current results in
[VALIDATION.md](../../apps/navigation/VALIDATION.md#voice-branch-integration--2026-09-16).
