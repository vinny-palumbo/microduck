# Voice-guided navigation

`duck-voice` listens for an instruction such as “go to the kitchen,” sends the duck's camera
and microphone to Gemini Robotics ER 2, and keeps choosing short guarded actions in one
conversation until it reports arrival, becomes blocked, or is stopped. This page owns the
live voice workflow. The [navigation bridge README](../../apps/navigation/README.md#guard-contract)
owns the sensor, gaze, and stop guards; [the simulator guide](simulation.md) owns simulator setup.

This is work in progress. A live model run has accepted a prerecorded spoken kitchen instruction,
walked twice, and stopped at an obstacle. It has not yet demonstrated a complete route into the
kitchen. That recording tests speech input through the API, not a person speaking into a live
microphone. Physical-robot navigation has not been validated.

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
The account must have access to `gemini-robotics-er-2-streaming-preview`.

The [robotics streaming endpoint](https://ai.google.dev/gemini-api/docs/robotics-streaming) accepts
camera, speech, and text input and returns text and tool calls. Audio replies use a separate local
speech program, described below.

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
DUCK_SIM_START_YAW_DEG=0 \
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

| Model tool | Bridge operation |
|---|---|
| `start_navigation(goal)` | Accept the spoken task; preserve the original goal if it is already active. |
| `observe()` | Read current camera/sensor readiness, depth summary, and robot odometry. |
| `look_at(x, y, z)` | Aim the camera toward a trunk-frame point in metres: x forward, y left, z up. |
| `advance(distance_m, heading_deg=0)` | Walk a short arc using robot odometry, stop, and check settling; positive heading steers left. |
| `remember_place(name, observation, explored)` | Retain visual observations and explored places in the current mission. |
| `say(message)` | Display a brief update and optionally play local TTS. |
| `stop()` | Cancel the mission and request that the duck stand still. |
| `finish(status, reason)` | End with `goal_observed` or `blocked`, citing visual evidence or the obstruction. |

`advance` currently accepts 0.05–0.20 m and headings within ±30°. Its result reports actual measured
movement; a completed, settled action does not promise an exact target pose. The next decision
uses the resulting camera image and odometry. Its walking command calibration is separate from
the legacy `duck-nav move_for` and `duck-nav turn_by` limits documented in the bridge README.

The model receives camera JPEGs at no more than one per second. A heartbeat asks it to continue an
active task only after the preceding model turn completes. Session history and remembered places
support exploration without a supplied floor plan. Place memory is saved in the recording, but
is not automatically loaded into a later run.

The model is instructed to enter the destination, not merely see it through a doorway. A failed
or negligible movement ends the mission as blocked. Tool cancellation, lost transport, model
failure, exhausted budgets, and operator cancellation use the existing
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

Exit 0 means that visual assessment; exit 2 means a blocked, cancelled, or failed mission; exit 1
means setup failure; and Ctrl-C exits 130. Interpret stop acknowledgements using the
[guard contract](../../apps/navigation/README.md#guard-contract).

After the mission ends, score the independent simulator log from the first terminal:

```sh
scripts/duck-sim down
cd ~/Pollen/microduck_rl
uv run python -m mjlab_microduck.sim.navigation_eval "$evaluation_dir/truth.jsonl"
```

Keep the score, model recording, and initial simulator placement together. The scorer verifies
the duck's position and settled arrival using simulator truth; the model does not receive that
information. A successful replayed speech test, a successful live-microphone test, and successful
physical-robot navigation are distinct evidence and should be reported separately.
