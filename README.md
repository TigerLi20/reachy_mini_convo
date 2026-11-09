# Reachy Mini conversation and functionality tests

To set up a new checkout on macOS (Python 3.12 required):

```sh
git clone --recurse-submodules https://github.com/TigerLi20/reachy_mini_convo.git
cd reachy_mini_convo
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
```

For an existing clone, run `git submodule update --init --recursive` before
installing dependencies. The upstream conversation app is pinned to the commit
in `source_versions.json`. Face enrollments, relational memory, recordings, logs,
downloaded models, and `.env` credentials are local data excluded from Git.

Basic conversation includes phase 3 attention, phase 4 recognition, and phase 5
person-scoped relational memory in the same script,
separate from the phase 1 checks:

```sh
# Terminal 1: leave the daemon running
.venv/bin/reachy-mini-daemon

# Terminal 2: start a new conversation
.venv/bin/python basic_conversation.py
```

Keep Reachy and the reSpeaker connected to the Mac via USB, with Reachy's power
supply connected. `basic_conversation.py` also accepts `--input-device NAME_OR_INDEX`,
`--input-channel N`, and `--robot-name NAME` like the functionality test.
It uses the existing hosted Hugging Face voice backend. Phase 5 adds automatic
local model downloads; no additional API key or cloud LLM account is needed.

Phase 5 is on by default. Reachy first recognizes one enrolled person, then asks
whether to continue as that person and use their memories. Answer **"Yes"** only
when the proposed identity is correct. This confirmation opens a new private
backend context; no prior conversation details are loaded before confirmation.
Each new process/session needs this confirmation. A spoken name alone cannot
select an existing person's private history.

The memory hierarchy is:

| Layer | Implementation |
| --- | --- |
| Working dialogue | Up to 16 recent user/assistant transcripts (about eight exchanges), within a 17,000-character context-data budget |
| Summary buffer | A local FLAN-T5 model selects source excerpts; only validated user text enters the rolling summary |
| Personal facts | Clear first-person statements saved with a stable key, source turn, quoted evidence, and person ID |
| Episodes | Dated, searchable consolidation snapshots linked to their source dialogue and person ID |

Models are [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) for excerpt
selection and [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
for 384-dimensional text embeddings. They run on the Mac with bounded CPU threads.
Model weights download once into the Hugging Face cache. Face embeddings and text
embeddings serve separate purposes. Facts use the existing conversational model's
structured tool calls, followed by runtime provenance and owner checks. Questions,
hypotheticals, guessed values, or another person's source turns are not new facts.
Ambiguous wording/transcription should prompt clarification.

Authoritative storage is `.relational_memory/memory.sqlite3`. SQLite keyword
search and person-filtered vector ranking retrieve relevant facts/episodes.
Database rows are scoped by the runtime's confirmed person ID before ranking;
the LLM has no owner-ID parameter. Voice text is stored only for a currently
verified, bound person. Unknown visitors can chat temporarily without saving
relational history. Face/name enrollment remains in `.person_data/identities.json`.

After roughly eight user turns, the app consolidates older dialogue in a worker,
waits until the reply finishes, then opens a fresh backend connection with bounded
summary/recent context. It does not repeat the greeting or require identity
confirmation for this internal rollover. Short audio buffering during connection
setup is tagged to its context generation and discarded on a person change.
Full source turns commit incrementally. Interrupted consolidation is recovered
from committed, scoped dialogue on the next launch. Normal farewell/sleep also
requests final consolidation.

Private memory is paused with **any detected group of two or more faces**, including
known+known, known+unknown, and unknown+unknown. Reachy does not identify the speaker
from sound direction or select the largest face as the memory owner. Entry of a
second face or a recognized person change cancels private playback/results and
replaces the old backend context. Generic group chat is allowed afterward. Once
one person remains, recognition and explicit session confirmation are required
again. After initial recognition, Reachy greets the person by name before asking
to open their personal memory. Once confirmed, that session (including memory
reads/writes) survives visual uncertainty for up to 20 seconds while recognition
retries silently. Reacquisition within that window needs no new confirmation.
At 20 seconds of continuous uncertainty, private context ends and Reachy asks
once for a camera adjustment, then allows general conversation. An identity
correction also ends the session immediately. Attention states and smooth
orientation continue independently.

This depends on the camera actually detecting visible faces. There is no speaker
verification or off-camera voice attribution: test with one active speaker.
Someone speaking off camera while another person's face is visible cannot be
reliably distinguished by this implementation.
During the 20-second grace period, continuity assumes the confirmed person is
still speaking; it cannot detect an off-camera or unrecognized replacement speaker.

Corrections update the current fact, redact the obsolete value from scoped
dialogue/quotes, invalidate derived summaries/index entries, and replace active
context. Deleting a fact does the same and removes the fact itself. Unrelated
facts remain; derived episodes/summaries are rebuilt from sanitized source turns.
An episode deletion first explains the number of source messages/derived facts
it would remove and requires a new explicit yes. "Forget all our conversations"
clears that person's dialogue, facts, summaries, episodes, and indexes, while
retaining their face/name enrollment. Other people's records are unaffected.
Revision checks prevent pending worker jobs from restoring deleted content.

Phase 5 operational logs omit conversation/tool payloads. Attributed dialogue
resides in the scoped database rather than duplicate debug files. Logs from
earlier phases are neither imported nor automatically removed. Deletion concerns
local retrievable memory and the active inference context, not provider-side logs.
The built-in upstream `remember`/`forget` file remains unused.

To test live, enroll two actual people separately first; changing your spoken
name does not switch the biometric person ID. Confirm the memory session on
each run, wait until Reachy finishes speaking, and test these cases:

| Test | Expected behavior |
| --- | --- |
| A: "My favorite food is ramen"; B in another run: "My favorite food is pasta" | After restart and confirmation, each gets only their own preference |
| A: "My dog is called Milo"; B: "My dog is called Luna" | "What is my pet called?" retrieves the correct person's dog |
| A: "Actually, my favorite food is tacos now" | Immediate and restarted recall uses tacos; B still has pasta |
| A: "Forget my dog's name" | No dog name is recalled immediately or after restart; B's Luna remains |
| A: "Forget all our conversations" | A's relational memory is empty; face/name recognition and B's records remain |
| Tell a short event, chat for more than eight exchanges, then restart | Recent dialogue and searchable episodes preserve grounded continuity without another greeting during rollover |
| Unknown visitor alone, including someone who claims A's name | General conversation only; A's memories cannot be opened by name alone |
| A+B, A+unknown, or two unknown people together | Personal recall/saving/enrollment is blocked; only generic conversation is allowed |
| Second person enters while Reachy is recalling something private | Reply stops after detection; no further private history is given; a fresh private session is needed afterward |
| A leaves and B takes their place without restarting the script | No A details enter B's new context; B must be recognized and confirm |
| Leave the frame, then return | Memory operations pause; prolonged loss requires fresh session confirmation |
| Ctrl+C after a saved fact, then restart | Committed facts survive; unfinished summaries recover without cross-user assignment |

Look for `MEMORY operation=... status=saved/corrected/forgotten` and
`MEMORY context_reset reason=...` in the console. A blocked/failed operation is
not a successful save. Recognition quality, model tool use, and physical playback
still need these live checks; automated tests do not establish real-world identity
accuracy. `--no-memory` restores the phase 4 identity-only conversation. With
`--no-recognition` or `--no-attention`, phase 5 cannot open a private memory session.

Person recognition is enabled by default. On your first conversation, Reachy asks
what to call you. Give your own name and face the camera alone in good light for
about four seconds. Reachy prepares the enrollment and asks whether it should
remember your face and exact name. Answer **"Yes"** after it finishes asking.
It saves only after that later confirmation and a fresh, consistent set of face
samples. A quoted or third-person name is not an enrollment request. If the
name was misheard, correct it before confirming. A declined or incomplete
enrollment does not save anything.

End the conversation normally, then run `basic_conversation.py` again. Reachy
checks the current face before greeting you and should use your enrolled name
only after a sustained, strong match. Unknown or ambiguous faces prompt a neutral
question such as "What should I call you?" without suggesting a candidate name.
Correcting a recognized name suppresses that identity for the rest of the session
and does not silently overwrite its record. You can still give a temporary name
for the current conversation without saving a face.

Names, UUID person IDs, and 128-dimensional face embeddings are stored locally in
`.person_data/identities.json`, separate from temporary session data. No photos,
conversation facts, preferences, or histories are stored in that identity file.
The app's `remember`/`forget` tools remain disabled. Existing diagnostic transcript
logs still exist under `artifacts/`, but are not read as memory. To reset all
enrollments, stop the app and remove `.person_data/identities.json`. Use
`--no-recognition` to disable identity lookup and enrollment without deleting it.
`--no-attention` also disables recognition because its camera sensor loop is off.

Recognition runs locally using [OpenCV YuNet and SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
The checked model files download once to `.face_models/` (about 39 MB) and are
verified against pinned SHA-256 hashes. Only names/IDs for accepted matches are
sent to the conversation backend; face images and embeddings are not sent by
this recognition component. Diagnostic `recognition.jsonl` records state, reason,
accepted IDs, and similarity, excluding names and face vectors. Rejected matches
also log similarity and runner-up margin, without revealing a candidate name.

An accepted match needs cosine similarity at least 0.60, a 0.12 lead over the
next enrolled identity, and at least five consistent observations spanning 1.5
seconds within a four-second window. Brief missed detections or quality failures
return `unknown` immediately while preserving fresh matching progress for up to
1.5 seconds. A longer gap, multiple faces, a mismatching embedding, or an identity
correction clears that progress. Startup identity lookup waits up to six seconds
for usable evidence. Those are conservative engineering thresholds, not calibrated confidence
percentages. Multiple faces, weak/close matches, poor image quality, or face loss
produce `unknown` and clear any remembered-name result. Enrollment needs eight
consistent, clear samples and verifies that the same face is still present when
you confirm. This is for one person at a time; it does not bind voices to faces
or check liveness. Accuracy across your lighting and camera angles needs a live
enrollment/restart test.

The recognition detector uses an image up to 640 pixels wide with area resampling
to preserve facial landmark detail. Embeddings still use the aligned crop from
the original camera resolution. Rejected or blurred frames are not enrollment
samples, and enrollment always requires a currently usable face.

Attention starts automatically. Keep the external reSpeaker fixed beside the
robot's base, aligned using your convention: **0° forward, +30° left by 30°**,
330° right by 30°. No calibration prompts or settings are required. This assumes
the array remains fixed relative to the base; mounting it on the rotating head
would require transforming its bearings into the base frame.

Face tracking uses a local YuNet detector, not the conversation LLM. With person
recognition enabled, detection is shared with the SFace alignment pipeline;
with recognition disabled it uses the SDK's detector. The largest face is
selected initially; the closest face to the previous image position is followed
during continuous tracking. Gaze targeting uses image position, while recognition
independently verifies embeddings before supplying a person's stored name.
A visible face refines head yaw/pitch; voiced sound direction turns the head
toward an off-camera person so visual tracking can acquire them. Visual targets
are held for 1.2 seconds through brief losses; sound targets expire after 2 seconds.
When both expire, Reachy smoothly returns to neutral/idle. Sound direction is
ignored during speaker playback and its 0.5-second echo tail.

The head settles smoothly at a maximum of 30°/second, with yaw limited to ±90°
and pitch to -20°/+15°. Bearings outside that range turn only as far as the limit.
The SDK's automatic body-yaw control can rotate the body to support large turns
while keeping head-to-body rotation within its mechanical limits.
The existing movement loop remains the only app motor writer. Autonomous daemon
face tracking and speech wobbling are disabled while this controller is active,
and attention stops before the farewell sleep motion.

| Attention state | Trigger |
| --- | --- |
| `idle` | No fresh face, speech direction, or active conversational turn |
| `attending` | A face or recent speech bearing is present, between turns |
| `listening` | User speech is detected by the backend or array |
| `thinking` | Waiting for/generating a response, before audible output |
| `speaking` | Audio is queued for the speaker, including the echo tail |

These states describe attention and conversation status. They are logged, but
there is no separate gesture or antenna animation implemented for each state.
Face/sound gaze tracking runs across these states; speaker playback suppresses
microphone input and sound-direction steering to avoid reacting to Reachy's voice.

Transitions print `Attention: ...` in the console. `attention.jsonl` in the run's
artifact folder logs states, target source, and commanded yaw/pitch once per
second and on each transition. These are control targets, not measured motor poses.
If a sensor fails, a warning is logged and fresh observations from the remaining
sensor can still guide attention; stale observations expire. If USB direction
access is unavailable, conversation and face tracking continue. Use `--no-doa`
for face-only attention or `--no-attention` to return to phase 2 behavior.

To check phase 3, speak from the front, 30° left, and 30° right; walk slowly
across the camera view during a conversation; then leave the view and stay
silent for at least three seconds. Check smooth turns, tracking, state logs,
and return to idle. Finally end the conversation and check farewell/sleep.
Physical tracking accuracy and mechanical behavior need a live check.

Reachy greets you by a confidently recognized name or as a first meeting, then
chats with one person. Wait until it
finishes speaking, plus a brief pause, before replying: the external mic is
suppressed during speaker playback to prevent echo from triggering interruptions.
Try a few conversational turns, then say "I should get going" or "that's all for
today". The model should recognize that intent, call `finish_conversation`, speak
a short farewell, and then the runtime waits for the estimated speaker queue to
drain before invoking sleep and stopping the app. Ambiguous endings should prompt
clarification. End recognition is model-driven and needs verification in use.
If farewell audio never arrives, the sleep wait times out and leaves Reachy awake.
Ctrl+C stops manually; it does not request sleep. Run the script again for a new
session; the app wakes Reachy if it is asleep. No settings web UI runs in phase 2.

Each run uses a fresh temporary upstream app-data directory, removed on exit.
Our face enrollment and phase 5 relational database are separate from it.
With `--no-memory`, only identity check/enrollment, farewell, and task-management
tools are enabled; conversation facts are not loaded or saved across sessions.
Phase 5 enables only our custom scoped memory tools, not the upstream `remember`
or `forget` store. Diagnostic logs stay in `artifacts/conversation-*/` and are
never imported as memory. Older/`--no-memory` logs can contain transcripts;
phase 5 operational logs omit their payloads.

`conversation.log` contains app output. `turn_timings.jsonl` contains UTC and
monotonic timestamps plus elapsed milliseconds since the detected speech end:

| Event | What is measured |
| --- | --- |
| `speech_ended` | Backend VAD speech-stop event received by the client |
| `processing_started` | Backend response-created event received by the client |
| `first_reply_audio_submitted` | First response audio submitted to the SDK speaker queue |
| `farewell_playback_finished` | Estimated queued audio duration plus 0.5 s has elapsed |
| `sleep_requested` | Runtime begins the app's sleep/shutdown callback |

These are client observations, not server-internal compute timestamps. Speaker
submission is a playback-start proxy, not a measurement of physical sound onset;
driver buffering adds latency. Timing observation uses this pinned conversation
app's realtime event logs, so check it when upgrading the vendored app.

The local Python 3.12 environment contains the Reachy Mini SDK and a checkout
of the official conversation app in `vendor/conversation_app`.

Connect and power the Lite. Start its daemon using the desktop app, or in another terminal:

```sh
.venv/bin/reachy-mini-daemon
```

Stop any other running conversation app, then run:

```sh
.venv/bin/python functionality_test.py
```

The script captures and opens a JPEG through the app's camera tool, records five
seconds from an external reSpeaker USB Mic Array, saves a WAV and audio levels, and replays the
recording through the robot speaker. It then launches the official conversation
app with a dedicated test personality and web UI at http://127.0.0.1:7860.
The personality asks the backend to analyze a fresh camera frame at startup.
Use the printed spoken prompts to check vision, hearing, speech and head movement
(or an optional dance). These app tests are interactive; judge the description,
repeated phrase and physical movement yourself. A logged tool call alone is not
proof that the physical action or visual interpretation succeeded.

```sh
.venv/bin/python functionality_test.py --mode check         # imports only; no hardware
.venv/bin/python functionality_test.py --mode media --seconds 8
.venv/bin/python functionality_test.py --mode conversation  # app only
.venv/bin/python functionality_test.py --list-audio-devices
.venv/bin/python functionality_test.py --input-device "ReSpeaker 4 Mic Array"
.venv/bin/python functionality_test.py --audio-source reachy # original built-in microphone
```

Connect the reSpeaker to the Mac via USB (a USB hub is fine). The default input
is the unique device whose name contains `respeaker`, case-insensitively. If its
macOS name differs, use `--list-audio-devices`, then pass its index or a unique
name fragment with `--input-device`. Missing or ambiguous devices cause an error;
the test never silently falls back to another microphone.

Both the media test and live conversation use this input selection. Camera,
motion, and speaker output still use Reachy. Capture uses 16 kHz and channel 0,
the processed speech channel on both the one-channel and six-channel firmware.
Use `--input-channel N` to select another zero-based channel if needed. The saved
WAV is mono and `audio_metrics.json` identifies the input device and channel.

The [reSpeaker USB Mic Array](https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/)
also supports playback through its 3.5 mm jack with headphones or powered
speakers attached; it has no built-in speaker. This test uses Reachy's speaker.
The array's playback-reference echo cancellation is not fed by this separate
speaker route. To prevent Reachy hearing and interrupting itself, the adapter
continues draining the mic but sends silence during queued speaker playback and
for 0.5 seconds afterward. Wait until Reachy finishes speaking before replying;
voice interruption during its speech is disabled with the external microphone.

Results go to timestamped `artifacts/` folders. Ctrl+C stops the conversation
app. The app uses its default hosted Hugging Face realtime backend (internet
required); optional local backend configuration can go in `.env`, using the
upstream `.env.example` as a reference. Camera capture alone does not analyze an
image: analysis occurs in the real conversation backend. Allow camera/microphone
access in macOS if prompted. The script does not start the daemon automatically.
App data is isolated under `.test_data` via `XDG_DATA_HOME`.

If microphone capture returns only zero samples, the media test saves the WAV
and metrics, then stops with an error. On macOS, check **System Settings >
Privacy & Security > Microphone** for the terminal app running Python. Restart
the daemon and test after changing access. Check the selected microphone in
**Sound > Input** while speaking: if its input meter also stays flat, investigate input
volume/mute and the robot's microphone/USB connection before the conversation app.

To recreate the environment with the same dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
```

Keep the accompanying `vendor/conversation_app` checkout: the lock references
it as an editable install. `source_versions.json` records its Git revision.
