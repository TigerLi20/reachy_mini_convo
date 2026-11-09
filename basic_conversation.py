#!/usr/bin/env python3
"""Run conversation with attention, person recognition, and scoped relational memory."""

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-device", default="respeaker")
    parser.add_argument("--input-channel", type=int, default=0)
    parser.add_argument("--robot-name")
    parser.add_argument("--no-attention", action="store_true", help="Use the phase 2 conversation behavior")
    parser.add_argument("--no-doa", action="store_true", help="Use face attention without USB sound direction")
    parser.add_argument("--no-recognition", action="store_true", help="Disable persistent person recognition and enrollment")
    parser.add_argument("--no-memory", action="store_true", help="Use phase 4 without relational conversation memory")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        # Environment is set by the parent before any app modules are imported.
        from reachy_mini import ReachyMini
        from reachy_mini_conversation_app.main import run
        from reachy_mini_conversation_app.utils import parse_args
        from external_microphone import ExternalMicrophoneMedia
        from conversation_runtime import ConversationTiming, TimedMedia
        from attention import AttentionController, AttentiveRobot
        from sound_direction import SoundDirection
        from person_recognition import IdentityStore, PersonRecognition

        sys.argv = [sys.argv[0], "--debug"]
        app_args, _ = parse_args()
        app_args.robot_name = args.robot_name
        timing = ConversationTiming(args.output / "turn_timings.jsonl")
        realtime_logger = logging.getLogger("reachy_mini_conversation_app.huggingface_realtime")
        realtime_logger.addHandler(timing)
        connection = {"robot_name": args.robot_name} if args.robot_name else {}
        attention = None
        sound = None
        recognition = None
        vision = None
        relational = None
        memory_worker = None
        memory_store = None
        if not args.no_memory:
            from memory_models import LocalMemoryModels, MemoryWorker
            from relational_memory import RelationalMemory
            from relational_store import MemoryStore
            models = LocalMemoryModels()
            memory_store = MemoryStore(ROOT / ".relational_memory" / "memory.sqlite3")
            memory_worker = MemoryWorker(memory_store, models)
            for person_id, session_id in memory_store.recoverable_sessions():
                memory_worker.submit(person_id, session_id, final=True)
        if not args.no_recognition and not args.no_attention:
            # A corrupt identity store fails closed; never silently replace enrollments.
            store = IdentityStore(ROOT / ".person_data" / "identities.json")
            try:
                from face_embeddings import FaceEmbeddings
                vision = FaceEmbeddings(ROOT / ".face_models")
            except Exception:
                logging.getLogger(__name__).exception("Recognition model unavailable; identities remain unknown")
            else:
                recognition = PersonRecognition(store, args.output / "recognition.jsonl")
        try:
            with ReachyMini(**connection) as robot:
                microphone = ExternalMicrophoneMedia(robot.media, args.input_device, args.input_channel)
                robot.media_manager = TimedMedia(microphone, timing, recognition)
                if memory_store is not None:
                    relational = RelationalMemory(memory_store, recognition, models, memory_worker)
                    robot.media.relational = relational
                active_robot = robot
                try:
                    if not args.no_attention:
                        if not args.no_doa:
                            try:
                                sound = SoundDirection()
                            except (RuntimeError, ValueError, OSError) as exc:
                                logging.getLogger(__name__).warning(
                                    "Sound direction unavailable: %s. Continuing with face attention.", exc)
                                if sound is not None:
                                    sound.close()
                                    sound = None
                        attention = AttentionController(robot, timing, args.output / "attention.jsonl", sound,
                                                        recognition, vision)
                        attention.start()
                        active_robot = AttentiveRobot(robot, attention)
                    if relational is not None:
                        from memory_runner import run_memory_conversation
                        run_memory_conversation(active_robot, relational, os.environ["PHASE2_INSTANCE_PATH"])
                    else:
                        run(app_args, robot=active_robot, instance_path=os.environ["PHASE2_INSTANCE_PATH"])
                finally:
                    if attention is not None:
                        attention.stop()
                    if sound is not None:
                        sound.close()
        finally:
            realtime_logger.removeHandler(timing)
            timing.close()
            if memory_worker is not None and memory_worker.close():
                memory_store.close()
        return 0

    output = ROOT / "artifacts" / ("conversation-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True)
    print(f"Conversation/attention logs: {output}\nKeep the Reachy daemon running. Both USB devices must be connected.\n"
          "Wait until Reachy finishes speaking before replying.\n"
          "Say 'I should get going' to end with a farewell and sleep; Ctrl+C stops manually.", flush=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--output", str(output),
               "--input-device", args.input_device, "--input-channel", str(args.input_channel)]
    if args.robot_name:
        command.extend(["--robot-name", args.robot_name])
    for flag in ["no_attention", "no_doa", "no_recognition", "no_memory"]:
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    with tempfile.TemporaryDirectory(prefix="reachy-phase2-") as session:
        env = os.environ.copy()
        env.update({"REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY": str(ROOT / "profiles"),
                    "REACHY_MINI_CUSTOM_PROFILE": "basic_conversation" if args.no_memory else "relational_conversation",
                    "REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY": str(ROOT / "conversation_tools"),
                    "AUTOLOAD_EXTERNAL_TOOLS": "false", "XDG_DATA_HOME": session,
                    "PHASE2_INSTANCE_PATH": str(Path(session) / "instance"),
                    "REACHY_MINI_APP_TIMEOUT_MINUTES": "0"})
        Path(env["PHASE2_INSTANCE_PATH"]).mkdir()
        with (output / "conversation.log").open("w") as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    return process.wait()
                except KeyboardInterrupt:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    print(f"\nStopped. Logs: {output}")
                    return 130


if __name__ == "__main__":
    raise SystemExit(main())
