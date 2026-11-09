#!/usr/bin/env python3
"""Check robot media, then run the official app with a test personality."""

import argparse
import asyncio
import base64
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
import webbrowser
from datetime import datetime

ROOT = Path(__file__).resolve().parent


def configure_input(robot, args) -> None:
    if args.audio_source == "respeaker":
        from external_microphone import ExternalMicrophoneMedia
        robot.media_manager = ExternalMicrophoneMedia(
            robot.media, args.input_device, args.input_channel
        )


def preflight(seconds: float, output: Path, args) -> None:
    """Capture a camera image and record/replay the selected microphone."""
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.tools.camera import Camera
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

    connection = {"timeout": 10}
    if args.robot_name:
        connection["robot_name"] = args.robot_name
    with ReachyMini(**connection) as robot:
        configure_input(robot, args)
        deps = ToolDependencies(robot, movement_manager=None, camera_enabled=True)
        deadline = time.monotonic() + 10
        while True:
            result = asyncio.run(Camera()(deps, question="Describe what you see."))
            if "b64_im" in result:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Camera tool failed: {result}")
            time.sleep(0.2)
        image = output / "camera.jpg"
        image.write_bytes(base64.b64decode(result["b64_im"]))
        webbrowser.open(image.as_uri())
        print(f"Camera tool captured {image}. Analysis follows in the conversation app.")

        input(f"Press Enter, then speak for {seconds:g} seconds to record your voice: ")
        chunks = []
        robot.media.start_recording()
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                samples = robot.media.get_audio_sample()
                if samples is not None and samples.size:
                    chunks.append(samples.copy())
                else:
                    time.sleep(0.01)
        finally:
            robot.media.stop_recording()
        if not chunks:
            raise RuntimeError("Microphone returned no audio samples.")
        audio = np.concatenate(chunks, axis=0)
        if audio.ndim == 1:
            audio = audio[:, None]
        rate = robot.media.get_input_audio_samplerate()
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio)))
        with wave.open(str(output / "microphone.wav"), "wb") as recording:
            recording.setnchannels(audio.shape[1])
            recording.setsampwidth(2)
            recording.setframerate(rate)
            recording.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        (output / "audio_metrics.json").write_text(
            json.dumps({"sample_rate": rate, "samples": len(audio), "rms": rms, "peak": peak,
                        "audio_source": args.audio_source,
                        "input_device": (robot.media.device["name"]
                                         if args.audio_source == "respeaker" else "Reachy Mini Audio"),
                        "input_channel": args.input_channel if args.audio_source == "respeaker" else None}, indent=2)
        )
        print(f"Recorded {len(audio) / rate:.2f}s; RMS={rms:.5f}, peak={peak:.5f}.")
        if peak == 0:
            raise RuntimeError(
                "Microphone returned only zero samples (digital silence). "
                "The WAV and audio metrics were saved, but there is no sound to replay. "
                "On macOS, check System Settings > Privacy & Security > Microphone "
                "for the terminal app running Python, then restart the daemon and test. "
                "Also check the selected microphone in Sound > Input for a moving input meter."
            )
        if peak < 0.001:
            print("Audio is nearly silent: check microphone selection/permissions.")
        input("Press Enter to replay the recording through Reachy's speaker: ")
        output_rate = robot.media.get_output_audio_samplerate()
        divisor = gcd(rate, output_rate)
        playback = resample_poly(audio.mean(axis=1), output_rate // divisor, rate // divisor)
        playback = playback.astype(np.float32)[:, None]
        robot.media.start_playing()
        try:
            robot.media.push_audio_sample(playback)
            time.sleep(len(playback) / output_rate + 0.5)
        finally:
            robot.media.stop_playing()
        print("Playback completed. Confirm by ear that you heard your recording.")


def main() -> int:
    """Run the selected preliminary functionality checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "media", "conversation", "check"], default="all")
    parser.add_argument("--seconds", type=float, default=5, help="Microphone recording duration (1–30 seconds)")
    parser.add_argument("--robot-name")
    parser.add_argument("--audio-source", choices=["respeaker", "reachy"], default="respeaker")
    parser.add_argument("--input-device", default="respeaker", help="USB input name fragment or device index")
    parser.add_argument("--input-channel", type=int, default=0, help="Zero-based USB input channel (0: processed voice)")
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument("--conversation-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list_audio_devices:
        from external_microphone import input_devices
        for index, device in input_devices():
            print(f'{index}: {device["name"]} ({device["max_input_channels"]} input channels)')
        return 0
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    if args.mode == "check":
        for name in ["reachy_mini", "reachy_mini_conversation_app.main",
                     "reachy_mini_conversation_app.tools.camera", "numpy", "scipy.signal", "sounddevice"]:
            importlib.import_module(name)
            print(f"OK: {name}")
        print("Environment ready. Hardware and backend have not been tested.")
        return 0

    if args.conversation_child:
        from reachy_mini import ReachyMini
        from reachy_mini_conversation_app.main import run
        from reachy_mini_conversation_app.utils import parse_args
        app_argv = ["--ui", "--debug"]
        if args.robot_name:
            app_argv.extend(["--robot-name", args.robot_name])
        sys.argv = [sys.argv[0], *app_argv]
        app_args, _ = parse_args()
        connection = {"robot_name": args.robot_name} if args.robot_name else {}
        with ReachyMini(**connection) as robot:
            configure_input(robot, args)
            run(app_args, robot=robot)
        return 0

    output = ROOT / "artifacts" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    print(f"Results: {output}\nStart the Lite daemon first and stop any other conversation app.")
    if args.mode in {"all", "media"}:
        try:
            preflight(args.seconds, output, args)
        except Exception as exc:
            (output / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            print(f"Media check failed: {exc}\nCheck the daemon, USB connection and camera/microphone permissions.", file=sys.stderr)
            return 1
    if args.mode in {"all", "conversation"}:
        print("\nConversation checks (spoken prompts):\n"
              "  1. 'What am I holding?' — verify a fresh camera call and accurate spoken description.\n"
              "  2. 'Repeat exactly: blue robot seven.' — verify hearing and spoken playback.\n"
              "  3. 'Move your head left, then back to front.' — observe actual movement.\n"
              "  4. Optional: 'Do a dance.'\n"
              "Open http://127.0.0.1:7860 for the app UI. Press Ctrl+C to stop.\n"
              "Tool logs prove calls; visual/audio accuracy must be checked by you.")
        env = os.environ.copy()
        env["REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY"] = str(ROOT / "test_profiles")
        env["REACHY_MINI_CUSTOM_PROFILE"] = "functionality_test"
        env["XDG_DATA_HOME"] = str(ROOT / ".test_data")
        command = [sys.executable, str(ROOT / "functionality_test.py"), "--conversation-child",
                   "--audio-source", args.audio_source, "--input-device", args.input_device,
                   "--input-channel", str(args.input_channel)]
        if args.robot_name:
            command.extend(["--robot-name", args.robot_name])
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
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    print(f"\nStopped. Log saved in {output}.")
                    return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
