"""External USB microphone input with Reachy's existing camera and speaker."""

import logging
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def input_devices() -> list[tuple[int, dict]]:
    """List available microphone devices."""
    return [(i, device) for i, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0]


def select_input(selector: str) -> tuple[int, dict]:
    """Resolve a microphone by index or unique name fragment."""
    devices = input_devices()
    matches = [(i, device) for i, device in devices
               if (i == int(selector) if selector.isdecimal()
                   else selector.casefold() in device["name"].casefold())]
    if len(matches) != 1:
        listing = ", ".join(f'{i}: {device["name"]}' for i, device in devices)
        raise ValueError(
            f"Microphone {selector!r} matched {len(matches)} inputs. "
            f"Choose --input-device from: {listing or 'no inputs detected'}"
        )
    return matches[0]


class ExternalMicrophoneMedia:
    """Delegate camera/output to Reachy while capturing an external microphone."""

    def __init__(self, media, selector: str, channel: int = 0) -> None:
        self.media = media
        self.device_index, self.device = select_input(selector)
        self.channels = int(self.device["max_input_channels"])
        if not 0 <= channel < self.channels:
            raise ValueError(f"Input channel must be between 0 and {self.channels - 1}")
        self.channel = channel
        self.stream = None
        self._last_overflow_warning = float("-inf")
        self._playback_until = 0.0
        sd.check_input_settings(device=self.device_index, samplerate=16000,
                                channels=self.channels, dtype="float32")
        print(f'Audio input: {self.device["name"]}, channel {channel}, 16000 Hz; '
              "output: Reachy speaker", flush=True)

    def __getattr__(self, name):
        return getattr(self.media, name)

    def start_recording(self) -> None:
        if self.stream is not None:
            return
        stream = sd.InputStream(device=self.device_index, samplerate=16000,
                                channels=self.channels, dtype="float32")
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self.stream = stream
        self._last_overflow_warning = float("-inf")

    def stop_recording(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def get_audio_sample(self) -> np.ndarray | None:
        if self.stream is None:
            return None
        available = self.stream.read_available
        if not available:
            return None
        samples, overflow = self.stream.read(min(available, 1600))
        # PortAudio already discarded the lost input; the returned samples remain usable.
        if overflow and time.monotonic() - self._last_overflow_warning >= 10:
            logger.warning("External microphone input overflow: some audio was dropped; "
                           "continuing capture. Repeated drops can interrupt speech recognition.")
            self._last_overflow_warning = time.monotonic()
        # Reachy's speaker does not feed the external array's echo-reference channel.
        # Send silence while output is queued, preserving VAD timing without self-interruption.
        if time.monotonic() < self._playback_until + 0.5:
            return np.zeros((len(samples), 1), dtype=np.float32)
        # Channel 0 is processed speech; other channels include raw mics/loopback.
        return samples[:, self.channel:self.channel + 1].copy()

    def push_audio_sample(self, samples: np.ndarray) -> None:
        self.media.push_audio_sample(samples)
        rate = self.media.get_output_audio_samplerate()
        self._playback_until = max(time.monotonic(), self._playback_until) + len(samples) / rate

    def stop_playing(self) -> None:
        self.media.stop_playing()
        self._playback_until = time.monotonic()

    def reset_echo_gate(self) -> None:
        """Forget queued duration after the SDK player has been flushed."""
        self._playback_until = time.monotonic()

    def get_input_audio_samplerate(self) -> int:
        return 16000

    def get_input_channels(self) -> int:
        return 1

    def get_DoA(self):
        # Reachy's built-in direction sensor does not describe this microphone.
        return None

    def close(self) -> None:
        try:
            self.stop_recording()
        finally:
            self.media.close()
