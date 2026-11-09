"""Read-only XVF3000 direction sensing for the reSpeaker USB Mic Array."""

import math
import struct

from libusb_package import get_libusb1_backend
import usb.core
import usb.util

def signed_angle(degrees: float) -> float:
    return (degrees + 180) % 360 - 180


class SoundDirection:
    """Access the external XVF3000, never Reachy's internal XVF3800."""

    def __init__(self) -> None:
        self.device = None
        devices = list(usb.core.find(find_all=True, idVendor=0x2886, idProduct=0x0018,
                                     backend=get_libusb1_backend()) or [])
        if len(devices) != 1:
            raise RuntimeError(f"Expected one external XVF3000 USB array; found {len(devices)}")
        self.device = devices[0]
        try:
            self.read_raw()
        except Exception:
            self.close()
            raise

    def read_parameter(self, resource: int, offset: int) -> int:
        # See Seeed's usb_4_mic_array/tuning.py: vendor integer read, 8-byte reply.
        response = self.device.ctrl_transfer(0xC0, 0, 0xC0 | offset, resource, 8, timeout=150)
        if len(response) != 8:
            raise ValueError("Unexpected reSpeaker control response length")
        return struct.unpack("<ii", bytes(response))[0]

    def read_raw(self) -> tuple[float, bool]:
        voice = bool(self.read_parameter(19, 32))
        angle = self.read_parameter(21, 0)
        if not 0 <= angle < 360:
            raise ValueError(f"Invalid reSpeaker direction: {angle}")
        return float(angle), voice

    def read(self) -> tuple[float, bool]:
        angle, voice = self.read_raw()
        # User's mounting convention: 0 degrees is front, +30 is 30 degrees left.
        return math.radians(signed_angle(angle)), voice

    def close(self) -> None:
        if self.device is not None:
            usb.util.dispose_resources(self.device)
            self.device = None
