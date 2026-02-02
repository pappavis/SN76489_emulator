## `sn76489_emulator.py` — v0.03 (volledige een-lêer implementasie)

#!/usr/bin/env python3
# SN76489 Emulator v0.03 — 2026-02-02
# Multi-chip, stereo routing, mixer & gain model
# Python 3.12 / MacOS

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


# =========================
# Config
# =========================

@dataclass
class AppConfig:
    sample_rate: int = 44100
    master_clock_hz: int = 3_579_545
    block_frames: int = 512
    master_gain: float = 0.25
    chip_gain: float = 1.0
    pan: str = "both"  # left | right | both
    debug: bool = False


# =========================
# SN76489 Core (mono)
# =========================

class ToneChannel:
    def __init__(self, sr, clk):
        self.sr = sr
        self.clk = clk
        self.period = 0
        self.volume = 15
        self.phase = 0.0

    def freq(self):
        if self.period <= 0:
            return 0.0
        return self.clk / (32.0 * self.period)

    def render(self, n, amp):
        if self.volume >= 15 or self.period <= 0:
            return np.zeros(n, np.float32)
        f = self.freq()
        inc = f / self.sr
        out = np.empty(n, np.float32)
        ph = self.phase
        a = amp[self.volume]
        for i in range(n):
            out[i] = (1.0 if ph < 0.5 else -1.0) * a
            ph += inc
            if ph >= 1.0:
                ph -= 1.0
        self.phase = ph
        return out


class NoiseChannel:
    def __init__(self, sr, clk):
        self.sr = sr
        self.clk = clk
        self.ctrl = 0
        self.volume = 15
        self.lfsr = 0x4000
        self.phase = 0.0
        self.tone2_period = lambda: 0

    def render(self, n, amp):
        if self.volume >= 15:
            return np.zeros(n, np.float32)

        rate = self.ctrl & 0x03
        white = (self.ctrl >> 2) & 1

        if rate == 0:
            f = self.clk / (32 * 16)
        elif rate == 1:
            f = self.clk / (32 * 32)
        elif rate == 2:
            f = self.clk / (32 * 64)
        else:
            p = self.tone2_period()
            f = self.clk / (32 * p) if p > 0 else 0

        if f <= 0:
            return np.zeros(n, np.float32)

        inc = f / self.sr
        out = np.empty(n, np.float32)
        ph = self.phase
        bit = self.lfsr & 1
        a = amp[self.volume]

        for i in range(n):
            ph += inc
            if ph >= 1.0:
                ph -= 1.0
                lsb = self.lfsr & 1
                self.lfsr >>= 1
                fb = (lsb ^ (self.lfsr & 1)) if white else lsb
                self.lfsr |= (fb << 14)
                bit = self.lfsr & 1
            out[i] = (1.0 if bit else -1.0) * a

        self.phase = ph
        return out


class SN76489Core:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.regs = [0] * 8
        self.latched = 0

        self.tone = [
            ToneChannel(cfg.sample_rate, cfg.master_clock_hz),
            ToneChannel(cfg.sample_rate, cfg.master_clock_hz),
            ToneChannel(cfg.sample_rate, cfg.master_clock_hz),
        ]
        self.noise = NoiseChannel(cfg.sample_rate, cfg.master_clock_hz)
        self.noise.tone2_period = lambda: self.tone[2].period

        self.amp = np.array(
            [0.0 if i == 15 else 10 ** (-i / 10) for i in range(16)],
            dtype=np.float32,
        )

        self.writes = 0
        self.renders = 0
        self.frames = 0

    def write(self, b: int):
        self.writes += 1
        if b & 0x80:
            self.latched = (b >> 4) & 7
            val = b & 0x0F
            self._write_nibble(self.latched, val)
        else:
            if self.latched in (0, 2, 4):
                lo = self.regs[self.latched] & 0x0F
                self.regs[self.latched] = ((b & 0x3F) << 4) | lo
                self._apply(self.latched)

    def _write_nibble(self, r, v):
        if r in (0, 2, 4):
            self.regs[r] = (self.regs[r] & 0x3F0) | v
        else:
            self.regs[r] = v
        self._apply(r)

    def _apply(self, r):
        if r == 0:
            self.tone[0].period = self.regs[0]
        elif r == 1:
            self.tone[0].volume = self.regs[1]
        elif r == 2:
            self.tone[1].period = self.regs[2]
        elif r == 3:
            self.tone[1].volume = self.regs[3]
        elif r == 4:
            self.tone[2].period = self.regs[4]
        elif r == 5:
            self.tone[2].volume = self.regs[5]
        elif r == 6:
            self.noise.ctrl = self.regs[6]
        elif r == 7:
            self.noise.volume = self.regs[7]

    def render(self, n):
        self.renders += 1
        self.frames += n
        return (
            self.tone[0].render(n, self.amp)
            + self.tone[1].render(n, self.amp)
            + self.tone[2].render(n, self.amp)
            + self.noise.render(n, self.amp)
        )


# =========================
# ChipBank + Mixer
# =========================

class ChipBank:
    def __init__(self, cfg: AppConfig, chips: int):
        self.cfg = cfg
        self.chips: List[SN76489Core] = [
            SN76489Core(cfg) for _ in range(chips)
        ]

    def render(self, n):
        left = np.zeros(n, np.float32)
        right = np.zeros(n, np.float32)

        pan_l, pan_r = {
            "left": (1.0, 0.0),
            "right": (0.0, 1.0),
            "both": (1.0, 1.0),
        }[self.cfg.pan]

        for chip in self.chips:
            mono = chip.render(n) * self.cfg.chip_gain
            left += mono * pan_l
            right += mono * pan_r

        left *= self.cfg.master_gain
        right *= self.cfg.master_gain
        np.clip(left, -1, 1, out=left)
        np.clip(right, -1, 1, out=right)
        return np.stack([left, right], axis=1)


# =========================
# Audio
# =========================

class AudioOut:
    def __init__(self, cfg: AppConfig):
        if sd is None:
            raise RuntimeError("pip install sounddevice")
        self.cfg = cfg

    def play(self, buf):
        sd.play(buf, self.cfg.sample_rate, blocking=True)
        sd.stop()


# =========================
# Test Harness
# =========================

def tone_period(clk, hz):
    return max(1, min(1023, int(clk / (32 * hz))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["beep", "noise", "sequence"], default="beep")
    p.add_argument("--freq", type=float, default=440)
    p.add_argument("--seconds", type=float, default=1)
    p.add_argument("--chips", type=int, default=1)
    p.add_argument("--pan", choices=["left", "right", "both"], default="both")
    p.add_argument("--master-gain", type=float, default=0.25)
    p.add_argument("--chip-gain", type=float, default=1.0)
    p.add_argument("--block-frames", type=int, default=512)
    p.add_argument("--dump-regs", action="store_true")
    p.add_argument("--counters", action="store_true")
    args = p.parse_args()

    cfg = AppConfig(
        block_frames=args.block_frames,
        master_gain=args.master_gain,
        chip_gain=args.chip_gain,
        pan=args.pan,
    )

    bank = ChipBank(cfg, min(args.chips, 128))
    audio = AudioOut(cfg)

    # Simple beep on chip 0
    chip0 = bank.chips[0]
    pval = tone_period(cfg.master_clock_hz, args.freq)
    chip0.write(0x80 | pval & 0x0F)
    chip0.write(pval >> 4)
    chip0.write(0x90 | 4)

    frames = int(cfg.sample_rate * args.seconds)
    buf = bank.render(frames)
    audio.play(buf)

    if args.counters:
        for i, c in enumerate(bank.chips):
            print(f"Chip {i}: writes={c.writes} renders={c.renders} frames={c.frames}")


if __name__ == "__main__":
    main()
