#!/usr/bin/env python3
# SN76489 Emulator (Python) - v0.01 2026-02-02
# Target: MacOS, audio out via sounddevice
#
# Install:
#   pip install numpy sounddevice
#
# Run:
#   python sn76489_emulator.py --test beep

from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


@dataclass
class AppConfig:
    SAMPLE_RATE: int = 44100
    MASTER_CLOCK_HZ: int = 3_579_545  # common NTSC-ish clock used by many SN76489 designs
    BLOCK_FRAMES: int = 512
    MASTER_GAIN: float = 0.25  # conservative to avoid clipping
    DEBUG: bool = False


class ToneChannel:
    def __init__(self, sample_rate: int, master_clock_hz: int):
        self.sample_rate = float(sample_rate)
        self.master_clock_hz = float(master_clock_hz)
        self.period_10bit: int = 0  # 10-bit
        self.volume_4bit: int = 15  # 0 loudest, 15 mute
        self.phase: float = 0.0

    def set_period(self, period_10bit: int) -> None:
        self.period_10bit = max(0, min(1023, int(period_10bit)))

    def set_volume(self, vol_4bit: int) -> None:
        self.volume_4bit = max(0, min(15, int(vol_4bit)))

    def _freq_hz(self) -> float:
        # f = clock / (32 * period)
        p = self.period_10bit
        if p <= 0:
            return 0.0
        return self.master_clock_hz / (32.0 * float(p))

    def render(self, n: int, amp_table: np.ndarray) -> np.ndarray:
        amp = float(amp_table[self.volume_4bit])
        f = self._freq_hz()
        if amp <= 0.0 or f <= 0.0:
            return np.zeros(n, dtype=np.float32)

        # phase increment per sample
        inc = f / self.sample_rate
        out = np.empty(n, dtype=np.float32)

        ph = self.phase
        for i in range(n):
            out[i] = (1.0 if ph < 0.5 else -1.0) * amp
            ph += inc
            if ph >= 1.0:
                ph -= 1.0

        self.phase = ph
        return out


class NoiseChannel:
    def __init__(self, sample_rate: int, master_clock_hz: int):
        self.sample_rate = float(sample_rate)
        self.master_clock_hz = float(master_clock_hz)
        self.volume_4bit: int = 15
        self.noise_ctrl_4bit: int = 0  # bits: [mode][rate1][rate0] in low bits
        self.lfsr: int = 0x4000  # 15-bit seed (bit14 set)
        self.phase: float = 0.0

        # tone2 coupling
        self._tone2_period_provider = lambda: 0

    def bind_tone2_period(self, fn):
        self._tone2_period_provider = fn

    def set_volume(self, vol_4bit: int) -> None:
        self.volume_4bit = max(0, min(15, int(vol_4bit)))

    def set_noise_ctrl(self, ctrl_4bit: int) -> None:
        self.noise_ctrl_4bit = int(ctrl_4bit) & 0x0F

    def _noise_freq_hz(self) -> float:
        rate = self.noise_ctrl_4bit & 0x03
        if rate == 0:
            return self.master_clock_hz / (32.0 * 16.0)
        if rate == 1:
            return self.master_clock_hz / (32.0 * 32.0)
        if rate == 2:
            return self.master_clock_hz / (32.0 * 64.0)
        # rate == 3: tone2 frequency
        p = int(self._tone2_period_provider())
        if p <= 0:
            return 0.0
        return self.master_clock_hz / (32.0 * float(p))

    def _lfsr_step(self) -> int:
        # bit2: mode (0 periodic, 1 white)
        white = (self.noise_ctrl_4bit >> 2) & 0x01

        lsb = self.lfsr & 0x01
        self.lfsr >>= 1

        if white:
            # Common simple tap: XOR bit0 and bit1 into new bit at bit14
            # (Not perfect spec-accurate for every variant, but good audible behavior.)
            newbit = (lsb ^ (self.lfsr & 0x01)) & 0x01
        else:
            # periodic: feedback = previous lsb
            newbit = lsb

        self.lfsr |= (newbit << 14)
        return self.lfsr & 0x01  # output bit

    def render(self, n: int, amp_table: np.ndarray) -> np.ndarray:
        amp = float(amp_table[self.volume_4bit])
        f = self._noise_freq_hz()
        if amp <= 0.0 or f <= 0.0:
            return np.zeros(n, dtype=np.float32)

        inc = f / self.sample_rate
        out = np.empty(n, dtype=np.float32)

        ph = self.phase
        # keep last noise bit to hold value between ticks
        bit = self.lfsr & 0x01

        for i in range(n):
            ph += inc
            if ph >= 1.0:
                ph -= 1.0
                bit = self._lfsr_step()

            out[i] = (1.0 if bit else -1.0) * amp

        self.phase = ph
        return out


class SN76489Core:
    """
    Minimal but complete-ish SN76489-style PSG emulator:
    - 3 tone channels (0..2)
    - 1 noise channel
    - Latch/Data write scheme
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.sample_rate = cfg.SAMPLE_RATE
        self.master_clock_hz = cfg.MASTER_CLOCK_HZ

        self.tone: List[ToneChannel] = [
            ToneChannel(cfg.SAMPLE_RATE, cfg.MASTER_CLOCK_HZ),
            ToneChannel(cfg.SAMPLE_RATE, cfg.MASTER_CLOCK_HZ),
            ToneChannel(cfg.SAMPLE_RATE, cfg.MASTER_CLOCK_HZ),
        ]
        self.noise = NoiseChannel(cfg.SAMPLE_RATE, cfg.MASTER_CLOCK_HZ)
        self.noise.bind_tone2_period(lambda: self.tone[2].period_10bit)

        # Registers (store raw)
        self.regs = [0] * 8

        # latch state
        self._latched_reg: int = 0

        # 16-entry amplitude table (0 loudest, 15 mute)
        # Approximate logarithmic curve (simple and audible)
        self.amp_table = np.zeros(16, dtype=np.float32)
        for v in range(16):
            if v >= 15:
                self.amp_table[v] = 0.0
            else:
                # 2 dB-ish per step feel; tweakable
                self.amp_table[v] = float(10.0 ** (-(v) / 10.0))

        # apply initial register values
        self._apply_all()

    def _dbg(self, msg: str) -> None:
        if self.cfg.DEBUG:
            print(msg)

    def write(self, b: int) -> None:
        b &= 0xFF
        if b & 0x80:
            # latch/data
            reg = (b >> 4) & 0x07
            data = b & 0x0F
            self._latched_reg = reg
            self._write_reg_nibble(reg, data, latched=True)
        else:
            # data byte (only meaningful for tone period regs, adds high 6 bits)
            reg = self._latched_reg
            data6 = b & 0x3F
            self._write_reg_data6(reg, data6)

    def _write_reg_nibble(self, reg: int, nibble: int, latched: bool) -> None:
        self._dbg(f"[SN76489] LATCH reg={reg} nibble=0x{nibble:X}")
        if reg in (0, 2, 4):
            # tone period low 4 bits
            prev = self.regs[reg] & 0x3F0  # keep high bits (bits 4..9) if any
            self.regs[reg] = prev | (nibble & 0x0F)
        elif reg in (1, 3, 5, 7):
            # volume nibble (4-bit)
            self.regs[reg] = nibble & 0x0F
        elif reg == 6:
            # noise control nibble
            self.regs[reg] = nibble & 0x0F
        else:
            self.regs[reg] = nibble & 0x0F

        self._apply_reg(reg)

    def _write_reg_data6(self, reg: int, data6: int) -> None:
        self._dbg(f"[SN76489] DATA reg={reg} data6=0x{data6:X}")
        if reg in (0, 2, 4):
            # tone period high 6 bits go into bits 4..9
            low = self.regs[reg] & 0x0F
            self.regs[reg] = ((data6 & 0x3F) << 4) | low
            self._apply_reg(reg)
        else:
            # ignore for non-tone regs
            pass

    def _apply_all(self) -> None:
        for r in range(8):
            self._apply_reg(r)

    def _apply_reg(self, reg: int) -> None:
        if reg == 0:
            self.tone[0].set_period(self.regs[0] & 0x3FF)
        elif reg == 1:
            self.tone[0].set_volume(self.regs[1] & 0x0F)
        elif reg == 2:
            self.tone[1].set_period(self.regs[2] & 0x3FF)
        elif reg == 3:
            self.tone[1].set_volume(self.regs[3] & 0x0F)
        elif reg == 4:
            self.tone[2].set_period(self.regs[4] & 0x3FF)
        elif reg == 5:
            self.tone[2].set_volume(self.regs[5] & 0x0F)
        elif reg == 6:
            self.noise.set_noise_ctrl(self.regs[6] & 0x0F)
        elif reg == 7:
            self.noise.set_volume(self.regs[7] & 0x0F)

    def render(self, num_frames: int) -> np.ndarray:
        # render each channel, sum, apply master gain, clip
        t0 = self.tone[0].render(num_frames, self.amp_table)
        t1 = self.tone[1].render(num_frames, self.amp_table)
        t2 = self.tone[2].render(num_frames, self.amp_table)
        nz = self.noise.render(num_frames, self.amp_table)

        mix = (t0 + t1 + t2 + nz) * float(self.cfg.MASTER_GAIN)
        np.clip(mix, -1.0, 1.0, out=mix)
        return mix.astype(np.float32)


class AudioOutSoundDevice:
    def __init__(self, sample_rate: int, block_frames: int):
        if sd is None:
            raise RuntimeError(
                "sounddevice ontbreek. Installeer met: pip install sounddevice"
            )
        self.sample_rate = sample_rate
        self.block_frames = block_frames

    def play_mono_buffer(self, buf: np.ndarray) -> None:
        # sounddevice accepts float32 in [-1,1]
        sd.play(buf, samplerate=self.sample_rate, blocking=True)
        sd.stop()

    def stream_from_renderer(self, renderer_fn, seconds: float) -> None:
        total_frames = int(self.sample_rate * seconds)
        frames_left = total_frames
        chunks = []
        while frames_left > 0:
            n = min(self.block_frames, frames_left)
            chunks.append(renderer_fn(n))
            frames_left -= n
        buf = np.concatenate(chunks).astype(np.float32)
        self.play_mono_buffer(buf)


class TestHarness:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.chip = SN76489Core(cfg)
        self.audio = AudioOutSoundDevice(cfg.SAMPLE_RATE, cfg.BLOCK_FRAMES)

    @staticmethod
    def _tone_period_from_freq(master_clock_hz: int, freq_hz: float) -> int:
        # f = clock/(32*period) => period = clock/(32*f)
        if freq_hz <= 0:
            return 0
        p = int(round(master_clock_hz / (32.0 * freq_hz)))
        return max(1, min(1023, p))

    def beep_tone_ch0(self, freq_hz: float = 440.0, seconds: float = 1.0) -> None:
        # Program channel 0 tone period + volume via SN76489-style writes
        period = self._tone_period_from_freq(self.cfg.MASTER_CLOCK_HZ, freq_hz)

        # Write tone0 period (R0) using latch (low nibble) + data (high 6 bits)
        low_n = period & 0x0F
        high_6 = (period >> 4) & 0x3F

        # Latch to reg 0 with low nibble
        # 1 rrr dddd : rrr=000 (reg0), dddd=low
        self.chip.write(0x80 | (0 << 4) | low_n)
        # Data byte with high 6 bits
        self.chip.write(high_6)

        # Set tone0 volume (R1), choose loud-ish but not max harsh
        vol = 4  # 0 loudest, 15 mute
        self.chip.write(0x80 | (1 << 4) | (vol & 0x0F))

        # Mute other channels to be safe
        self.chip.write(0x80 | (3 << 4) | 0x0F)
        self.chip.write(0x80 | (5 << 4) | 0x0F)
        self.chip.write(0x80 | (7 << 4) | 0x0F)

        # Stream for duration
        self.audio.stream_from_renderer(self.chip.render, seconds)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["beep", "none"], default="beep")
    p.add_argument("--freq", type=float, default=440.0)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    cfg = AppConfig(DEBUG=bool(args.debug))

    if args.test == "beep":
        th = TestHarness(cfg)
        th.beep_tone_ch0(freq_hz=args.freq, seconds=args.seconds)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
