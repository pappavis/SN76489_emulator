#!/usr/bin/env python3
# SN76489 Emulator (Python) - v0.02 2026-02-02
# Target: MacOS, audio out via sounddevice
#
# Install:
#   pip install numpy sounddevice
#
# Examples:
#   python sn76489_emulator.py --test beep --seconds 1 --freq 440 --dump-regs --counters
#   python sn76489_emulator.py --test noise --seconds 1 --noise-mode white --dump-regs --counters
#   python sn76489_emulator.py --test noise --seconds 1 --noise-mode periodic --dump-regs --counters
#   python sn76489_emulator.py --test sequence --sequence preset1 --bpm 120 --dump-regs --counters

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


@dataclass
class AppConfig:
    SAMPLE_RATE: int = 44100
    MASTER_CLOCK_HZ: int = 3_579_545
    BLOCK_FRAMES: int = 512
    MASTER_GAIN: float = 0.25
    DEBUG: bool = False
    DEBUG_RATE_LIMIT_SEC: float = 1.0  # print once per N seconds worth of frames


class ToneChannel:
    def __init__(self, sample_rate: int, master_clock_hz: int):
        self.sample_rate = float(sample_rate)
        self.master_clock_hz = float(master_clock_hz)
        self.period_10bit: int = 0
        self.volume_4bit: int = 15
        self.phase: float = 0.0

    def set_period(self, period_10bit: int) -> None:
        self.period_10bit = max(0, min(1023, int(period_10bit)))

    def set_volume(self, vol_4bit: int) -> None:
        self.volume_4bit = max(0, min(15, int(vol_4bit)))

    def freq_hz(self) -> float:
        p = self.period_10bit
        if p <= 0:
            return 0.0
        return self.master_clock_hz / (32.0 * float(p))

    def render(self, n: int, amp_table: np.ndarray) -> np.ndarray:
        amp = float(amp_table[self.volume_4bit])
        f = self.freq_hz()
        if amp <= 0.0 or f <= 0.0:
            return np.zeros(n, dtype=np.float32)

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
        self.noise_ctrl_4bit: int = 0
        self.lfsr: int = 0x4000  # bit14 set
        self.phase: float = 0.0
        self._tone2_period_provider: Callable[[], int] = lambda: 0

    def bind_tone2_period(self, fn: Callable[[], int]) -> None:
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
        # rate == 3: tone2-linked
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
            # simple, stable, plausible: xor of bit0 and bit1 into new bit14
            newbit = (lsb ^ (self.lfsr & 0x01)) & 0x01
        else:
            # periodic: feedback = previous lsb
            newbit = lsb

        self.lfsr |= (newbit << 14)
        return self.lfsr & 0x01

    def render(self, n: int, amp_table: np.ndarray) -> np.ndarray:
        amp = float(amp_table[self.volume_4bit])
        f = self._noise_freq_hz()
        if amp <= 0.0 or f <= 0.0:
            return np.zeros(n, dtype=np.float32)

        inc = f / self.sample_rate
        out = np.empty(n, dtype=np.float32)

        ph = self.phase
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
    SN76489-style PSG emulator:
    - 3 tone channels
    - 1 noise channel
    - latch/data write scheme
    - debug counters + register dump
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.sample_rate = int(cfg.SAMPLE_RATE)
        self.master_clock_hz = int(cfg.MASTER_CLOCK_HZ)

        self.tone: List[ToneChannel] = [
            ToneChannel(self.sample_rate, self.master_clock_hz),
            ToneChannel(self.sample_rate, self.master_clock_hz),
            ToneChannel(self.sample_rate, self.master_clock_hz),
        ]
        self.noise = NoiseChannel(self.sample_rate, self.master_clock_hz)
        self.noise.bind_tone2_period(lambda: self.tone[2].period_10bit)

        # raw registers
        self.regs = [0] * 8
        self._latched_reg: int = 0

        # counters
        self.writes_total: int = 0
        self.writes_latch: int = 0
        self.writes_data: int = 0
        self.render_calls_total: int = 0
        self.frames_rendered_total: int = 0

        # debug rate limit
        self._debug_frames_accum: int = 0

        # amplitude table (0 loudest, 15 mute)
        self.amp_table = np.zeros(16, dtype=np.float32)
        for v in range(16):
            if v >= 15:
                self.amp_table[v] = 0.0
            else:
                # log-ish attenuation curve (tweakable)
                self.amp_table[v] = float(10.0 ** (-(v) / 10.0))

        self._apply_all()

    def _dbg(self, msg: str) -> None:
        if self.cfg.DEBUG:
            print(msg)

    def write(self, b: int) -> None:
        b &= 0xFF
        self.writes_total += 1

        if b & 0x80:
            self.writes_latch += 1
            reg = (b >> 4) & 0x07
            data = b & 0x0F
            self._latched_reg = reg
            if self.cfg.DEBUG:
                self._dbg(f"[WRITE] LATCH reg=R{reg} nibble=0x{data:X}")
            self._write_reg_nibble(reg, data)
        else:
            self.writes_data += 1
            reg = self._latched_reg
            data6 = b & 0x3F
            if self.cfg.DEBUG:
                self._dbg(f"[WRITE] DATA  reg=R{reg} data6=0x{data6:X}")
            self._write_reg_data6(reg, data6)

    def _write_reg_nibble(self, reg: int, nibble: int) -> None:
        if reg in (0, 2, 4):
            prev = self.regs[reg] & 0x3F0  # keep high bits (4..9)
            self.regs[reg] = prev | (nibble & 0x0F)
        elif reg in (1, 3, 5, 7):
            self.regs[reg] = nibble & 0x0F
        elif reg == 6:
            self.regs[reg] = nibble & 0x0F
        else:
            self.regs[reg] = nibble & 0x0F

        self._apply_reg(reg)

    def _write_reg_data6(self, reg: int, data6: int) -> None:
        if reg in (0, 2, 4):
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
        self.render_calls_total += 1
        self.frames_rendered_total += int(num_frames)

        t0 = self.tone[0].render(num_frames, self.amp_table)
        t1 = self.tone[1].render(num_frames, self.amp_table)
        t2 = self.tone[2].render(num_frames, self.amp_table)
        nz = self.noise.render(num_frames, self.amp_table)

        mix = (t0 + t1 + t2 + nz) * float(self.cfg.MASTER_GAIN)
        np.clip(mix, -1.0, 1.0, out=mix)

        # rate-limited debug summary
        if self.cfg.DEBUG:
            self._debug_frames_accum += int(num_frames)
            sec_limit_frames = int(self.cfg.DEBUG_RATE_LIMIT_SEC * self.sample_rate)
            if sec_limit_frames > 0 and self._debug_frames_accum >= sec_limit_frames:
                self._debug_frames_accum = 0
                st = self.get_state_summary()
                self._dbg(
                    f"[DBG] renders={self.render_calls_total} frames={self.frames_rendered_total} "
                    f"t0={st['tone'][0]['freq_hz']:.1f}Hz v={st['tone'][0]['vol']} "
                    f"noise={st['noise']['mode']} rate={st['noise']['rate']} v={st['noise']['vol']}"
                )

        return mix.astype(np.float32)

    def get_state_summary(self) -> Dict[str, Any]:
        tone_summary = []
        for i in range(3):
            f = self.tone[i].freq_hz()
            vol = int(self.tone[i].volume_4bit)
            amp = float(self.amp_table[vol])
            tone_summary.append(
                {
                    "period": int(self.tone[i].period_10bit),
                    "freq_hz": float(f),
                    "vol": vol,
                    "amp": amp,
                }
            )

        noise_ctrl = int(self.noise.noise_ctrl_4bit) & 0x0F
        mode = "WHITE" if ((noise_ctrl >> 2) & 0x01) else "PERIODIC"
        rate_sel = noise_ctrl & 0x03
        rate_map = {0: "DIV16", 1: "DIV32", 2: "DIV64", 3: "TONE2"}
        rate = rate_map.get(rate_sel, "UNKNOWN")

        nvol = int(self.noise.volume_4bit)
        namp = float(self.amp_table[nvol])

        return {
            "latched_reg": int(self._latched_reg),
            "tone": tone_summary,
            "noise": {
                "ctrl": noise_ctrl,
                "mode": mode,
                "rate": rate,
                "vol": nvol,
                "amp": namp,
            },
        }

    def dump_regs(self) -> str:
        st = self.get_state_summary()
        lat = st["latched_reg"]

        lines = []
        lines.append("SN76489 REG DUMP")
        lines.append(f"  latched_reg: R{lat}")
        # R0..R7
        # Tone0
        p0 = self.regs[0] & 0x3FF
        v0 = self.regs[1] & 0x0F
        lines.append(
            f"  R0 tone0_period : 0x{p0:03X} ({p0:4d})  f= {st['tone'][0]['freq_hz']:8.1f} Hz"
        )
        lines.append(
            f"  R1 tone0_vol    : 0x{v0:01X} ({v0:2d})    amp={st['tone'][0]['amp']:.3f}"
        )
        # Tone1
        p1 = self.regs[2] & 0x3FF
        v1 = self.regs[3] & 0x0F
        lines.append(
            f"  R2 tone1_period : 0x{p1:03X} ({p1:4d})  f= {st['tone'][1]['freq_hz']:8.1f} Hz"
        )
        lines.append(
            f"  R3 tone1_vol    : 0x{v1:01X} ({v1:2d})    amp={st['tone'][1]['amp']:.3f}"
        )
        # Tone2
        p2 = self.regs[4] & 0x3FF
        v2 = self.regs[5] & 0x0F
        lines.append(
            f"  R4 tone2_period : 0x{p2:03X} ({p2:4d})  f= {st['tone'][2]['freq_hz']:8.1f} Hz"
        )
        lines.append(
            f"  R5 tone2_vol    : 0x{v2:01X} ({v2:2d})    amp={st['tone'][2]['amp']:.3f}"
        )
        # Noise
        nc = self.regs[6] & 0x0F
        nv = self.regs[7] & 0x0F
        lines.append(
            f"  R6 noise_ctrl   : 0x{nc:01X} ({nc:2d})    mode={st['noise']['mode']} rate={st['noise']['rate']}"
        )
        lines.append(
            f"  R7 noise_vol    : 0x{nv:01X} ({nv:2d})    amp={st['noise']['amp']:.3f}"
        )
        return "\n".join(lines)

    def dump_counters(self) -> str:
        sr = float(self.sample_rate)
        seconds = (float(self.frames_rendered_total) / sr) if sr > 0 else 0.0
        lines = []
        lines.append("SN76489 COUNTERS")
        lines.append(f"  writes_total        : {self.writes_total}")
        lines.append(f"  writes_latch        : {self.writes_latch}")
        lines.append(f"  writes_data         : {self.writes_data}")
        lines.append(f"  render_calls_total  : {self.render_calls_total}")
        lines.append(f"  frames_rendered     : {self.frames_rendered_total}")
        lines.append(f"  seconds_rendered    : {seconds:.3f}")
        return "\n".join(lines)


class AudioOutSoundDevice:
    def __init__(self, sample_rate: int, block_frames: int):
        if sd is None:
            raise RuntimeError("sounddevice ontbreek. Installeer met: pip install sounddevice")
        self.sample_rate = int(sample_rate)
        self.block_frames = int(block_frames)

    def play_mono_buffer(self, buf: np.ndarray) -> None:
        sd.play(buf, samplerate=self.sample_rate, blocking=True)
        sd.stop()

    def stream_from_renderer(self, renderer_fn, seconds: float) -> None:
        total_frames = int(self.sample_rate * float(seconds))
        frames_left = total_frames
        chunks = []
        while frames_left > 0:
            n = min(self.block_frames, frames_left)
            chunks.append(renderer_fn(n))
            frames_left -= n
        buf = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
        self.play_mono_buffer(buf)


class TestHarness:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.chip = SN76489Core(cfg)
        self.audio = AudioOutSoundDevice(cfg.SAMPLE_RATE, cfg.BLOCK_FRAMES)

    @staticmethod
    def _tone_period_from_freq(master_clock_hz: int, freq_hz: float) -> int:
        if freq_hz <= 0:
            return 0
        p = int(round(master_clock_hz / (32.0 * float(freq_hz))))
        return max(1, min(1023, p))

    @staticmethod
    def _midi_to_freq(note: int) -> float:
        # A4=69 -> 440 Hz
        return 440.0 * (2.0 ** ((int(note) - 69) / 12.0))

    def _write_tone_period(self, tone_reg_base: int, period_10bit: int) -> None:
        # tone_reg_base: 0 for tone0, 2 for tone1, 4 for tone2
        low_n = int(period_10bit) & 0x0F
        high_6 = (int(period_10bit) >> 4) & 0x3F
        self.chip.write(0x80 | (tone_reg_base << 4) | low_n)
        self.chip.write(high_6)

    def _write_volume(self, vol_reg: int, vol_4bit: int) -> None:
        self.chip.write(0x80 | (vol_reg << 4) | (int(vol_4bit) & 0x0F))

    def _mute_all(self) -> None:
        # volumes: R1,R3,R5,R7
        self._write_volume(1, 0x0F)
        self._write_volume(3, 0x0F)
        self._write_volume(5, 0x0F)
        self._write_volume(7, 0x0F)

    def run_beep(self, freq_hz: float = 440.0, seconds: float = 1.0, vol: int = 4) -> None:
        self._mute_all()

        period = self._tone_period_from_freq(self.cfg.MASTER_CLOCK_HZ, freq_hz)
        self._write_tone_period(0, period)  # tone0
        self._write_volume(1, vol)          # tone0 volume

        self.audio.stream_from_renderer(self.chip.render, seconds)

    def run_noise(self, mode: str = "white", seconds: float = 1.0, vol: int = 4) -> None:
        self._mute_all()

        # Noise ctrl nibble:
        # bits1..0 rate: choose DIV32 (1) as a good default
        # bit2 mode: 1 white, 0 periodic
        rate_sel = 1  # DIV32
        mode_bit = 1 if mode.lower().startswith("w") else 0
        noise_ctrl = (mode_bit << 2) | rate_sel

        self.chip.write(0x80 | (6 << 4) | (noise_ctrl & 0x0F))  # R6
        self._write_volume(7, vol)                               # R7 noise vol

        self.audio.stream_from_renderer(self.chip.render, seconds)

    def run_sequence(self, preset: str = "preset1", bpm: float = 120.0, vol: int = 4) -> None:
        """
        Simple note sequencer: plays a fixed melody on tone0 only.
        - Uses SN76489 register writes for each note period + volume gate.
        """
        self._mute_all()

        # Choose a preset note list (MIDI note numbers)
        presets: Dict[str, List[int]] = {
            "preset1": [60, 64, 67, 72],          # C4 E4 G4 C5
            "preset2": [69, 69, 67, 64, 60],      # A4 A4 G4 E4 C4
            "arp1":    [60, 64, 67, 64] * 2,      # arpeggio loop
        }
        notes = presets.get(preset, presets["preset1"])

        # Timing: quarter-note duration
        bpm = max(30.0, min(300.0, float(bpm)))
        quarter_sec = 60.0 / bpm
        gate = 0.85  # portion of note held before muting (simple "staccato" gate)

        for n in notes:
            f = self._midi_to_freq(n)
            period = self._tone_period_from_freq(self.cfg.MASTER_CLOCK_HZ, f)

            # Note on
            self._write_tone_period(0, period)
            self._write_volume(1, vol)

            # Hold gated portion
            self.audio.stream_from_renderer(self.chip.render, quarter_sec * gate)

            # Note off (mute)
            self._write_volume(1, 0x0F)

            # Short gap
            gap = max(0.0, quarter_sec * (1.0 - gate))
            if gap > 0:
                self.audio.stream_from_renderer(self.chip.render, gap)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()

    p.add_argument("--test", choices=["beep", "noise", "sequence"], default="beep")
    p.add_argument("--freq", type=float, default=440.0, help="Beep frequency (Hz)")
    p.add_argument("--seconds", type=float, default=1.0, help="Duration for beep/noise tests")
    p.add_argument("--noise-mode", choices=["white", "periodic"], default="white")
    p.add_argument("--sequence", default="preset1", help="Sequence preset: preset1, preset2, arp1")
    p.add_argument("--bpm", type=float, default=120.0, help="Tempo for sequence test")
    p.add_argument("--vol", type=int, default=4, help="0 loudest .. 15 mute (default 4)")

    p.add_argument("--dump-regs", action="store_true", help="Print register dump before/after test")
    p.add_argument("--counters", action="store_true", help="Print debug counters at end")
    p.add_argument("--debug", action="store_true", help="Verbose debug (rate-limited)")

    args = p.parse_args(argv)

    cfg = AppConfig(DEBUG=bool(args.debug))
    th = TestHarness(cfg)

    if args.dump_regs:
        print(th.chip.dump_regs())
        print()

    if args.test == "beep":
        th.run_beep(freq_hz=float(args.freq), seconds=float(args.seconds), vol=int(args.vol))
    elif args.test == "noise":
        th.run_noise(mode=str(args.noise_mode), seconds=float(args.seconds), vol=int(args.vol))
    elif args.test == "sequence":
        th.run_sequence(preset=str(args.sequence), bpm=float(args.bpm), vol=int(args.vol))

    if args.dump_regs:
        print()
        print(th.chip.dump_regs())

    if args.counters:
        print()
        print(th.chip.dump_counters())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())