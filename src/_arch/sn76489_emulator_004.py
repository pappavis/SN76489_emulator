#!/usr/bin/env python3
# SN76489 Emulator v0.04 — 2026-02-02
# TS v1.3: CoreMIDI input + channel→chip mapping, noise rate+seed, envelope attack/decay via volume-writes
# Python 3.12 / MacOS

from __future__ import annotations
import argparse
import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
import threading

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import rtmidi  # python-rtmidi
except ImportError:
    rtmidi = None


# =========================
# Config / helpers
# =========================

@dataclass
class AppConfig:
    sample_rate: int = 44100
    master_clock_hz: int = 3_579_545
    block_frames: int = 512
    master_gain: float = 0.25
    chip_gain: float = 1.0
    pan: str = "both"  # left|right|both
    debug: bool = False


def clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def clamp_float(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def note_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def freq_to_period(master_clock_hz: int, freq: float) -> int:
    if freq <= 0:
        return 1
    p = int(round(master_clock_hz / (32.0 * freq)))
    return clamp_int(p, 1, 1023)


def velocity_to_sn_volume(velocity: int) -> int:
    # TS v1.3 mapping: volume = 15 - ceil(velocity/9)
    if velocity <= 0:
        return 15
    v = 15 - int(math.ceil(velocity / 9.0))
    return clamp_int(v, 0, 15)


def parse_int_maybe_hex(s: str) -> int:
    s = s.strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    return int(s, 10)


# =========================
# SN76489 Core (mono)
# =========================

class ToneChannel:
    def __init__(self, sr: int, clk: int):
        self.sr = sr
        self.clk = clk
        self.period = 0
        self.volume = 15
        self.phase = 0.0

    def freq(self) -> float:
        if self.period <= 0:
            return 0.0
        return self.clk / (32.0 * self.period)

    def render(self, n: int, amp_lut: np.ndarray) -> np.ndarray:
        if self.volume >= 15 or self.period <= 0:
            return np.zeros(n, np.float32)

        f = self.freq()
        inc = f / self.sr
        out = np.empty(n, np.float32)
        ph = self.phase
        a = float(amp_lut[self.volume])

        for i in range(n):
            out[i] = (1.0 if ph < 0.5 else -1.0) * a
            ph += inc
            if ph >= 1.0:
                ph -= 1.0

        self.phase = ph
        return out


class NoiseChannel:
    def __init__(self, sr: int, clk: int):
        self.sr = sr
        self.clk = clk
        self.ctrl = 0
        self.volume = 15
        self.lfsr = 0x4000
        self.phase = 0.0
        self.tone2_period = lambda: 0

    def set_seed(self, seed: int) -> None:
        seed &= 0x7FFF
        if seed == 0:
            seed = 0x4000
        # keep to 15-bit LFSR domain, typical init uses bit 14 set
        self.lfsr = seed

    def render(self, n: int, amp_lut: np.ndarray) -> np.ndarray:
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
            p = int(self.tone2_period())
            f = self.clk / (32 * p) if p > 0 else 0.0

        if f <= 0:
            return np.zeros(n, np.float32)

        inc = f / self.sr
        out = np.empty(n, np.float32)
        ph = self.phase
        bit = self.lfsr & 1
        a = float(amp_lut[self.volume])

        for i in range(n):
            ph += inc
            if ph >= 1.0:
                ph -= 1.0

                lsb = self.lfsr & 1
                self.lfsr >>= 1

                # White: XOR taps (simple 2-bit XOR), Periodic: feedback is LSB
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

        # Simple log-ish volume LUT; volume=15 is mute.
        self.amp = np.array(
            [0.0 if i == 15 else 10 ** (-i / 10) for i in range(16)],
            dtype=np.float32,
        )

        # Counters
        self.writes_total = 0
        self.writes_latch = 0
        self.writes_data = 0
        self.renders = 0
        self.frames = 0

    # ---- SN76489 register write emulation
    def write(self, b: int) -> None:
        self.writes_total += 1
        if b & 0x80:
            self.writes_latch += 1
            self.latched = (b >> 4) & 0x07
            val = b & 0x0F
            self._write_nibble(self.latched, val)
        else:
            self.writes_data += 1
            # Only tone period regs accept the data byte here (R0/R2/R4)
            if self.latched in (0, 2, 4):
                lo = self.regs[self.latched] & 0x0F
                self.regs[self.latched] = ((b & 0x3F) << 4) | lo
                self._apply(self.latched)

    def _write_nibble(self, r: int, v: int) -> None:
        if r in (0, 2, 4):
            self.regs[r] = (self.regs[r] & 0x3F0) | (v & 0x0F)
        else:
            self.regs[r] = v & 0x0F
        self._apply(r)

    def _apply(self, r: int) -> None:
        if r == 0:
            self.tone[0].period = int(self.regs[0])
        elif r == 1:
            self.tone[0].volume = int(self.regs[1])
        elif r == 2:
            self.tone[1].period = int(self.regs[2])
        elif r == 3:
            self.tone[1].volume = int(self.regs[3])
        elif r == 4:
            self.tone[2].period = int(self.regs[4])
        elif r == 5:
            self.tone[2].volume = int(self.regs[5])
        elif r == 6:
            self.noise.ctrl = int(self.regs[6]) & 0x0F
        elif r == 7:
            self.noise.volume = int(self.regs[7])

    def render_mono(self, n: int) -> np.ndarray:
        self.renders += 1
        self.frames += n
        return (
            self.tone[0].render(n, self.amp)
            + self.tone[1].render(n, self.amp)
            + self.tone[2].render(n, self.amp)
            + self.noise.render(n, self.amp)
        )


# =========================
# Envelope engine (attack/decay via volume writes)
# =========================

@dataclass
class EnvState:
    active: bool = False
    phase: str = "off"  # "attack" | "decay" | "off"
    vol_target: int = 15
    current_vol: int = 15
    step_interval_samples: int = 1
    next_step_in_samples: int = 0


class EnvelopeEngine:
    """
    TS v1.3:
      - Attack: 15 -> vol_target
      - Decay:  vol_target -> 15
      - Steps are volume register writes (quantized)
      - Block-accurate scheduling acceptable
    """
    def __init__(self, sample_rate: int, attack_ms: float, decay_ms: float):
        self.sr = sample_rate
        self.attack_ms = max(0.0, attack_ms)
        self.decay_ms = max(0.0, decay_ms)
        self.states: Dict[int, EnvState] = {}  # key: chip_id
        self.env_steps_total = 0

    def configure(self, attack_ms: float, decay_ms: float) -> None:
        self.attack_ms = max(0.0, attack_ms)
        self.decay_ms = max(0.0, decay_ms)

    def note_on(self, chip_id: int, vol_target: int) -> None:
        vol_target = clamp_int(vol_target, 0, 14)
        st = self.states.get(chip_id, EnvState())
        st.active = True
        st.phase = "attack"
        st.vol_target = vol_target
        st.current_vol = 15

        steps_attack = max(1, 15 - vol_target)
        dt_attack_ms = self.attack_ms / steps_attack if steps_attack > 0 else 0.0
        interval = max(1, int(round((dt_attack_ms / 1000.0) * self.sr)))
        st.step_interval_samples = interval
        st.next_step_in_samples = 0  # step immediately at next tick
        self.states[chip_id] = st

    def note_off(self, chip_id: int) -> None:
        # For v0.04, NOTE_OFF triggers decay to mute from current/target
        st = self.states.get(chip_id)
        if not st or not st.active:
            return
        st.phase = "decay"
        st.next_step_in_samples = 0

        # decay steps: from current toward 15
        steps_decay = max(1, 15 - st.current_vol)
        dt_decay_ms = self.decay_ms / steps_decay if steps_decay > 0 else 0.0
        interval = max(1, int(round((dt_decay_ms / 1000.0) * self.sr)))
        st.step_interval_samples = interval
        self.states[chip_id] = st

    def tick(self, frames: int, apply_volume_write) -> None:
        """
        Called once per audio block.
        apply_volume_write(chip_id, volume_0_15) -> None
        """
        for chip_id, st in list(self.states.items()):
            if not st.active or st.phase == "off":
                continue

            remaining = frames
            # block-accurate stepping; we may execute multiple steps inside one block
            while remaining > 0 and st.active and st.phase != "off":
                if st.next_step_in_samples > remaining:
                    st.next_step_in_samples -= remaining
                    remaining = 0
                    break

                # Consume time until step
                remaining -= st.next_step_in_samples
                st.next_step_in_samples = st.step_interval_samples

                if st.phase == "attack":
                    # Step: 15 -> vol_target (downwards)
                    if st.current_vol > st.vol_target:
                        st.current_vol -= 1
                        apply_volume_write(chip_id, st.current_vol)
                        self.env_steps_total += 1
                    if st.current_vol <= st.vol_target:
                        # Auto transition into decay after reaching target
                        st.phase = "decay"
                        st.next_step_in_samples = 0
                        steps_decay = max(1, 15 - st.current_vol)
                        dt_decay_ms = self.decay_ms / steps_decay if steps_decay > 0 else 0.0
                        st.step_interval_samples = max(1, int(round((dt_decay_ms / 1000.0) * self.sr)))

                elif st.phase == "decay":
                    if st.current_vol < 15:
                        st.current_vol += 1
                        apply_volume_write(chip_id, st.current_vol)
                        self.env_steps_total += 1
                    if st.current_vol >= 15:
                        st.phase = "off"
                        st.active = False

            self.states[chip_id] = st


# =========================
# ChipBank + stereo mixer
# =========================

class ChipBank:
    def __init__(self, cfg: AppConfig, chips: int):
        chips = clamp_int(chips, 1, 128)
        self.cfg = cfg
        self.chips: List[SN76489Core] = [SN76489Core(cfg) for _ in range(chips)]

        self.midi_events_total = 0
        self.note_on_total = 0
        self.note_off_total = 0

    def _pan_gains(self) -> Tuple[float, float]:
        return {
            "left": (1.0, 0.0),
            "right": (0.0, 1.0),
            "both": (1.0, 1.0),
        }[self.cfg.pan]

    def render_stereo(self, n: int) -> np.ndarray:
        left = np.zeros(n, np.float32)
        right = np.zeros(n, np.float32)
        pan_l, pan_r = self._pan_gains()

        for chip in self.chips:
            mono = chip.render_mono(n) * float(self.cfg.chip_gain)
            left += mono * pan_l
            right += mono * pan_r

        left *= float(self.cfg.master_gain)
        right *= float(self.cfg.master_gain)
        np.clip(left, -1.0, 1.0, out=left)
        np.clip(right, -1.0, 1.0, out=right)
        return np.stack([left, right], axis=1)

    def dump_regs_text(self) -> str:
        lines = []
        for i, c in enumerate(self.chips):
            lines.append(f"CHIP {i}: latched={c.latched} regs=" + " ".join(f"R{r}={c.regs[r]:03d}/0x{c.regs[r]:02X}" for r in range(8)))
            # Derived
            f0 = c.tone[0].freq()
            f1 = c.tone[1].freq()
            f2 = c.tone[2].freq()
            rate = c.noise.ctrl & 0x03
            white = (c.noise.ctrl >> 2) & 1
            rate_s = {0: "div16", 1: "div32", 2: "div64", 3: "tone2"}[rate]
            mode_s = "white" if white else "periodic"
            lines.append(f"  derived: tone_freqs=[{f0:.2f},{f1:.2f},{f2:.2f}] noise_mode={mode_s} noise_rate={rate_s} noise_seed=0x{c.noise.lfsr & 0x7FFF:04X}")
        return "\n".join(lines)

    def dump_counters_text(self, env: Optional[EnvelopeEngine] = None) -> str:
        lines = []
        for i, c in enumerate(self.chips):
            lines.append(
                f"CHIP {i}: writes_total={c.writes_total} latch={c.writes_latch} data={c.writes_data} "
                f"renders={c.renders} frames={c.frames}"
            )
        lines.append(f"TOTAL: midi_events_total={self.midi_events_total} note_on_total={self.note_on_total} note_off_total={self.note_off_total}")
        if env is not None:
            lines.append(f"TOTAL: env_steps_total={env.env_steps_total}")
        return "\n".join(lines)


# =========================
# Audio out
# =========================

class AudioOut:
    def __init__(self, cfg: AppConfig):
        if sd is None:
            raise RuntimeError("Audio backend missing. Install: pip install sounddevice")
        self.cfg = cfg

    def play_blocking(self, stereo_buf: np.ndarray) -> None:
        sd.play(stereo_buf, self.cfg.sample_rate, blocking=True)
        sd.stop()

    def stream_run(self, render_callback, block_frames: int) -> None:
        """
        Start a continuous OutputStream. Stops on KeyboardInterrupt.
        render_callback(frames)-> stereo float32 array shape (frames,2)
        """
        def cb(outdata, frames, _time, status):
            if status:
                # Keep minimal; avoid flooding
                pass
            buf = render_callback(frames)
            outdata[:] = buf

        with sd.OutputStream(
            samplerate=self.cfg.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=block_frames,
            callback=cb,
        ):
            while True:
                time.sleep(0.05)


# =========================
# MIDI input (python-rtmidi)
# =========================

class MidiIn:
    """
    Minimal CoreMIDI input wrapper using python-rtmidi.
    Pushes raw MIDI messages into a thread-safe queue.
    """
    def __init__(self):
        if rtmidi is None:
            raise RuntimeError("MIDI backend missing. Install: pip install python-rtmidi")
        self.midi = rtmidi.MidiIn()
        self.queue: Deque[Tuple[List[int], float]] = deque()
        self.lock = threading.Lock()

    def list_ports(self) -> List[str]:
        return list(self.midi.get_ports())

    def open_port_by_name(self, name: Optional[str]) -> str:
        ports = self.list_ports()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        if name is None:
            self.midi.open_port(0)
            opened = ports[0]
        else:
            idx = None
            for i, p in enumerate(ports):
                if name.lower() in p.lower():
                    idx = i
                    break
            if idx is None:
                raise RuntimeError(f"MIDI port not found: {name}")
            self.midi.open_port(idx)
            opened = ports[idx]

        def _cb(msg, _data=None):
            message, dt = msg
            with self.lock:
                self.queue.append((list(message), float(dt)))

        self.midi.set_callback(_cb)
        return opened

    def pop_all(self) -> List[Tuple[List[int], float]]:
        with self.lock:
            items = list(self.queue)
            self.queue.clear()
        return items


# =========================
# Test harness + engine ops
# =========================

def sn_write_tone0_period(chip: SN76489Core, period: int) -> None:
    period = clamp_int(period, 1, 1023)
    lo = period & 0x0F
    hi = (period >> 4) & 0x3F
    chip.write(0x80 | lo)  # latch R0 + low nibble
    chip.write(hi)         # data byte updates high bits


def sn_write_tone0_volume(chip: SN76489Core, vol: int) -> None:
    vol = clamp_int(vol, 0, 15)
    chip.write(0x90 | (vol & 0x0F))  # latch R1 volume nibble


def sn_write_noise_ctrl(chip: SN76489Core, mode: str, rate: str) -> None:
    # R6 nibble: bits1..0 rate, bit2 mode(white=1), bit3 reserved=0
    rate_bits = {"div16": 0, "div32": 1, "div64": 2, "tone2": 3}[rate]
    white_bit = 1 if mode == "white" else 0
    ctrl = (white_bit << 2) | rate_bits
    chip.write(0xE0 | (ctrl & 0x0F))  # latch R6


def sn_write_noise_volume(chip: SN76489Core, vol: int) -> None:
    vol = clamp_int(vol, 0, 15)
    chip.write(0xF0 | (vol & 0x0F))  # latch R7 volume


# =========================
# Sequencer (simple preset)
# =========================

def sequence_preset1() -> List[int]:
    # C4-E4-G4-C5 MIDI notes
    return [60, 64, 67, 72]


# =========================
# Main
# =========================

def main():
    p = argparse.ArgumentParser()
    # Modes: tests OR midi input
    p.add_argument("--test", choices=["beep", "noise", "sequence"], default=None)
    p.add_argument("--midi-in", action="store_true")
    p.add_argument("--midi-list", action="store_true")
    p.add_argument("--midi-port", type=str, default=None)

    # Common engine params
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--block-frames", type=int, default=512)
    p.add_argument("--chips", type=int, default=1)
    p.add_argument("--pan", choices=["left", "right", "both"], default="both")
    p.add_argument("--master-gain", type=float, default=0.25)
    p.add_argument("--chip-gain", type=float, default=1.0)

    # Debug / inspect
    p.add_argument("--dump-regs", action="store_true")
    p.add_argument("--counters", action="store_true")
    p.add_argument("--debug", action="store_true")

    # Test params
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--freq", type=float, default=440.0)

    # Noise params (v0.04)
    p.add_argument("--noise-mode", choices=["white", "periodic"], default="white")
    p.add_argument("--noise-rate", choices=["div16", "div32", "div64", "tone2"], default="div32")
    p.add_argument("--noise-seed", type=str, default="0x4000")

    # Envelope params (v0.04)
    p.add_argument("--attack-ms", type=float, default=5.0)
    p.add_argument("--decay-ms", type=float, default=80.0)

    # Sequence params
    p.add_argument("--bpm", type=float, default=120.0)
    p.add_argument("--sequence", type=str, default="preset1")

    args = p.parse_args()

    if sd is None:
        raise SystemExit("Missing dependency: pip install sounddevice numpy")

    cfg = AppConfig(
        sample_rate=args.sample_rate,
        block_frames=args.block_frames,
        master_gain=clamp_float(args.master_gain, 0.0, 2.0),
        chip_gain=clamp_float(args.chip_gain, 0.0, 2.0),
        pan=args.pan,
        debug=args.debug,
    )

    audio = AudioOut(cfg)
    bank = ChipBank(cfg, args.chips)
    env = EnvelopeEngine(cfg.sample_rate, args.attack_ms, args.decay_ms)

    # MIDI list only
    if args.midi_list:
        if rtmidi is None:
            raise SystemExit("Missing dependency: pip install python-rtmidi")
        mi = MidiIn()
        ports = mi.list_ports()
        if not ports:
            print("No MIDI input ports found.")
        else:
            print("MIDI input ports:")
            for i, name in enumerate(ports):
                print(f"  [{i}] {name}")
        return

    # Choose mode
    if args.midi_in and args.test is not None:
        raise SystemExit("Choose either --midi-in OR --test, not both.")

    if not args.midi_in and args.test is None:
        # default to beep if nothing specified (keeps old behavior)
        args.test = "beep"

    # ---- TEST MODES (blocking render)
    if args.test is not None:
        frames = int(cfg.sample_rate * max(0.05, args.seconds))

        chip0 = bank.chips[0]

        if args.test == "beep":
            period = freq_to_period(cfg.master_clock_hz, float(args.freq))
            sn_write_tone0_period(chip0, period)
            sn_write_tone0_volume(chip0, 4)  # audible

        elif args.test == "noise":
            seed = parse_int_maybe_hex(args.noise_seed)
            chip0.noise.set_seed(seed)
            sn_write_noise_ctrl(chip0, args.noise_mode, args.noise_rate)
            sn_write_noise_volume(chip0, 4)  # audible
            # mute tones to isolate noise
            sn_write_tone0_volume(chip0, 15)
            bank.chips[0].tone[1].volume = 15
            bank.chips[0].tone[2].volume = 15

        elif args.test == "sequence":
            # Simple step sequencer: one note per beat (quarter note)
            notes = sequence_preset1() if args.sequence == "preset1" else sequence_preset1()
            sec_per_beat = 60.0 / max(1.0, float(args.bpm))
            note_frames = int(cfg.sample_rate * sec_per_beat)
            total_frames = 0

            # Pre-allocate target length based on args.seconds if provided >0
            # If seconds explicitly set, play up to that; else play full pattern once.
            limit_frames = frames if args.seconds > 0 else note_frames * len(notes)

            chunks = []
            idx = 0
            while total_frames < limit_frames:
                note = notes[idx % len(notes)]
                idx += 1

                # NOTE ON: program tone0 period and start envelope
                period = freq_to_period(cfg.master_clock_hz, note_to_freq(note))
                sn_write_tone0_period(chip0, period)
                # target volume from a fixed "velocity-like" value (e.g., 96)
                vol_target = velocity_to_sn_volume(96)
                env.note_on(0, vol_target)

                # render this note duration in blocks, ticking envelope
                remain = min(note_frames, limit_frames - total_frames)
                while remain > 0:
                    blk = min(cfg.block_frames, remain)
                    # apply envelope steps for chip0 via volume writes
                    env.tick(
                        blk,
                        apply_volume_write=lambda chip_id, vol: sn_write_tone0_volume(bank.chips[chip_id], vol),
                    )
                    chunks.append(bank.render_stereo(blk))
                    total_frames += blk
                    remain -= blk

                # NOTE OFF: decay to mute (envelope)
                env.note_off(0)

        else:
            raise SystemExit("Unknown test mode")

        if args.test != "sequence":
            buf = bank.render_stereo(frames)
            audio.play_blocking(buf)
        else:
            buf = np.concatenate(chunks, axis=0) if chunks else bank.render_stereo(frames)
            audio.play_blocking(buf)

        if args.dump_regs:
            print(bank.dump_regs_text())
        if args.counters:
            print(bank.dump_counters_text(env))

        return

    # ---- MIDI MODE (continuous stream)
    if args.midi_in:
        if rtmidi is None:
            raise SystemExit("Missing dependency: pip install python-rtmidi")

        mi = MidiIn()
        opened = mi.open_port_by_name(args.midi_port)
        print(f"MIDI IN: {opened}")
        print("Running. Stop with Ctrl+C.")

        # State: current note per chip (tone0 only)
        current_note: Dict[int, int] = {}

        def apply_volume(chip_id: int, vol: int) -> None:
            sn_write_tone0_volume(bank.chips[chip_id], vol)

        def process_midi_messages() -> None:
            # Pop all, translate to writes
            msgs = mi.pop_all()
            if not msgs:
                return
            for message, _dt in msgs:
                if not message:
                    continue
                status = message[0] & 0xF0
                ch = (message[0] & 0x0F) + 1  # 1..16
                chip_id = (ch - 1) % len(bank.chips)

                # NOTE ON/OFF are 3-byte messages
                if status == 0x90 and len(message) >= 3:  # Note On
                    note = int(message[1])
                    vel = int(message[2])
                    bank.midi_events_total += 1
                    if vel == 0:
                        # treat as Note Off
                        bank.note_off_total += 1
                        env.note_off(chip_id)
                        current_note.pop(chip_id, None)
                        continue

                    bank.note_on_total += 1
                    current_note[chip_id] = note

                    period = freq_to_period(cfg.master_clock_hz, note_to_freq(note))
                    sn_write_tone0_period(bank.chips[chip_id], period)

                    vol_target = velocity_to_sn_volume(vel)
                    env.note_on(chip_id, vol_target)

                    if cfg.debug:
                        print(f"NOTE_ON ch={ch} chip={chip_id} note={note} vel={vel} period={period} vol_target={vol_target}")

                elif status == 0x80 and len(message) >= 3:  # Note Off
                    note = int(message[1])
                    bank.midi_events_total += 1
                    bank.note_off_total += 1
                    env.note_off(chip_id)
                    current_note.pop(chip_id, None)
                    if cfg.debug:
                        print(f"NOTE_OFF ch={ch} chip={chip_id} note={note}")

                else:
                    # ignore others in v0.04
                    pass

        # Audio render callback used by sounddevice stream
        def render_callback(frames: int) -> np.ndarray:
            # 1) process midi messages (non-blocking queue pop)
            process_midi_messages()
            # 2) envelope tick (volume writes)
            env.tick(frames, apply_volume_write=apply_volume)
            # 3) render audio
            return bank.render_stereo(frames)

        try:
            audio.stream_run(render_callback, cfg.block_frames)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if args.dump_regs:
                print(bank.dump_regs_text())
            if args.counters:
                print(bank.dump_counters_text(env))


if __name__ == "__main__":
    main()
