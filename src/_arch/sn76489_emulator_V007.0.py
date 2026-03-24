#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SN76489 Emulator v0.07
macOS / Python 3.12
Single-file implementation

Features:
- SN76489 core: 3 tone + 1 noise
- Multi-chip engine (1..128)
- Stereo routing: global pan + per-voice pan
- ADSR-lite envelope via register-style volume steps
- Voice allocation + deterministic voice stealing
- Velocity curves: linear / log / exp
- Sustain pedal (CC64)
- Pitch bend (+/- 2 semitone)
- MIDI input via python-rtmidi
- VGM playback subset
- Stable RUN CONFIG / dump-regs / counters / debug

Dependencies:
    pip install numpy sounddevice
    pip install python-rtmidi   # for MIDI input
    brew install portaudio      # if needed on macOS
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BLOCK_FRAMES = 512
DEFAULT_MASTER_GAIN = 0.25
DEFAULT_SN_CLOCK_HZ = 3579545.0
DEFAULT_VGM_BASE_DIR = (
    "/Volumes/data1/Yandex.Disk.localized/michiele/Arduino/PCB Ontwerp/"
    "KiCAD/github/SN76489-synth-midi/src/tmp/src/"
)


# -----------------------------
# Helpers
# -----------------------------

def hard_fail(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}")
    raise SystemExit(code)


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def hex_seed(v: int) -> str:
    return f"0x{(int(v) & 0x7FFF):04X}"


def midi_note_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((int(note) - 69) / 12.0))


def freq_to_period(freq_hz: float, clock_hz: float) -> int:
    if freq_hz <= 0.0:
        return 0x3FF
    p = int(round(clock_hz / (32.0 * freq_hz)))
    return clamp_int(p, 1, 0x3FF)


def period_to_freq(period: int, clock_hz: float) -> float:
    p = max(1, int(period))
    return clock_hz / (32.0 * p)


def vol4_to_amp(vol: int) -> float:
    # 0 loudest, 15 silent
    v = clamp_int(vol, 0, 15)
    if v >= 15:
        return 0.0
    return float(2.0 ** (-(v / 2.0)))


# -----------------------------
# Config / counters
# -----------------------------

@dataclass
class Config:
    mode: str = "test"
    test: str = "beep"

    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_frames: int = DEFAULT_BLOCK_FRAMES
    chips: int = 1
    pan: str = "both"
    master_gain: float = DEFAULT_MASTER_GAIN

    attack_ms: float = 5.0
    decay_ms: float = 80.0
    sustain_vol: int = 8
    release_ms: float = 120.0

    noise_mode: str = "white"
    noise_rate: str = "div32"
    noise_seed: int = 0x4000

    velocity_curve: str = "linear"
    voice_pan: str = "default"

    midi_in: bool = False
    midi_port: str = "none"

    vgm_path: str = "none"
    vgm_base_dir: str = DEFAULT_VGM_BASE_DIR
    vgm_loop: bool = False
    vgm_speed: float = 1.0

    dump_regs: bool = False
    counters: bool = False
    debug: bool = False


@dataclass
class ChipCounters:
    writes_total: int = 0
    writes_latch: int = 0
    writes_data: int = 0
    renders: int = 0
    frames: int = 0


@dataclass
class EngineCounters:
    midi_events_total: int = 0
    note_on_total: int = 0
    note_off_total: int = 0
    pitch_bend_events_total: int = 0
    voices_used_total: int = 0
    note_ignored_no_voice: int = 0
    voice_steal_events_total: int = 0
    sustain_hold_events_total: int = 0
    sustain_release_events_total: int = 0
    envelope_steps_total: int = 0

    vgm_commands_total: int = 0
    vgm_psg_writes_total: int = 0
    vgm_wait_events_total: int = 0
    vgm_wait_samples_total: int = 0
    vgm_loops_total: int = 0


@dataclass
class Voice:
    voice_id: int
    midi_note: Optional[int] = None
    midi_channel: Optional[int] = None
    velocity: int = 0
    active: bool = False
    phase: str = "IDLE"
    allocation_time: int = -1
    sustain_hold: bool = False
    current_period: int = 0x3FF
    base_period: int = 0x3FF
    current_volume: int = 15
    target_volume: int = 15
    pan: str = "both"
    pitch_bend_value: int = 8192
    envelope_target_volume: int = 15
    envelope_step_counter: int = 0
    attack_interval_blocks: int = 1
    decay_interval_blocks: int = 1
    release_interval_blocks: int = 1


# -----------------------------
# SN76489 chip
# -----------------------------

class SN76489Chip:
    def __init__(self, chip_id: int, sample_rate: int, clock_hz: float = DEFAULT_SN_CLOCK_HZ):
        self.chip_id = chip_id
        self.sample_rate = int(sample_rate)
        self.clock_hz = float(clock_hz)

        # Logical registers:
        # 0 tone0 period, 1 vol0, 2 tone1 period, 3 vol1,
        # 4 tone2 period, 5 vol2, 6 noise ctrl, 7 noise vol
        self.regs: List[int] = [0] * 8
        self.regs[0] = 0x3FF
        self.regs[1] = 15
        self.regs[2] = 0x3FF
        self.regs[3] = 15
        self.regs[4] = 0x3FF
        self.regs[5] = 15
        self.regs[6] = 0x00
        self.regs[7] = 15

        self.latched_reg: int = 0

        self.tone_phase = [0, 0, 0]
        self.tone_counter = [0.0, 0.0, 0.0]

        self.noise_lfsr = 0x4000
        self.noise_out = 1
        self.noise_counter = 0.0

        self.counters = ChipCounters()

    def set_noise_seed(self, seed: int) -> None:
        seed &= 0x7FFF
        self.noise_lfsr = seed if seed != 0 else 0x4000

    def write_byte(self, b: int) -> None:
        b &= 0xFF
        self.counters.writes_total += 1

        if (b & 0x80) != 0:
            self.counters.writes_latch += 1
            reg = (b >> 4) & 0x07
            data = b & 0x0F
            self.latched_reg = reg

            if reg in (0, 2, 4):
                cur = self.regs[reg] & 0x3F0
                self.regs[reg] = cur | data
            elif reg == 6:
                self.regs[6] = data & 0x0F
            else:
                self.regs[reg] = data & 0x0F
        else:
            self.counters.writes_data += 1
            data = b & 0x3F
            reg = self.latched_reg
            if reg in (0, 2, 4):
                low = self.regs[reg] & 0x00F
                self.regs[reg] = ((data << 4) & 0x3F0) | low

    def set_tone_period(self, voice_id: int, period: int) -> None:
        reg = [0, 2, 4][voice_id]
        self.regs[reg] = clamp_int(period, 1, 0x3FF)
        self.counters.writes_total += 1

    def set_tone_volume(self, voice_id: int, vol: int) -> None:
        reg = [1, 3, 5][voice_id]
        self.regs[reg] = clamp_int(vol, 0, 15)
        self.counters.writes_total += 1

    def set_noise_ctrl(self, mode: str, rate: str) -> None:
        mode_bit = 1 if mode == "white" else 0
        rate_bits = {"div16": 0, "div32": 1, "div64": 2, "tone2": 3}[rate]
        self.regs[6] = ((mode_bit << 2) | rate_bits) & 0x0F

    def _tone_step_samples(self, period: int) -> float:
        f = period_to_freq(period, self.clock_hz)
        if f <= 0.0:
            return float(self.sample_rate)
        half_cycle_s = 0.5 / f
        return max(1.0, half_cycle_s * self.sample_rate)

    def _noise_step_samples(self) -> float:
        ctrl = self.regs[6] & 0x0F
        rate = ctrl & 0x03
        if rate == 0:
            f = self.clock_hz / (32.0 * 16.0)
        elif rate == 1:
            f = self.clock_hz / (32.0 * 32.0)
        elif rate == 2:
            f = self.clock_hz / (32.0 * 64.0)
        else:
            p2 = max(1, int(self.regs[4]))
            f = period_to_freq(p2, self.clock_hz)
        if f <= 0:
            return float(self.sample_rate)
        return max(1.0, (1.0 / f) * self.sample_rate)

    def render_mono(self, nframes: int) -> np.ndarray:
        out = np.zeros((nframes,), dtype=np.float32)

        a0 = vol4_to_amp(self.regs[1])
        a1 = vol4_to_amp(self.regs[3])
        a2 = vol4_to_amp(self.regs[5])
        an = vol4_to_amp(self.regs[7])

        step0 = self._tone_step_samples(self.regs[0])
        step1 = self._tone_step_samples(self.regs[2])
        step2 = self._tone_step_samples(self.regs[4])
        nstep = self._noise_step_samples()

        ctrl = self.regs[6] & 0x0F
        noise_is_white = ((ctrl >> 2) & 0x01) == 1

        for i in range(nframes):
            for idx, step in enumerate((step0, step1, step2)):
                self.tone_counter[idx] -= 1.0
                if self.tone_counter[idx] <= 0.0:
                    self.tone_counter[idx] += step
                    self.tone_phase[idx] ^= 1

            s0 = (1.0 if self.tone_phase[0] else -1.0) * a0
            s1 = (1.0 if self.tone_phase[1] else -1.0) * a1
            s2 = (1.0 if self.tone_phase[2] else -1.0) * a2

            self.noise_counter -= 1.0
            if self.noise_counter <= 0.0:
                self.noise_counter += nstep
                if noise_is_white:
                    fb = (self.noise_lfsr ^ (self.noise_lfsr >> 1)) & 0x01
                else:
                    fb = self.noise_lfsr & 0x01
                self.noise_lfsr = ((self.noise_lfsr >> 1) | (fb << 14)) & 0x7FFF
                self.noise_out = self.noise_lfsr & 0x01

            sn = (1.0 if self.noise_out else -1.0) * an
            out[i] = (s0 + s1 + s2 + sn)

        out *= 0.25
        self.counters.renders += 1
        self.counters.frames += nframes
        return out


# -----------------------------
# Engine
# -----------------------------

class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chips: List[SN76489Chip] = [
            SN76489Chip(i, cfg.sample_rate, DEFAULT_SN_CLOCK_HZ) for i in range(cfg.chips)
        ]
        self.voices: List[List[Voice]] = [
            [Voice(0), Voice(1), Voice(2)] for _ in range(cfg.chips)
        ]
        self.active_notes: Dict[Tuple[int, int, int], int] = {}
        self.engine_counters = EngineCounters()
        self.alloc_counter = 0

        self.sustain_by_channel: Dict[int, bool] = {ch: False for ch in range(1, 17)}
        self.pitch_bend_by_channel: Dict[int, int] = {ch: 8192 for ch in range(1, 17)}

        self._sd_stream = None
        self._debug_lines_this_second = 0
        self._debug_sec = int(time.time())
        self.vgm_index: Dict[str, str] = {}

        for chip in self.chips:
            chip.set_noise_seed(cfg.noise_seed)
            chip.set_noise_ctrl(cfg.noise_mode, cfg.noise_rate)

    # ---------- debug helpers ----------

    def debug(self, line: str) -> None:
        if not self.cfg.debug:
            return
        now_sec = int(time.time())
        if now_sec != self._debug_sec:
            self._debug_sec = now_sec
            self._debug_lines_this_second = 0
        if self._debug_lines_this_second >= 20:
            return
        self._debug_lines_this_second += 1
        print(f"DEBUG: {line}")

    # ---------- run config ----------

    def print_run_config(self) -> None:
        c = self.cfg
        print("RUN CONFIG:")
        print(f"mode={c.mode}")
        print(f"test={c.test}")
        print(f"sample_rate={c.sample_rate}")
        print(f"block_frames={c.block_frames}")
        print(f"chips={c.chips}")
        print(f"pan={c.pan}")
        print(f"master_gain={c.master_gain}")
        print(f"attack_ms={c.attack_ms}")
        print(f"decay_ms={c.decay_ms}")
        print(f"sustain_vol={c.sustain_vol}")
        print(f"release_ms={c.release_ms}")
        print(f"noise_mode={c.noise_mode}")
        print(f"noise_rate={c.noise_rate}")
        print(f"noise_seed={hex_seed(c.noise_seed)}")
        print(f"velocity_curve={c.velocity_curve}")
        print(f"voice_pan={c.voice_pan}")
        print(f"midi_in={1 if c.midi_in else 0}")
        print(f"midi_port={c.midi_port}")
        print(f"vgm_path={c.vgm_path}")
        print(f"vgm_base_dir={c.vgm_base_dir}")
        print(f"vgm_loop={1 if c.vgm_loop else 0}")
        print(f"vgm_speed={c.vgm_speed}")
        print(f"dump_regs={1 if c.dump_regs else 0}")
        print(f"counters={1 if c.counters else 0}")
        print(f"debug={1 if c.debug else 0}")

    # ---------- timing helpers ----------

    def block_duration_ms(self) -> float:
        return (self.cfg.block_frames / self.cfg.sample_rate) * 1000.0

    def blocks_for_ms(self, ms: float) -> int:
        if ms <= 0.0:
            return 1
        return max(1, int(math.ceil(ms / self.block_duration_ms())))

    # ---------- curves / period ----------

    def velocity_to_volume(self, velocity: int) -> int:
        velocity = clamp_int(velocity, 1, 127)
        curve = self.cfg.velocity_curve

        if curve == "linear":
            vol = 15 - math.floor(velocity / 8.5)
        elif curve == "log":
            vol = 15 - math.floor((math.log2(max(1, velocity)) / math.log2(127)) * 15)
        elif curve == "exp":
            vol = 15 - math.floor(((velocity / 127.0) ** 2.0) * 15)
        else:
            vol = 15 - math.floor(velocity / 8.5)
        return clamp_int(vol, 0, 15)

    def pitch_bend_to_offset(self, bend_value: int) -> float:
        bend = (bend_value - 8192) / 8192.0
        return bend * 2.0

    def bent_period_for_note(self, note: int, bend_value: int) -> int:
        base_freq = midi_note_to_freq(note)
        offset = self.pitch_bend_to_offset(bend_value)
        bent_freq = base_freq * (2.0 ** (offset / 12.0))
        return freq_to_period(bent_freq, DEFAULT_SN_CLOCK_HZ)

    # ---------- pan helpers ----------

    def default_voice_pan(self, voice_id: int) -> str:
        mode = self.cfg.voice_pan
        if mode == "center":
            return "both"
        # default == spread
        return {0: "left", 1: "both", 2: "right"}[voice_id]

    def effective_voice_pan(self, voice_pan: str) -> str:
        if self.cfg.pan == "left":
            return "left"
        if self.cfg.pan == "right":
            return "right"
        return voice_pan

    def pan_gains(self, pan: str) -> Tuple[float, float]:
        if pan == "left":
            return (1.0, 0.0)
        if pan == "right":
            return (0.0, 1.0)
        return (1.0, 1.0)

    # ---------- note mapping ----------

    def chip_for_channel(self, midi_channel: int) -> int:
        return (midi_channel - 1) % self.cfg.chips

    def note_key(self, midi_channel: int, midi_note: int, chip_id: int) -> Tuple[int, int, int]:
        return (midi_channel, midi_note, chip_id)

    # ---------- voice allocation ----------

    def find_free_voice(self, chip_id: int) -> Optional[Voice]:
        for v in self.voices[chip_id]:
            if v.phase == "IDLE" or not v.active:
                return v
        return None

    def select_voice_to_steal(self, chip_id: int) -> Voice:
        voices = self.voices[chip_id]

        release_voices = [v for v in voices if v.phase == "RELEASE" and v.active]
        if release_voices:
            victim = min(release_voices, key=lambda x: x.allocation_time)
            reason = "release"
        else:
            held_voices = [v for v in voices if v.phase == "SUSTAIN" and v.sustain_hold and v.active]
            if held_voices:
                victim = min(held_voices, key=lambda x: x.allocation_time)
                reason = "sustain_hold"
            else:
                active_voices = [v for v in voices if v.active]
                victim = min(active_voices, key=lambda x: x.allocation_time)
                reason = "oldest"

        self.engine_counters.voice_steal_events_total += 1
        self.debug(
            f"VOICE_STEAL chip={chip_id} voice={victim.voice_id} old_note={victim.midi_note} "
            f"old_phase={victim.phase} reason={reason}"
        )
        return victim

    def allocate_voice(self, midi_channel: int, midi_note: int, velocity: int) -> Tuple[int, Voice]:
        chip_id = self.chip_for_channel(midi_channel)
        free = self.find_free_voice(chip_id)
        if free is not None:
            return chip_id, free
        self.engine_counters.note_ignored_no_voice += 0  # explicit no-op for compatibility
        return chip_id, self.select_voice_to_steal(chip_id)

    # ---------- envelope init/update ----------

    def init_envelope_for_voice(self, v: Voice, target_volume: int) -> None:
        v.phase = "ATTACK"
        v.current_volume = 15
        v.target_volume = target_volume
        v.envelope_target_volume = target_volume
        v.envelope_step_counter = 0
        v.attack_interval_blocks = max(1, self.blocks_for_ms(self.cfg.attack_ms) // max(1, 15 - target_volume))
        sustain_diff = abs(self.cfg.sustain_vol - target_volume)
        v.decay_interval_blocks = max(1, self.blocks_for_ms(self.cfg.decay_ms) // max(1, sustain_diff if sustain_diff > 0 else 1))
        release_diff = max(1, 15 - self.cfg.sustain_vol)
        v.release_interval_blocks = max(1, self.blocks_for_ms(self.cfg.release_ms) // release_diff)
        v.sustain_hold = False

    def update_voice_envelope(self, chip_id: int, v: Voice) -> None:
        if not v.active:
            return

        v.envelope_step_counter += 1
        chip = self.chips[chip_id]

        if v.phase == "ATTACK":
            if v.envelope_step_counter >= v.attack_interval_blocks:
                v.envelope_step_counter = 0
                old = v.current_volume
                v.current_volume = max(v.target_volume, v.current_volume - 1)
                if v.current_volume != old:
                    chip.set_tone_volume(v.voice_id, v.current_volume)
                    self.engine_counters.envelope_steps_total += 1
                    self.debug(
                        f"ENVELOPE_STEP chip={chip_id} voice={v.voice_id} phase=ATTACK old={old} new={v.current_volume}"
                    )
                if v.current_volume <= v.target_volume:
                    v.phase = "DECAY"

        elif v.phase == "DECAY":
            if v.envelope_step_counter >= v.decay_interval_blocks:
                v.envelope_step_counter = 0
                old = v.current_volume
                if v.current_volume < self.cfg.sustain_vol:
                    v.current_volume = min(self.cfg.sustain_vol, v.current_volume + 1)
                elif v.current_volume > self.cfg.sustain_vol:
                    v.current_volume = max(self.cfg.sustain_vol, v.current_volume - 1)

                if v.current_volume != old:
                    chip.set_tone_volume(v.voice_id, v.current_volume)
                    self.engine_counters.envelope_steps_total += 1
                    self.debug(
                        f"ENVELOPE_STEP chip={chip_id} voice={v.voice_id} phase=DECAY old={old} new={v.current_volume}"
                    )
                if v.current_volume == self.cfg.sustain_vol:
                    v.phase = "SUSTAIN"

        elif v.phase == "SUSTAIN":
            # steady
            pass

        elif v.phase == "RELEASE":
            if v.envelope_step_counter >= v.release_interval_blocks:
                v.envelope_step_counter = 0
                old = v.current_volume
                v.current_volume = min(15, v.current_volume + 1)
                if v.current_volume != old:
                    chip.set_tone_volume(v.voice_id, v.current_volume)
                    self.engine_counters.envelope_steps_total += 1
                    self.debug(
                        f"ENVELOPE_STEP chip={chip_id} voice={v.voice_id} phase=RELEASE old={old} new={v.current_volume}"
                    )
                if v.current_volume >= 15:
                    v.phase = "IDLE"
                    v.active = False
                    v.midi_note = None
                    v.midi_channel = None
                    v.velocity = 0
                    v.sustain_hold = False

        elif v.phase == "IDLE":
            pass

    # ---------- MIDI event handlers ----------

    def handle_note_on(self, midi_channel: int, midi_note: int, velocity: int) -> None:
        self.engine_counters.midi_events_total += 1
        self.engine_counters.note_on_total += 1

        if velocity == 0:
            self.handle_note_off(midi_channel, midi_note)
            return

        chip_id, v = self.allocate_voice(midi_channel, midi_note, velocity)
        key = self.note_key(midi_channel, midi_note, chip_id)

        # If stealing, remove old mapping
        if v.active and v.midi_note is not None and v.midi_channel is not None:
            old_key = self.note_key(v.midi_channel, v.midi_note, chip_id)
            self.active_notes.pop(old_key, None)

        self.alloc_counter += 1
        v.voice_id = v.voice_id
        v.midi_note = midi_note
        v.midi_channel = midi_channel
        v.velocity = velocity
        v.active = True
        v.allocation_time = self.alloc_counter
        v.pan = self.default_voice_pan(v.voice_id)
        v.pitch_bend_value = self.pitch_bend_by_channel[midi_channel]
        v.base_period = freq_to_period(midi_note_to_freq(midi_note), DEFAULT_SN_CLOCK_HZ)
        v.current_period = self.bent_period_for_note(midi_note, v.pitch_bend_value)

        target_vol = self.velocity_to_volume(velocity)
        self.init_envelope_for_voice(v, target_vol)

        chip = self.chips[chip_id]
        chip.set_tone_period(v.voice_id, v.current_period)
        chip.set_tone_volume(v.voice_id, v.current_volume)

        self.active_notes[key] = v.voice_id
        self.engine_counters.voices_used_total += 1
        self.debug(
            f"VOICE_ASSIGN chip={chip_id} voice={v.voice_id} ch={midi_channel} note={midi_note} "
            f"vel={velocity} period={v.current_period} vol={v.current_volume}"
        )

    def handle_note_off(self, midi_channel: int, midi_note: int) -> None:
        self.engine_counters.midi_events_total += 1
        self.engine_counters.note_off_total += 1

        chip_id = self.chip_for_channel(midi_channel)
        key = self.note_key(midi_channel, midi_note, chip_id)
        voice_id = self.active_notes.get(key)

        if voice_id is None:
            return

        v = self.voices[chip_id][voice_id]
        if self.sustain_by_channel[midi_channel]:
            v.phase = "SUSTAIN"
            v.sustain_hold = True
            self.engine_counters.sustain_hold_events_total += 1
            self.debug(f"SUSTAIN_HOLD chip={chip_id} voice={voice_id} ch={midi_channel} note={midi_note}")
        else:
            v.phase = "RELEASE"
            v.envelope_step_counter = 0

    def handle_pitch_bend(self, midi_channel: int, raw_value: int) -> None:
        self.engine_counters.midi_events_total += 1
        self.engine_counters.pitch_bend_events_total += 1

        raw_value = clamp_int(raw_value, 0, 16383)
        self.pitch_bend_by_channel[midi_channel] = raw_value
        chip_id = self.chip_for_channel(midi_channel)

        for v in self.voices[chip_id]:
            if v.active and v.midi_channel == midi_channel and v.midi_note is not None:
                v.pitch_bend_value = raw_value
                new_period = self.bent_period_for_note(v.midi_note, raw_value)
                if new_period != v.current_period:
                    v.current_period = new_period
                    self.chips[chip_id].set_tone_period(v.voice_id, v.current_period)

        offset = self.pitch_bend_to_offset(raw_value)
        self.debug(f"PITCH_BEND_UPDATE ch={midi_channel} raw={raw_value} semitone_offset={offset:.4f}")

    def handle_cc64(self, midi_channel: int, value: int) -> None:
        was_on = self.sustain_by_channel[midi_channel]
        now_on = value >= 64
        self.sustain_by_channel[midi_channel] = now_on
        chip_id = self.chip_for_channel(midi_channel)

        if was_on and not now_on:
            for v in self.voices[chip_id]:
                if v.active and v.midi_channel == midi_channel and v.sustain_hold:
                    v.sustain_hold = False
                    v.phase = "RELEASE"
                    v.envelope_step_counter = 0
                    self.engine_counters.sustain_release_events_total += 1
                    self.debug(f"SUSTAIN_RELEASE chip={chip_id} voice={v.voice_id} ch={midi_channel}")

    # ---------- per-block engine update ----------

    def update_block_state(self) -> None:
        for chip_id in range(self.cfg.chips):
            for v in self.voices[chip_id]:
                if v.active and v.midi_note is not None and v.midi_channel is not None:
                    # pitch bend at block boundaries
                    new_period = self.bent_period_for_note(v.midi_note, self.pitch_bend_by_channel[v.midi_channel])
                    if new_period != v.current_period:
                        v.current_period = new_period
                        self.chips[chip_id].set_tone_period(v.voice_id, v.current_period)
                self.update_voice_envelope(chip_id, v)

    # ---------- render ----------

    def render_stereo_block(self, frames: int) -> np.ndarray:
        left = np.zeros((frames,), dtype=np.float32)
        right = np.zeros((frames,), dtype=np.float32)

        for chip_id, chip in enumerate(self.chips):
            tone_blocks = [np.zeros((frames,), dtype=np.float32) for _ in range(3)]

            # Temporarily isolate tone voices by muting others one at a time
            original_vols = [chip.regs[1], chip.regs[3], chip.regs[5], chip.regs[7]]

            # render each tone voice separately for per-voice pan
            for voice_id in range(3):
                chip.regs[1], chip.regs[3], chip.regs[5], chip.regs[7] = 15, 15, 15, 15
                chip.regs[[1, 3, 5][voice_id]] = original_vols[[1, 3, 5][voice_id] // 2] if False else original_vols[[1, 3, 5][voice_id] // 2]
                # easier explicit:
                if voice_id == 0:
                    chip.regs[1] = original_vols[0]
                elif voice_id == 1:
                    chip.regs[3] = original_vols[1]
                else:
                    chip.regs[5] = original_vols[2]
                tone_blocks[voice_id] = chip.render_mono(frames)

            # render noise separately
            chip.regs[1], chip.regs[3], chip.regs[5], chip.regs[7] = 15, 15, 15, original_vols[3]
            noise_block = chip.render_mono(frames)

            # restore
            chip.regs[1], chip.regs[3], chip.regs[5], chip.regs[7] = original_vols

            # route tone voices
            for voice_id, mono in enumerate(tone_blocks):
                v = self.voices[chip_id][voice_id]
                pan = self.effective_voice_pan(v.pan)
                gl, gr = self.pan_gains(pan)
                left += mono * gl
                right += mono * gr

            # route noise using global pan only
            if self.cfg.pan == "left":
                left += noise_block
            elif self.cfg.pan == "right":
                right += noise_block
            else:
                left += noise_block
                right += noise_block

        stereo = np.stack([left, right], axis=1)
        stereo *= float(self.cfg.master_gain)
        np.clip(stereo, -1.0, 1.0, out=stereo)
        return stereo.astype(np.float32, copy=False)

    # ---------- audio backend ----------

    def open_stream(self):
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            hard_fail("Missing audio backend. Install: pip install sounddevice numpy (and brew install portaudio).")

        def callback(outdata, frames, time_info, status):
            self.update_block_state()
            outdata[:] = self.render_stereo_block(frames)

        self._sd_stream = sd.OutputStream(
            samplerate=self.cfg.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.cfg.block_frames,
            callback=callback,
        )
        self._sd_stream.start()

    def close_stream(self):
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
            except Exception:
                pass
            try:
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

    # ---------- dump / counters ----------

    def dump_regs(self):
        for chip in self.chips:
            r = chip.regs
            print(f"CHIP {chip.chip_id}:")
            print(f"  latched={chip.latched_reg}")
            print(f"  R0={r[0]:04X}  R1={r[1]:02X}  R2={r[2]:04X}  R3={r[3]:02X}")
            print(f"  R4={r[4]:04X}  R5={r[5]:02X}  R6={r[6]:02X}  R7={r[7]:02X}")
            print(
                f"  tone0_hz={period_to_freq(max(1, r[0]), DEFAULT_SN_CLOCK_HZ):.2f} "
                f"tone1_hz={period_to_freq(max(1, r[2]), DEFAULT_SN_CLOCK_HZ):.2f} "
                f"tone2_hz={period_to_freq(max(1, r[4]), DEFAULT_SN_CLOCK_HZ):.2f}"
            )
            nr = {0: "div16", 1: "div32", 2: "div64", 3: "tone2"}[r[6] & 0x03]
            nm = "white" if ((r[6] >> 2) & 1) else "periodic"
            print(f"  noise_mode={nm} noise_rate={nr} noise_seed={hex_seed(chip.noise_lfsr)}")

            for v in self.voices[chip.chip_id]:
                print(
                    f"  voice_id={v.voice_id} midi_note={v.midi_note} midi_channel={v.midi_channel} "
                    f"velocity={v.velocity} phase={v.phase} active={1 if v.active else 0} "
                    f"sustain_hold={1 if v.sustain_hold else 0} current_volume={v.current_volume} "
                    f"current_period={v.current_period} pan={v.pan} pitch_bend_value={v.pitch_bend_value}"
                )

    def print_counters(self):
        for chip in self.chips:
            c = chip.counters
            print(f"CHIP {chip.chip_id} COUNTERS:")
            print(f"  writes_total={c.writes_total}")
            print(f"  writes_latch={c.writes_latch}")
            print(f"  writes_data={c.writes_data}")
            print(f"  renders={c.renders}")
            print(f"  frames={c.frames}")

        ec = self.engine_counters
        print("ENGINE COUNTERS:")
        print(f"  midi_events_total={ec.midi_events_total}")
        print(f"  note_on_total={ec.note_on_total}")
        print(f"  note_off_total={ec.note_off_total}")
        print(f"  voice_steal_events_total={ec.voice_steal_events_total}")
        print(f"  sustain_hold_events_total={ec.sustain_hold_events_total}")
        print(f"  sustain_release_events_total={ec.sustain_release_events_total}")
        print(f"  pitch_bend_events_total={ec.pitch_bend_events_total}")
        print(f"  envelope_steps_total={ec.envelope_steps_total}")
        print(f"  voices_used_total={ec.voices_used_total}")
        print(f"  note_ignored_no_voice={ec.note_ignored_no_voice}")

        print("VGM COUNTERS:")
        print(f"  vgm_commands_total={ec.vgm_commands_total}")
        print(f"  vgm_psg_writes_total={ec.vgm_psg_writes_total}")
        print(f"  vgm_wait_events_total={ec.vgm_wait_events_total}")
        print(f"  vgm_wait_samples_total={ec.vgm_wait_samples_total}")
        print(f"  vgm_loops_total={ec.vgm_loops_total}")

    # ---------- tests ----------

    def run_test(self, seconds: float, freq: float):
        test = self.cfg.test
        self.open_stream()
        try:
            if test == "beep":
                self._test_beep(seconds, freq)
            elif test == "noise":
                self._test_noise(seconds)
            elif test == "sequence":
                self._test_sequence(seconds)
            elif test == "chords":
                self._test_chords(seconds)
            elif test == "sweep":
                self._test_sweep(seconds)
            else:
                hard_fail("Unknown test mode.")
        finally:
            self.close_stream()

    def _test_beep(self, seconds: float, freq: float):
        chip = self.chips[0]
        chip.set_tone_period(0, freq_to_period(freq, DEFAULT_SN_CLOCK_HZ))
        chip.set_tone_volume(0, 2)
        chip.set_tone_volume(1, 15)
        chip.set_tone_volume(2, 15)
        chip.regs[7] = 15
        time.sleep(seconds)

    def _test_noise(self, seconds: float):
        chip = self.chips[0]
        chip.set_noise_ctrl(self.cfg.noise_mode, self.cfg.noise_rate)
        chip.set_noise_seed(self.cfg.noise_seed)
        chip.regs[1] = 15
        chip.regs[3] = 15
        chip.regs[5] = 15
        chip.regs[7] = 2
        time.sleep(seconds)

    def _test_sequence(self, seconds: float):
        notes = [(60, 100), (64, 80), (67, 120), (72, 90)]
        step = max(0.08, seconds / max(1, len(notes)))
        for n, vel in notes:
            self.handle_note_on(1, n, vel)
            time.sleep(step / 2)
            self.handle_note_off(1, n)
            time.sleep(step / 2)

    def _test_chords(self, seconds: float):
        self.handle_note_on(1, 60, 110)
        self.handle_note_on(1, 64, 100)
        self.handle_note_on(1, 67, 120)
        time.sleep(seconds)
        self.handle_note_off(1, 60)
        self.handle_note_off(1, 64)
        self.handle_note_off(1, 67)
        time.sleep(min(0.5, seconds / 2))

    def _test_sweep(self, seconds: float):
        start = time.time()
        self.handle_note_on(1, 57, 110)
        while time.time() - start < seconds:
            t = (time.time() - start) / max(seconds, 0.001)
            bend = int(8192 + (math.sin(t * math.pi) * 8191))
            self.handle_pitch_bend(1, bend)
            time.sleep(self.block_duration_ms() / 1000.0)
        self.handle_note_off(1, 57)

    # ---------- MIDI ----------

    def midi_list(self):
        try:
            import rtmidi  # type: ignore
        except Exception:
            hard_fail("Missing MIDI backend. Install: pip install python-rtmidi")
        mi = rtmidi.MidiIn()
        ports = mi.get_ports()
        print("MIDI PORTS:")
        for idx, name in enumerate(ports):
            print(f"  [{idx}] {name}")

    def run_midi(self):
        try:
            import rtmidi  # type: ignore
        except Exception:
            hard_fail("Missing MIDI backend. Install: pip install python-rtmidi")

        mi = rtmidi.MidiIn()
        ports = mi.get_ports()
        if not ports:
            hard_fail("No MIDI input ports available.")

        if self.cfg.midi_port in ("auto", "none", ""):
            port_index = 0
        else:
            needle = self.cfg.midi_port.lower()
            matches = [i for i, p in enumerate(ports) if needle in p.lower()]
            if not matches:
                hard_fail(f"MIDI port not found matching substring: {self.cfg.midi_port}")
            port_index = matches[0]

        mi.open_port(port_index)
        self.open_stream()
        try:
            while True:
                msg = mi.get_message()
                if msg:
                    data, _dt = msg
                    if data:
                        status = data[0] & 0xF0
                        ch = (data[0] & 0x0F) + 1
                        if status == 0x90:
                            note = data[1]
                            vel = data[2]
                            self.handle_note_on(ch, note, vel)
                        elif status == 0x80:
                            self.handle_note_off(ch, data[1])
                        elif status == 0xE0:
                            raw = (data[2] << 7) | data[1]
                            self.handle_pitch_bend(ch, raw)
                        elif status == 0xB0 and data[1] == 64:
                            self.handle_cc64(ch, data[2])
                time.sleep(self.block_duration_ms() / 1000.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.close_stream()
            try:
                mi.close_port()
            except Exception:
                pass

    # ---------- VGM ----------

    def vgm_list(self):
        base = os.path.expanduser(self.cfg.vgm_base_dir)
        if not os.path.isdir(base):
            hard_fail(f"VGM base dir not found: {base}")
        files = sorted([f for f in os.listdir(base) if f.lower().endswith(".vgm")], key=str.lower)
        self.vgm_index = {f: os.path.join(base, f) for f in files}
        print(f"VGM LIST: {base}")
        for f in files:
            print(f)

    def _u16le(self, b: bytes, o: int) -> int:
        return int.from_bytes(b[o:o+2], "little", signed=False)

    def _u32le(self, b: bytes, o: int) -> int:
        return int.from_bytes(b[o:o+4], "little", signed=False)

    def run_vgm(self):
        path = os.path.expanduser(self.cfg.vgm_path)
        if not os.path.isfile(path):
            hard_fail(f"VGM file not found: {path}")
        if self.cfg.vgm_speed <= 0.0:
            hard_fail("--vgm-speed must be > 0.")

        data = open(path, "rb").read()
        if len(data) < 0x40:
            hard_fail("VGM file too small.")
        if data[:4] != b"Vgm ":
            hard_fail("Invalid VGM magic (expected 'Vgm ').")

        data_offset = self._u32le(data, 0x34)
        start = 0x40 if data_offset == 0 else 0x34 + data_offset
        if start >= len(data):
            hard_fail("VGM data offset out of range.")

        self.open_stream()
        pos = start
        sr_scale = self.cfg.sample_rate / 44100.0
        try:
            while True:
                if pos >= len(data):
                    if self.cfg.vgm_loop:
                        self.engine_counters.vgm_loops_total += 1
                        pos = start
                        continue
                    break

                cmd = data[pos]
                pos += 1
                self.engine_counters.vgm_commands_total += 1

                if cmd == 0x50:
                    if pos >= len(data):
                        hard_fail("Truncated PSG write.")
                    dd = data[pos]
                    pos += 1
                    self.chips[0].write_byte(dd)
                    self.engine_counters.vgm_psg_writes_total += 1

                elif cmd == 0x61:
                    if pos + 2 > len(data):
                        hard_fail("Truncated VGM wait 0x61.")
                    n = self._u16le(data, pos)
                    pos += 2
                    self.engine_counters.vgm_wait_events_total += 1
                    self._vgm_wait_scaled(n, sr_scale)

                elif cmd == 0x62:
                    self.engine_counters.vgm_wait_events_total += 1
                    self._vgm_wait_scaled(735, sr_scale)

                elif cmd == 0x63:
                    self.engine_counters.vgm_wait_events_total += 1
                    self._vgm_wait_scaled(882, sr_scale)

                elif 0x70 <= cmd <= 0x7F:
                    n = (cmd & 0x0F) + 1
                    self.engine_counters.vgm_wait_events_total += 1
                    self._vgm_wait_scaled(n, sr_scale)

                elif cmd == 0x66:
                    if self.cfg.vgm_loop:
                        self.engine_counters.vgm_loops_total += 1
                        pos = start
                    else:
                        break

                else:
                    hard_fail(f"Unsupported VGM command 0x{cmd:02X} at offset 0x{(pos-1):08X}")
        except KeyboardInterrupt:
            pass
        finally:
            self.close_stream()

    def _vgm_wait_scaled(self, wait_vgm_samples: int, sr_scale: float):
        wait_engine = int(round(wait_vgm_samples * sr_scale))
        effective = int(round(wait_engine / self.cfg.vgm_speed))
        if wait_engine > 0 and effective < 1:
            effective = 1
        self.engine_counters.vgm_wait_samples_total += effective

        remaining = effective
        while remaining > 0:
            time.sleep(min(self.cfg.block_frames, remaining) / self.cfg.sample_rate)
            remaining -= min(self.cfg.block_frames, remaining)


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sn76489_emulator.py")

    p.add_argument("--test", choices=["beep", "noise", "sequence", "chords", "sweep"])
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--freq", type=float, default=440.0)

    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--block-frames", type=int, default=DEFAULT_BLOCK_FRAMES)
    p.add_argument("--chips", type=int, default=1)
    p.add_argument("--pan", choices=["left", "right", "both"], default="both")
    p.add_argument("--master-gain", type=float, default=DEFAULT_MASTER_GAIN)

    p.add_argument("--attack-ms", type=float, default=5.0)
    p.add_argument("--decay-ms", type=float, default=80.0)
    p.add_argument("--sustain-vol", type=int, default=8)
    p.add_argument("--release-ms", type=float, default=120.0)

    p.add_argument("--noise-mode", choices=["white", "periodic"], default="white")
    p.add_argument("--noise-rate", choices=["div16", "div32", "div64", "tone2"], default="div32")
    p.add_argument("--noise-seed", type=lambda s: int(s, 0), default=0x4000)

    p.add_argument("--velocity-curve", choices=["linear", "log", "exp"], default="linear")
    p.add_argument("--voice-pan", choices=["default", "center", "spread"], default="default")

    p.add_argument("--dump-regs", action="store_true")
    p.add_argument("--counters", action="store_true")
    p.add_argument("--debug", action="store_true")

    p.add_argument("--midi-list", action="store_true")
    p.add_argument("--midi-in", action="store_true")
    p.add_argument("--midi-port", type=str, default="auto")

    p.add_argument("--vgm-path", type=str, default=None)
    p.add_argument("--vgm-base-dir", type=str, default=DEFAULT_VGM_BASE_DIR)
    p.add_argument("--vgm-loop", action="store_true")
    p.add_argument("--vgm-speed", type=float, default=1.0)
    p.add_argument("--vgm-list", action="store_true")

    return p


def determine_mode(args: argparse.Namespace) -> Tuple[str, str]:
    has_test = args.test is not None
    has_midi = bool(args.midi_in)
    has_midi_list = bool(args.midi_list)
    has_vgm = args.vgm_path is not None
    has_vgm_list = bool(args.vgm_list)

    if has_midi and has_vgm:
        hard_fail("Choose either --midi-in OR --vgm-path (mutually exclusive).")
    if has_test and has_vgm:
        hard_fail("Choose either --test OR --vgm-path (mutually exclusive).")
    if has_test and has_midi:
        hard_fail("Choose either --test OR --midi-in (mutually exclusive).")
    if has_midi_list and (has_midi or has_test or has_vgm or has_vgm_list):
        hard_fail("Choose --midi-list alone (do not combine with playback modes).")
    if has_vgm_list and (has_midi or has_test or has_vgm or has_midi_list):
        hard_fail("Choose --vgm-list alone (do not combine with playback modes).")

    if has_midi_list:
        return ("midi_list", "none")
    if has_vgm_list:
        return ("vgm_list", "none")
    if has_midi:
        return ("midi", "none")
    if has_vgm:
        return ("vgm", "none")
    if has_test:
        return ("test", args.test)
    return ("test", "beep")


def cfg_from_args(args: argparse.Namespace, mode: str, test: str) -> Config:
    chips = clamp_int(args.chips, 1, 128)
    if chips != args.chips:
        hard_fail("--chips must be in range 1..128")

    if args.vgm_speed <= 0:
        hard_fail("--vgm-speed must be > 0.")
    if args.noise_rate not in ("div16", "div32", "div64", "tone2"):
        hard_fail("Invalid --noise-rate.")

    return Config(
        mode=mode,
        test=test,
        sample_rate=clamp_int(args.sample_rate, 8000, 192000),
        block_frames=clamp_int(args.block_frames, 64, 8192),
        chips=chips,
        pan=args.pan,
        master_gain=clamp_float(args.master_gain, 0.0, 4.0),
        attack_ms=clamp_float(args.attack_ms, 0.0, 5000.0),
        decay_ms=clamp_float(args.decay_ms, 0.0, 5000.0),
        sustain_vol=clamp_int(args.sustain_vol, 0, 15),
        release_ms=clamp_float(args.release_ms, 0.0, 5000.0),
        noise_mode=args.noise_mode,
        noise_rate=args.noise_rate,
        noise_seed=int(args.noise_seed),
        velocity_curve=args.velocity_curve,
        voice_pan=args.voice_pan,
        midi_in=bool(args.midi_in),
        midi_port=args.midi_port if bool(args.midi_in) else "none",
        vgm_path=args.vgm_path if args.vgm_path is not None else "none",
        vgm_base_dir=args.vgm_base_dir,
        vgm_loop=bool(args.vgm_loop),
        vgm_speed=float(args.vgm_speed),
        dump_regs=bool(args.dump_regs),
        counters=bool(args.counters),
        debug=bool(args.debug),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    mode, test = determine_mode(args)
    cfg = cfg_from_args(args, mode, test)
    eng = Engine(cfg)
    eng.print_run_config()

    try:
        if mode == "midi_list":
            eng.midi_list()
        elif mode == "vgm_list":
            eng.vgm_list()
        elif mode == "midi":
            eng.run_midi()
        elif mode == "vgm":
            eng.run_vgm()
        elif mode == "test":
            eng.run_test(cfg.seconds if hasattr(cfg, "seconds") else args.seconds, args.freq)
        else:
            hard_fail(f"Unknown mode: {mode}")
    finally:
        if cfg.dump_regs:
            eng.dump_regs()
        if cfg.counters:
            eng.print_counters()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

