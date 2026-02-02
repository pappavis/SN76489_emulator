#!/usr/bin/env python3
# SN76489 Emulator
# v0.06 2026-02-02
#
# Single-file implementation (MacOS-focused, Python 3.12)
# Core rule: audio is produced ONLY by SN76489-style register writes.
#
# v0.06 adds:
# - VGM playback subset (0x50, 0x61, 0x62, 0x63, 0x70..0x7F, 0x66)
# - vgm list/loop/speed
# - strict RUN CONFIG echo contract
# - strict CLI mutual exclusivity (test vs midi vs vgm)
# - counters include VGM metrics
#
# Dependencies:
#   pip install numpy sounddevice
#   pip install python-rtmidi  (for MIDI mode)
#   brew install portaudio     (if sounddevice fails)
#
# Notes:
# - Emulation is "audio-rate accurate" (not cycle-perfect).
# - Tone is square-wave via divider model, noise via LFSR.
# - Envelope is implemented via volume register writes (ADSR-lite), evaluated block-accurate.

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def _hard_fail(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}")
    raise SystemExit(code)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _hz_to_sn_period(hz: float, sn_clock_hz: float) -> int:
    """
    SN76489 tone frequency formula (approx):
      f_out = clock / (32 * period)
    => period = clock / (32 * f_out)
    """
    if hz <= 0:
        return 0x3FF
    p = int(round(sn_clock_hz / (32.0 * hz)))
    return _clamp_int(p, 1, 0x3FF)


def _sn_period_to_hz(period: int, sn_clock_hz: float) -> float:
    period = max(1, int(period))
    return sn_clock_hz / (32.0 * period)


def _vol4_to_amp(vol4: int) -> float:
    """
    SN76489 volume is 4-bit attenuation, 0 = loudest, 15 = mute.
    We map to a simple exponential-ish curve to sound reasonable.
    """
    v = _clamp_int(vol4, 0, 15)
    if v >= 15:
        return 0.0
    return float(2.0 ** (-(v / 2.0)))


@dataclasses.dataclass
class SnCounters:
    writes_total: int = 0
    writes_latch: int = 0
    writes_data: int = 0
    renders: int = 0
    frames: int = 0


@dataclasses.dataclass
class VoiceCounters:
    midi_events_total: int = 0
    note_on_total: int = 0
    note_off_total: int = 0
    pitch_bend_events: int = 0
    voices_used_total: int = 0
    note_ignored_no_voice: int = 0
    env_steps_total: int = 0


@dataclasses.dataclass
class VgmCounters:
    vgm_commands_total: int = 0
    vgm_psg_writes_total: int = 0
    vgm_wait_events_total: int = 0
    vgm_wait_samples_total: int = 0
    vgm_loops_total: int = 0


@dataclasses.dataclass
class VoiceState:
    active: bool = False
    midi_note: int = -1
    base_period: int = 0x3FF
    current_period: int = 0x3FF
    target_vol: int = 8
    sustain_vol: int = 8
    phase: str = "IDLE"  # IDLE/ATTACK/DECAY/SUSTAIN/RELEASE
    attack_samples: int = 0
    decay_samples: int = 0
    release_samples: int = 0
    phase_rem: int = 0
    cur_vol: int = 15
    sustain_hold: bool = False


class SN76489Chip:
    def __init__(self, chip_index: int, sn_clock_hz: float = 3579545.0, sample_rate: int = 44100):
        self.chip_index = chip_index
        self.sn_clock_hz = float(sn_clock_hz)
        self.sample_rate = int(sample_rate)

        self.regs: List[int] = [0] * 8
        self.regs[0] = 0x3FF
        self.regs[2] = 0x3FF
        self.regs[4] = 0x3FF
        self.regs[6] = 0x00
        self.regs[1] = 0x0F
        self.regs[3] = 0x0F
        self.regs[5] = 0x0F
        self.regs[7] = 0x0F

        self.latched_reg: int = 0

        self._tone_phase: List[int] = [0, 0, 0]
        self._tone_counter: List[float] = [0.0, 0.0, 0.0]

        self.noise_lfsr: int = 0x4000
        self.noise_out: int = 1
        self._noise_counter: float = 0.0

        self.counters = SnCounters()

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

    def _tone_step_samples(self, period: int) -> float:
        period = max(1, int(period))
        f = _sn_period_to_hz(period, self.sn_clock_hz)
        if f <= 0.0:
            return float(self.sample_rate)
        half_cycle = 0.5 / f
        return max(1.0, half_cycle * self.sample_rate)

    def _noise_step_samples(self) -> float:
        ctrl = self.regs[6] & 0x0F
        rate = ctrl & 0x03
        if rate == 0:
            f = self.sn_clock_hz / (32.0 * 16.0)
        elif rate == 1:
            f = self.sn_clock_hz / (32.0 * 32.0)
        elif rate == 2:
            f = self.sn_clock_hz / (32.0 * 64.0)
        else:
            p2 = max(1, int(self.regs[4]))
            f = _sn_period_to_hz(p2, self.sn_clock_hz)
        if f <= 0:
            return float(self.sample_rate)
        return max(1.0, (1.0 / f) * self.sample_rate)

    def set_noise_seed(self, seed: int) -> None:
        seed &= 0x7FFF
        if seed == 0:
            seed = 0x4000
        self.noise_lfsr = seed

    def render_mono(self, nframes: int) -> np.ndarray:
        nframes = int(nframes)
        out = np.zeros((nframes,), dtype=np.float32)

        a0 = _vol4_to_amp(self.regs[1])
        a1 = _vol4_to_amp(self.regs[3])
        a2 = _vol4_to_amp(self.regs[5])
        an = _vol4_to_amp(self.regs[7])

        step0 = self._tone_step_samples(self.regs[0])
        step1 = self._tone_step_samples(self.regs[2])
        step2 = self._tone_step_samples(self.regs[4])

        nstep = self._noise_step_samples()
        ctrl = self.regs[6] & 0x0F
        noise_is_white = ((ctrl >> 2) & 0x01) == 1

        for i in range(nframes):
            self._tone_counter[0] -= 1.0
            if self._tone_counter[0] <= 0.0:
                self._tone_counter[0] += step0
                self._tone_phase[0] ^= 1
            s0 = (1.0 if self._tone_phase[0] else -1.0) * a0

            self._tone_counter[1] -= 1.0
            if self._tone_counter[1] <= 0.0:
                self._tone_counter[1] += step1
                self._tone_phase[1] ^= 1
            s1 = (1.0 if self._tone_phase[1] else -1.0) * a1

            self._tone_counter[2] -= 1.0
            if self._tone_counter[2] <= 0.0:
                self._tone_counter[2] += step2
                self._tone_phase[2] ^= 1
            s2 = (1.0 if self._tone_phase[2] else -1.0) * a2

            self._noise_counter -= 1.0
            if self._noise_counter <= 0.0:
                self._noise_counter += nstep
                if noise_is_white:
                    fb = (self.noise_lfsr ^ (self.noise_lfsr >> 1)) & 0x01
                else:
                    fb = self.noise_lfsr & 0x01
                self.noise_lfsr = (self.noise_lfsr >> 1) | (fb << 14)
                self.noise_out = self.noise_lfsr & 0x01
            sn = (1.0 if self.noise_out else -1.0) * an

            out[i] = (s0 + s1 + s2 + sn)

        out *= 0.25
        self.counters.renders += 1
        self.counters.frames += nframes
        return out


@dataclasses.dataclass
class EngineConfig:
    sample_rate: int = 44100
    block_frames: int = 512
    chips: int = 1
    pan: str = "both"
    master_gain: float = 0.25

    attack_ms: float = 5.0
    decay_ms: float = 80.0
    sustain_vol: int = 8
    release_ms: float = 120.0

    noise_mode: str = "white"
    noise_rate: str = "div32"
    noise_seed: int = 0x4000

    dump_regs: bool = False
    counters: bool = False
    debug: bool = False

    mode: str = "test"
    test: str = "beep"

    midi_in: bool = False
    midi_port: str = "none"

    vgm_path: str = "none"
    vgm_base_dir: str = "none"
    vgm_loop: bool = False
    vgm_speed: float = 1.0


class SNEngine:
    def __init__(self, cfg: EngineConfig, sn_clock_hz: float = 3579545.0):
        self.cfg = cfg
        self.sn_clock_hz = float(sn_clock_hz)

        self.chips: List[SN76489Chip] = [
            SN76489Chip(i, sn_clock_hz=self.sn_clock_hz, sample_rate=cfg.sample_rate)
            for i in range(cfg.chips)
        ]

        for ch in self.chips:
            ch.set_noise_seed(cfg.noise_seed)
            self._apply_noise_mode_rate(ch, cfg.noise_mode, cfg.noise_rate)

        self.voices: List[List[VoiceState]] = [
            [VoiceState(), VoiceState(), VoiceState()] for _ in range(cfg.chips)
        ]
        self.voice_counters = VoiceCounters()
        self.vgm_counters = VgmCounters()

        self._debug_last_print_t = 0.0
        self._debug_print_budget = 0

        self._sustain_channel: List[bool] = [False] * 16
        self._pitchbend_channel: List[int] = [8192] * 16

        self._sd_stream = None
        self.vgm_index: Dict[str, str] = {}

    def _debug_log(self, line: str) -> None:
        if not self.cfg.debug:
            return
        now = time.time()
        if now - self._debug_last_print_t >= 1.0:
            self._debug_last_print_t = now
            self._debug_print_budget = 20
        if self._debug_print_budget <= 0:
            return
        self._debug_print_budget -= 1
        print(f"DEBUG: {line}")

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
        print(f"noise_seed=0x{int(c.noise_seed) & 0x7FFF:04X}")
        print(f"midi_in={1 if c.midi_in else 0}")
        print(f"midi_port={c.midi_port}")
        print(f"vgm_path={c.vgm_path}")
        print(f"vgm_base_dir={c.vgm_base_dir}")
        print(f"vgm_loop={1 if c.vgm_loop else 0}")
        print(f"vgm_speed={c.vgm_speed}")
        print(f"dump_regs={1 if c.dump_regs else 0}")
        print(f"counters={1 if c.counters else 0}")
        print(f"debug={1 if c.debug else 0}")

    def _apply_noise_mode_rate(self, chip: SN76489Chip, mode: str, rate: str) -> None:
        mode = (mode or "white").lower()
        rate = (rate or "div32").lower()

        mode_bit = 1 if mode == "white" else 0

        if rate == "div16":
            rate_bits = 0
        elif rate == "div32":
            rate_bits = 1
        elif rate == "div64":
            rate_bits = 2
        elif rate == "tone2":
            rate_bits = 3
        else:
            _hard_fail("Invalid --noise-rate (use div16|div32|div64|tone2).")

        chip.regs[6] = (mode_bit << 2) | rate_bits

    def dump_regs(self) -> None:
        for chip in self.chips:
            idx = chip.chip_index
            r = chip.regs
            print(f"CHIP {idx}:")
            print(f"  latched={chip.latched_reg}")
            print(f"  R0={r[0]:04X}  R1={r[1]:02X}  R2={r[2]:04X}  R3={r[3]:02X}")
            print(f"  R4={r[4]:04X}  R5={r[5]:02X}  R6={r[6]:02X}  R7={r[7]:02X}")

            f0 = _sn_period_to_hz(max(1, r[0]), chip.sn_clock_hz)
            f1 = _sn_period_to_hz(max(1, r[2]), chip.sn_clock_hz)
            f2 = _sn_period_to_hz(max(1, r[4]), chip.sn_clock_hz)
            print(f"  tone0_hz={f0:.2f}  tone1_hz={f1:.2f}  tone2_hz={f2:.2f}")

            nm = "white" if ((r[6] >> 2) & 1) else "periodic"
            nr_bits = r[6] & 0x03
            nr = {0: "div16", 1: "div32", 2: "div64", 3: "tone2"}.get(nr_bits, "div32")
            print(f"  noise_mode={nm}  noise_rate={nr}  noise_seed=0x{chip.noise_lfsr & 0x7FFF:04X}")

            vs = self.voices[idx]
            for vi, v in enumerate(vs):
                print(
                    f"  voice{vi}: active={1 if v.active else 0} note={v.midi_note} "
                    f"phase={v.phase} cur_vol={v.cur_vol} target_vol={v.target_vol} sustain_vol={v.sustain_vol} "
                    f"period={v.current_period:04X}"
                )

    def print_counters(self) -> None:
        for chip in self.chips:
            c = chip.counters
            print(f"CHIP {chip.chip_index} COUNTERS:")
            print(f"  writes_total={c.writes_total} writes_latch={c.writes_latch} writes_data={c.writes_data}")
            print(f"  renders={c.renders} frames={c.frames}")

        vc = self.voice_counters
        print("VOICE COUNTERS:")
        print(f"  midi_events_total={vc.midi_events_total}")
        print(f"  note_on_total={vc.note_on_total} note_off_total={vc.note_off_total} pitch_bend_events={vc.pitch_bend_events}")
        print(f"  voices_used_total={vc.voices_used_total} note_ignored_no_voice={vc.note_ignored_no_voice}")
        print(f"  env_steps_total={vc.env_steps_total}")

        vg = self.vgm_counters
        print("VGM COUNTERS:")
        print(f"  vgm_commands_total={vg.vgm_commands_total}")
        print(f"  vgm_psg_writes_total={vg.vgm_psg_writes_total}")
        print(f"  vgm_wait_events_total={vg.vgm_wait_events_total}")
        print(f"  vgm_wait_samples_total={vg.vgm_wait_samples_total}")
        print(f"  vgm_loops_total={vg.vgm_loops_total}")

    def render_stereo_block(self, nframes: int) -> np.ndarray:
        mix_mono = np.zeros((nframes,), dtype=np.float32)
        for chip in self.chips:
            mix_mono += chip.render_mono(nframes)

        mix_mono *= float(self.cfg.master_gain)

        pan = self.cfg.pan.lower()
        if pan == "left":
            left = mix_mono
            right = np.zeros_like(mix_mono)
        elif pan == "right":
            left = np.zeros_like(mix_mono)
            right = mix_mono
        else:
            left = mix_mono
            right = mix_mono

        stereo = np.stack([left, right], axis=1)
        np.clip(stereo, -1.0, 1.0, out=stereo)
        return stereo.astype(np.float32, copy=False)

    def _ms_to_samples(self, ms: float) -> int:
        return max(0, int(round((float(ms) / 1000.0) * self.cfg.sample_rate)))

    def _voice_apply_vol(self, chip: SN76489Chip, voice_idx: int, vol: int) -> None:
        vol = _clamp_int(vol, 0, 15)
        reg = [1, 3, 5][voice_idx]
        chip.regs[reg] = vol & 0x0F
        chip.counters.writes_total += 1

    def _voice_apply_period(self, chip: SN76489Chip, voice_idx: int, period: int) -> None:
        period = _clamp_int(period, 1, 0x3FF)
        reg = [0, 2, 4][voice_idx]
        chip.regs[reg] = period
        chip.counters.writes_total += 1

    def _step_envelopes_block(self, nframes: int) -> None:
        for ci, chip in enumerate(self.chips):
            for vi, v in enumerate(self.voices[ci]):
                if not v.active:
                    continue

                v.phase_rem -= nframes
                if v.phase_rem > 0:
                    continue

                if v.phase == "ATTACK":
                    if v.cur_vol > v.target_vol:
                        v.cur_vol -= 1
                        self._voice_apply_vol(chip, vi, v.cur_vol)
                        self.voice_counters.env_steps_total += 1
                        if v.cur_vol <= v.target_vol:
                            v.phase = "DECAY"
                            v.phase_rem = max(1, v.decay_samples)
                        else:
                            v.phase_rem = max(1, v.attack_samples // max(1, (15 - v.target_vol)))
                    else:
                        v.phase = "DECAY"
                        v.phase_rem = max(1, v.decay_samples)

                elif v.phase == "DECAY":
                    if v.cur_vol < v.sustain_vol:
                        v.cur_vol += 1
                        self._voice_apply_vol(chip, vi, v.cur_vol)
                        self.voice_counters.env_steps_total += 1
                        if v.cur_vol >= v.sustain_vol:
                            v.phase = "SUSTAIN"
                            v.phase_rem = 10**9
                        else:
                            v.phase_rem = max(1, v.decay_samples // max(1, abs(v.sustain_vol - v.target_vol)))
                    elif v.cur_vol > v.sustain_vol:
                        v.cur_vol -= 1
                        self._voice_apply_vol(chip, vi, v.cur_vol)
                        self.voice_counters.env_steps_total += 1
                        if v.cur_vol <= v.sustain_vol:
                            v.phase = "SUSTAIN"
                            v.phase_rem = 10**9
                        else:
                            v.phase_rem = max(1, v.decay_samples // max(1, abs(v.sustain_vol - v.target_vol)))
                    else:
                        v.phase = "SUSTAIN"
                        v.phase_rem = 10**9

                elif v.phase == "SUSTAIN":
                    v.phase_rem = 10**9

                elif v.phase == "RELEASE":
                    if v.cur_vol < 15:
                        v.cur_vol += 1
                        self._voice_apply_vol(chip, vi, v.cur_vol)
                        self.voice_counters.env_steps_total += 1
                        if v.cur_vol >= 15:
                            v.active = False
                            v.phase = "IDLE"
                            v.midi_note = -1
                        else:
                            v.phase_rem = max(1, v.release_samples // max(1, (15 - v.target_vol)))
                    else:
                        v.active = False
                        v.phase = "IDLE"
                        v.midi_note = -1

    def _velocity_to_target_vol(self, velocity: int) -> int:
        vel = _clamp_int(velocity, 0, 127)
        att = int(round(15 - (vel / 127.0) * 15))
        return _clamp_int(att, 0, 15)

    def _midi_note_to_hz(self, note: int) -> float:
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    def _apply_pitchbend_period(self, midi_channel_0based: int, base_period: int, clock_hz: float) -> int:
        bend = self._pitchbend_channel[_clamp_int(midi_channel_0based, 0, 15)]
        delta = (bend - 8192) / 8192.0
        semis = 2.0 * delta
        freq_mult = 2.0 ** (semis / 12.0)
        hz = _sn_period_to_hz(max(1, base_period), clock_hz) * freq_mult
        return _hz_to_sn_period(hz, clock_hz)

    def note_on(self, midi_channel_0based: int, note: int, velocity: int) -> None:
        self.voice_counters.midi_events_total += 1
        self.voice_counters.note_on_total += 1

        ch = _clamp_int(midi_channel_0based, 0, 15)
        chip_idx = (ch % len(self.chips)) if self.chips else 0
        chip = self.chips[chip_idx]

        voices = self.voices[chip_idx]
        voice_idx = None
        for vi, v in enumerate(voices):
            if not v.active:
                voice_idx = vi
                break
        if voice_idx is None:
            self.voice_counters.note_ignored_no_voice += 1
            self._debug_log(f"NOTE_ON ignored (no voice): ch={ch+1} note={note}")
            return

        v = voices[voice_idx]
        v.active = True
        v.midi_note = int(note)
        v.sustain_vol = _clamp_int(self.cfg.sustain_vol, 0, 15)
        v.target_vol = self._velocity_to_target_vol(velocity)
        v.cur_vol = 15
        v.phase = "ATTACK"
        v.attack_samples = self._ms_to_samples(self.cfg.attack_ms)
        v.decay_samples = self._ms_to_samples(self.cfg.decay_ms)
        v.release_samples = self._ms_to_samples(self.cfg.release_ms)
        v.phase_rem = max(1, v.attack_samples if v.attack_samples > 0 else 1)

        hz = self._midi_note_to_hz(note)
        base_period = _hz_to_sn_period(hz, chip.sn_clock_hz)
        v.base_period = base_period
        v.current_period = self._apply_pitchbend_period(ch, base_period, chip.sn_clock_hz)
        self._voice_apply_period(chip, voice_idx, v.current_period)
        self._voice_apply_vol(chip, voice_idx, v.cur_vol)

        self.voice_counters.voices_used_total += 1
        self._debug_log(f"NOTE_ON: ch={ch+1} chip={chip_idx} voice={voice_idx} note={note} vel={velocity} period=0x{v.current_period:03X}")

    def note_off(self, midi_channel_0based: int, note: int) -> None:
        self.voice_counters.midi_events_total += 1
        self.voice_counters.note_off_total += 1

        ch = _clamp_int(midi_channel_0based, 0, 15)
        chip_idx = (ch % len(self.chips)) if self.chips else 0
        voices = self.voices[chip_idx]

        for vi, v in enumerate(voices):
            if v.active and v.midi_note == int(note):
                if self._sustain_channel[ch]:
                    v.sustain_hold = True
                    v.phase = "SUSTAIN"
                    v.phase_rem = 10**9
                    self._debug_log(f"NOTE_OFF (sustain hold): ch={ch+1} note={note}")
                else:
                    v.phase = "RELEASE"
                    v.phase_rem = max(1, v.release_samples if v.release_samples > 0 else 1)
                    self._debug_log(f"NOTE_OFF: ch={ch+1} note={note} -> RELEASE")
                return

    def set_sustain(self, midi_channel_0based: int, on: bool) -> None:
        ch = _clamp_int(midi_channel_0based, 0, 15)
        self._sustain_channel[ch] = bool(on)
        if not on:
            chip_idx = ch % len(self.chips)
            for v in self.voices[chip_idx]:
                if v.active and v.sustain_hold:
                    v.sustain_hold = False
                    v.phase = "RELEASE"
                    v.phase_rem = max(1, v.release_samples if v.release_samples > 0 else 1)

    def pitch_bend(self, midi_channel_0based: int, value14: int) -> None:
        self.voice_counters.midi_events_total += 1
        self.voice_counters.pitch_bend_events += 1

        ch = _clamp_int(midi_channel_0based, 0, 15)
        value14 = _clamp_int(value14, 0, 16383)
        self._pitchbend_channel[ch] = value14

        chip_idx = ch % len(self.chips)
        chip = self.chips[chip_idx]
        for vi, v in enumerate(self.voices[chip_idx]):
            if v.active:
                v.current_period = self._apply_pitchbend_period(ch, v.base_period, chip.sn_clock_hz)
                self._voice_apply_period(chip, vi, v.current_period)

    def _open_audio_stream(self):
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            _hard_fail("Missing audio backend. Install: pip install sounddevice numpy (and brew install portaudio).")

        def callback(outdata, frames, time_info, status):
            stereo = self.render_stereo_block(frames)
            outdata[:] = stereo

        self._sd_stream = sd.OutputStream(
            samplerate=self.cfg.sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self.cfg.block_frames,
            callback=callback,
        )
        self._sd_stream.start()

    def _close_audio_stream(self):
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

    def run_test(self, seconds: float, freq: float):
        if self.cfg.test == "beep":
            self._test_beep(seconds, freq)
        elif self.cfg.test == "noise":
            self._test_noise(seconds)
        elif self.cfg.test == "sequence":
            self._test_sequence(seconds)
        elif self.cfg.test == "chords":
            self._test_chords(seconds)
        elif self.cfg.test == "sweep":
            self._test_sweep(seconds)
        else:
            _hard_fail("Unknown test mode. Use beep|noise|sequence|chords|sweep.")

    def _test_beep(self, seconds: float, freq: float):
        chip = self.chips[0]
        period = _hz_to_sn_period(freq, chip.sn_clock_hz)
        chip.regs[0] = period
        chip.regs[1] = 2
        chip.regs[3] = 15
        chip.regs[5] = 15
        chip.regs[7] = 15

        self._open_audio_stream()
        t_end = time.time() + seconds
        try:
            while time.time() < t_end:
                self._step_envelopes_block(self.cfg.block_frames)
                time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
        finally:
            self._close_audio_stream()

    def _test_noise(self, seconds: float):
        chip = self.chips[0]
        chip.regs[7] = 2
        chip.regs[1] = 15
        chip.regs[3] = 15
        chip.regs[5] = 15

        self._open_audio_stream()
        t_end = time.time() + seconds
        try:
            while time.time() < t_end:
                self._step_envelopes_block(self.cfg.block_frames)
                time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
        finally:
            self._close_audio_stream()

    def _test_sequence(self, seconds: float):
        notes = [60, 64, 67, 72]
        step = max(0.05, seconds / max(1, len(notes)))
        self._open_audio_stream()
        try:
            for n in notes:
                self.note_on(0, n, 110)
                t_end = time.time() + step
                while time.time() < t_end:
                    self._step_envelopes_block(self.cfg.block_frames)
                    time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
                self.note_off(0, n)
                t_rel = time.time() + min(0.15, step / 2.0)
                while time.time() < t_rel:
                    self._step_envelopes_block(self.cfg.block_frames)
                    time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
        finally:
            self._close_audio_stream()

    def _test_chords(self, seconds: float):
        self._open_audio_stream()
        try:
            self.note_on(0, 60, 110)
            self.note_on(0, 64, 110)
            self.note_on(0, 67, 110)
            t_end = time.time() + seconds
            while time.time() < t_end:
                self._step_envelopes_block(self.cfg.block_frames)
                time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
            self.note_off(0, 60)
            self.note_off(0, 64)
            self.note_off(0, 67)
            t_rel = time.time() + min(0.4, seconds / 2.0)
            while time.time() < t_rel:
                self._step_envelopes_block(self.cfg.block_frames)
                time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
        finally:
            self._close_audio_stream()

    def _test_sweep(self, seconds: float, f0: float = 220.0, f1: float = 880.0):
        chip = self.chips[0]
        chip.regs[1] = 3
        chip.regs[3] = 15
        chip.regs[5] = 15
        chip.regs[7] = 15

        self._open_audio_stream()
        t0 = time.time()
        t_end = t0 + seconds
        try:
            while time.time() < t_end:
                t = (time.time() - t0) / max(0.001, seconds)
                f = f0 + (f1 - f0) * t
                chip.regs[0] = _hz_to_sn_period(f, chip.sn_clock_hz)
                self._step_envelopes_block(self.cfg.block_frames)
                time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
        finally:
            self._close_audio_stream()

    def midi_list_ports(self) -> None:
        try:
            import rtmidi  # type: ignore
        except Exception:
            _hard_fail("Missing MIDI backend. Install: pip install python-rtmidi")

        mi = rtmidi.MidiIn()
        ports = mi.get_ports()
        if not ports:
            print("MIDI PORTS: (none)")
            return
        print("MIDI PORTS:")
        for i, p in enumerate(ports):
            print(f"  [{i}] {p}")

    def run_midi(self, port_substring: str):
        try:
            import rtmidi  # type: ignore
        except Exception:
            _hard_fail("Missing MIDI backend. Install: pip install python-rtmidi")

        mi = rtmidi.MidiIn()
        ports = mi.get_ports()
        if not ports:
            _hard_fail("No MIDI input ports available.")

        sub = (port_substring or "auto").strip()
        if sub.lower() == "auto":
            port_index = 0
            chosen_name = ports[0]
        else:
            found = None
            for i, p in enumerate(ports):
                if sub.lower() in p.lower():
                    found = (i, p)
                    break
            if found is None:
                _hard_fail(f"MIDI port not found matching substring: {sub}")
            port_index, chosen_name = found

        mi.open_port(port_index)
        self._open_audio_stream()
        print(f"MIDI OPEN: {chosen_name}")
        try:
            while True:
                msg = mi.get_message()
                if msg:
                    data, dt = msg
                    if not data:
                        continue
                    status = data[0] & 0xF0
                    ch = data[0] & 0x0F

                    if status == 0x90:
                        note = data[1]
                        vel = data[2]
                        if vel == 0:
                            self.note_off(ch, note)
                        else:
                            self.note_on(ch, note, vel)
                    elif status == 0x80:
                        note = data[1]
                        self.note_off(ch, note)
                    elif status == 0xE0:
                        lsb = data[1]
                        msb = data[2]
                        val14 = (msb << 7) | lsb
                        self.pitch_bend(ch, val14)
                    elif status == 0xB0:
                        cc = data[1]
                        val = data[2]
                        if cc == 64:
                            self.set_sustain(ch, val >= 64)

                self._step_envelopes_block(self.cfg.block_frames)
                time.sleep(self.cfg.block_frames / self.cfg.sample_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self._close_audio_stream()
            try:
                mi.close_port()
            except Exception:
                pass

    def vgm_list(self, base_dir: str) -> None:
        base_dir = os.path.expanduser(base_dir)
        if not os.path.isdir(base_dir):
            _hard_fail(f"VGM base dir not found: {base_dir}")

        files = []
        for fn in os.listdir(base_dir):
            if fn.lower().endswith(".vgm"):
                files.append(fn)
        files.sort(key=lambda s: s.lower())

        self.vgm_index = {fn: os.path.join(base_dir, fn) for fn in files}

        print(f"VGM LIST: {base_dir}")
        for fn in files:
            print(fn)

    def _vgm_read_u32le(self, buf: bytes, off: int) -> int:
        if off + 4 > len(buf):
            return 0
        return int.from_bytes(buf[off:off+4], "little", signed=False)

    def _vgm_read_u16le(self, buf: bytes, off: int) -> int:
        if off + 2 > len(buf):
            return 0
        return int.from_bytes(buf[off:off+2], "little", signed=False)

    def run_vgm(self, path: str):
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            _hard_fail(f"VGM file not found: {path}")

        data = open(path, "rb").read()
        if len(data) < 0x40:
            _hard_fail("VGM file too small.")

        if data[0:4] != b"Vgm ":
            _hard_fail("Invalid VGM magic (expected 'Vgm ').")

        data_offset = self._vgm_read_u32le(data, 0x34)
        if data_offset == 0:
            data_start = 0x40
        else:
            data_start = 0x34 + data_offset

        if data_start >= len(data):
            _hard_fail("VGM data offset out of range.")

        sr_scale = float(self.cfg.sample_rate) / 44100.0

        chip0 = self.chips[0]
        for ci in range(1, len(self.chips)):
            self.chips[ci].regs[1] = 15
            self.chips[ci].regs[3] = 15
            self.chips[ci].regs[5] = 15
            self.chips[ci].regs[7] = 15

        self._open_audio_stream()

        pos = data_start
        try:
            while True:
                if pos >= len(data):
                    if self.cfg.vgm_loop:
                        self.vgm_counters.vgm_loops_total += 1
                        pos = data_start
                        continue
                    break

                cmd = data[pos]
                pos += 1
                self.vgm_counters.vgm_commands_total += 1

                if cmd == 0x50:
                    if pos >= len(data):
                        _hard_fail("Truncated PSG write.")
                    dd = data[pos]
                    pos += 1
                    chip0.write_byte(dd)
                    self.vgm_counters.vgm_psg_writes_total += 1
                    continue

                if cmd == 0x61:
                    if pos + 2 > len(data):
                        _hard_fail("Truncated wait 0x61.")
                    n = self._vgm_read_u16le(data, pos)
                    pos += 2
                    self.vgm_counters.vgm_wait_events_total += 1
                    self._vgm_wait_samples(n, sr_scale)
                    continue

                if cmd == 0x62:
                    self.vgm_counters.vgm_wait_events_total += 1
                    self._vgm_wait_samples(735, sr_scale)
                    continue

                if cmd == 0x63:
                    self.vgm_counters.vgm_wait_events_total += 1
                    self._vgm_wait_samples(882, sr_scale)
                    continue

                if 0x70 <= cmd <= 0x7F:
                    n = (cmd & 0x0F) + 1
                    self.vgm_counters.vgm_wait_events_total += 1
                    self._vgm_wait_samples(n, sr_scale)
                    continue

                if cmd == 0x66:
                    if self.cfg.vgm_loop:
                        self.vgm_counters.vgm_loops_total += 1
                        pos = data_start
                        continue
                    break

                _hard_fail(f"Unsupported VGM command 0x{cmd:02X} at offset 0x{(pos-1):08X}")

        except KeyboardInterrupt:
            pass
        finally:
            self._close_audio_stream()

    def _vgm_wait_samples(self, wait_samples_vgm: int, sr_scale: float) -> None:
        wait_engine = int(round(float(wait_samples_vgm) * sr_scale))
        if wait_engine < 0:
            wait_engine = 0

        spd = float(self.cfg.vgm_speed)
        if spd <= 0.0:
            _hard_fail("--vgm-speed must be > 0.")
        effective = int(round(wait_engine / spd))
        if wait_engine > 0 and effective < 1:
            effective = 1

        self.vgm_counters.vgm_wait_samples_total += effective

        remaining = effective
        while remaining > 0:
            blk = min(self.cfg.block_frames, remaining)
            self._step_envelopes_block(blk)
            time.sleep(blk / self.cfg.sample_rate)
            remaining -= blk


DEFAULT_VGM_BASE_DIR = "/Volumes/data1/Yandex.Disk.localized/michiele/Arduino/PCB Ontwerp/KiCAD/github/SN76489-synth-midi/src/tmp/src/"


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sn76489_emulator.py", add_help=True)

    p.add_argument("--test", choices=["beep", "noise", "sequence", "chords", "sweep"], help="Run a built-in test.")
    p.add_argument("--seconds", type=float, default=1.0, help="Duration for tests (seconds).")
    p.add_argument("--freq", type=float, default=440.0, help="Beep test frequency (Hz).")

    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--block-frames", type=int, default=512)
    p.add_argument("--chips", type=int, default=1)
    p.add_argument("--pan", choices=["left", "right", "both"], default="both")
    p.add_argument("--master-gain", type=float, default=0.25)

    p.add_argument("--attack-ms", type=float, default=5.0)
    p.add_argument("--decay-ms", type=float, default=80.0)
    p.add_argument("--sustain-vol", type=int, default=8)
    p.add_argument("--release-ms", type=float, default=120.0)

    p.add_argument("--noise-mode", choices=["white", "periodic"], default="white")
    p.add_argument("--noise-rate", choices=["div16", "div32", "div64", "tone2"], default="div32")
    p.add_argument("--noise-seed", type=lambda s: int(s, 0), default=0x4000)

    p.add_argument("--dump-regs", action="store_true")
    p.add_argument("--counters", action="store_true")
    p.add_argument("--debug", action="store_true")

    p.add_argument("--midi-list", action="store_true", help="List MIDI input ports and exit.")
    p.add_argument("--midi-in", action="store_true", help="Run in MIDI input mode (CoreMIDI via python-rtmidi).")
    p.add_argument("--midi-port", type=str, default="auto", help="MIDI port substring or 'auto'.")

    p.add_argument("--vgm-path", type=str, default=None, help="Path to .vgm file; play immediately.")
    p.add_argument("--vgm-base-dir", type=str, default=None, help="Base directory for --vgm-list.")
    p.add_argument("--vgm-loop", action="store_true")
    p.add_argument("--vgm-speed", type=float, default=1.0)
    p.add_argument("--vgm-list", action="store_true")

    return p.parse_args(argv)


def _build_cfg(ns: argparse.Namespace) -> EngineConfig:
    chips = _clamp_int(ns.chips, 1, 128)
    cfg = EngineConfig(
        sample_rate=_clamp_int(ns.sample_rate, 8000, 192000),
        block_frames=_clamp_int(ns.block_frames, 64, 8192),
        chips=chips,
        pan=str(ns.pan),
        master_gain=_clamp_float(ns.master_gain, 0.0, 2.0),

        attack_ms=_clamp_float(ns.attack_ms, 0.0, 5000.0),
        decay_ms=_clamp_float(ns.decay_ms, 0.0, 5000.0),
        sustain_vol=_clamp_int(ns.sustain_vol, 0, 15),
        release_ms=_clamp_float(ns.release_ms, 0.0, 5000.0),

        noise_mode=str(ns.noise_mode),
        noise_rate=str(ns.noise_rate),
        noise_seed=int(ns.noise_seed) & 0x7FFF,

        dump_regs=bool(ns.dump_regs),
        counters=bool(ns.counters),
        debug=bool(ns.debug),
    )
    return cfg


def _enforce_mode_exclusivity(ns: argparse.Namespace) -> Tuple[str, str]:
    has_test = ns.test is not None
    has_midi = bool(ns.midi_in)
    has_vgm = ns.vgm_path is not None
    has_vgm_list = bool(ns.vgm_list)
    has_midi_list = bool(ns.midi_list)

    if has_midi_list and (has_test or has_midi or has_vgm or has_vgm_list):
        _hard_fail("Choose --midi-list alone (do not combine with playback modes).")
    if has_vgm_list and (has_test or has_midi or has_vgm or has_midi_list):
        _hard_fail("Choose --vgm-list alone (do not combine with playback modes).")

    if has_midi and has_vgm:
        _hard_fail("Choose either --midi-in OR --vgm-path (mutually exclusive).")
    if has_test and has_vgm:
        _hard_fail("Choose either --test OR --vgm-path (mutually exclusive).")
    if has_test and has_midi:
        _hard_fail("Choose either --test OR --midi-in (mutually exclusive).")

    if has_midi_list:
        return ("midi_list", "none")
    if has_vgm_list:
        return ("vgm_list", "none")
    if has_vgm:
        return ("vgm", "none")
    if has_midi:
        return ("midi", "none")
    if has_test:
        return ("test", str(ns.test))
    return ("test", "beep")


def main(argv: Optional[List[str]] = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    mode, test = _enforce_mode_exclusivity(ns)

    cfg = _build_cfg(ns)
    cfg.mode = mode
    cfg.test = test

    cfg.vgm_path = ns.vgm_path if ns.vgm_path is not None else "none"
    cfg.vgm_base_dir = ns.vgm_base_dir if ns.vgm_base_dir is not None else DEFAULT_VGM_BASE_DIR
    cfg.vgm_loop = bool(ns.vgm_loop)
    cfg.vgm_speed = float(ns.vgm_speed)

    cfg.midi_in = bool(ns.midi_in)
    cfg.midi_port = str(ns.midi_port) if cfg.midi_in else "none"

    eng = SNEngine(cfg)

    eng.print_run_config()

    try:
        if mode == "midi_list":
            eng.midi_list_ports()
            return 0

        if mode == "vgm_list":
            eng.vgm_list(cfg.vgm_base_dir)
            return 0

        if mode == "test":
            eng.run_test(seconds=float(ns.seconds), freq=float(ns.freq))
        elif mode == "midi":
            eng.run_midi(port_substring=cfg.midi_port)
        elif mode == "vgm":
            if cfg.vgm_speed <= 0.0:
                _hard_fail("--vgm-speed must be > 0.")
            eng.run_vgm(cfg.vgm_path)
        else:
            _hard_fail(f"Unknown mode: {mode}")

    finally:
        if cfg.dump_regs:
            eng.dump_regs()
        if cfg.counters:
            eng.print_counters()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
