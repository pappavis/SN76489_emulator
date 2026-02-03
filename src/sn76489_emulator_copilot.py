#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
SN76489 Emulator v0.07 (MacOS, Python 3.12)
Single-file implementation per TS v1.6 (builds on v0.06 TS v1.5.1).

New in v0.07 (TS v1.6):
- Instrument profiles (JSON) + CLI: --instrument <name>
- Velocity curves: --vel-curve {linear|soft|hard|log}
- Voice allocation modes: --voice-alloc {first_free|round_robin}
- Sustain pedal support (CC64) via --sustain
- Dry-run mode (--dry-run): no audio and no effective SN register writes
- Debug latency reporting (--debug-latency)
- Chip batch-write API for efficiency
- State snapshot/restore: --save-state/--load-state

Kept from v0.06:
- VGM playback (path/list/loop/speed) with counters; exact subset; error format
- RUN CONFIG block (extended with new keys appended)
- Test modes and MIDI input

Notes:
- All sound originates only from SN76489 register writes (or their simulated form in dry-run).
- ASCII-only source to comply with commit policy.
"""

import argparse
import sys
import os
import math
import time
import threading
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# Optional imports guarded
try:
    import numpy as np
except Exception:
    np = None
try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    import rtmidi  # python-rtmidi
except Exception:
    rtmidi = None

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BLOCK_FRAMES = 512
MAX_CHIPS = 128

VGM_MAGIC = b'Vgm '
APP_DEFAULT_VGM_BASE_DIR = '/Volumes/data1/Yandex.Disk.localized/michiele/Arduino/PCB Ontwerp/KiCAD/github/SN76489-synth-midi/src/tmp/src/'

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def error_exit(msg: str, code: int = 2):
    print(f"ERROR: {msg}")
    sys.exit(code)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ------------------------------------------------------------
# Instrument profiles
# ------------------------------------------------------------

def load_instrument(name: str) -> Dict[str, object]:
    # Load from instruments/<name>.json if present, else defaults
    inst = {
        "name": name or "default",
        "volume_curve": "linear",
        "attack_ms": 5.0,
        "decay_ms": 60.0,
        "sustain_vol": 8,
        "release_ms": 180.0,
        "detune_cents": 0.0,
        "bend_range_semitones": 2.0,
    }
    if name and name != 'default':
        path = os.path.join('instruments', f'{name}.json')
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                inst.update(data)
            except Exception:
                pass
    return inst


# ------------------------------------------------------------
# SN76489 Chip Emulation
# ------------------------------------------------------------

class SN76489Chip:
    def __init__(self, sample_rate: int, noise_mode: str = 'white', noise_rate: str = 'div32', noise_seed: int = 0x4000, dry_run: bool = False):
        self.sample_rate = sample_rate
        self.regs = [0] * 8  # R0..R7
        self.latched = 0
        self.tone_period = [0] * 3
        self.vol = [0xF] * 4
        self.noise_ctrl = 0
        self.noise_mode = noise_mode
        self.noise_rate_sel = noise_rate
        self.lfsr = noise_seed & 0x7FFF
        if self.lfsr == 0:
            self.lfsr = 0x4000
        self.noise_seed = self.lfsr
        self.phase = [0.0, 0.0, 0.0]
        self.derived_freq = [0.0, 0.0, 0.0]
        # Counters
        self.writes_total = 0
        self.writes_latch = 0
        self.writes_data = 0
        self.renders = 0
        self.frames = 0
        # Voice meta (for dumps)
        self.voice_state = ['IDLE', 'IDLE', 'IDLE']
        self.env_phase = ['IDLE', 'IDLE', 'IDLE']
        # Mode
        self.dry_run = dry_run

    _VOL_TABLE = [
        1.0, 0.794, 0.631, 0.501, 0.398, 0.316, 0.251, 0.199,
        0.158, 0.126, 0.100, 0.079, 0.063, 0.050, 0.040, 0.000,
    ]

    def _update_tone_freqs_for_dump(self):
        for i in range(3):
            period = (self.regs[i*2] | ((self.regs[i*2+1] & 0x3) << 8)) & 0x3FF
            self.tone_period[i] = period
            self.derived_freq[i] = 0.0 if period == 0 else self.sample_rate / max(1, (32 * period))

    def _set_noise_ctrl(self, val: int):
        # 4-bit payload; b2 white/periodic; b0..b1 rate
        if not self.dry_run:
            self.regs[6] = val & 0xFF
        rate_bits = val & 0x3
        white = (val >> 2) & 0x1
        self.noise_mode = 'white' if white else 'periodic'
        self.noise_rate_sel = {0: 'div16', 1: 'div32', 2: 'div64', 3: 'tone2'}[rate_bits]

    def psg_write(self, byte: int):
        # In dry-run we still count writes, but we avoid mutating effective state except for meta we expose in dumps
        self.writes_total += 1
        if (byte & 0x80):
            self.writes_latch += 1
            chan = (byte >> 5) & 0x3
            is_vol = (byte >> 4) & 0x1
            data4 = byte & 0x0F
            if chan == 3:
                if is_vol:
                    if not self.dry_run:
                        self.regs[7] = data4
                        self.vol[3] = data4 & 0xF
                    self.latched = 7
                else:
                    if not self.dry_run:
                        self._set_noise_ctrl(data4)
                    else:
                        self._set_noise_ctrl(data4)  # keep mode/rate labels updated
                    self.latched = 6
            else:
                if is_vol:
                    idx = 1 + chan*2
                    if not self.dry_run:
                        self.regs[idx+1] = 0
                        self.regs[idx] = data4
                        self.vol[chan] = data4 & 0xF
                    self.latched = idx
                else:
                    idx = chan*2
                    if not self.dry_run:
                        prev = self.regs[idx]
                        self.regs[idx] = (prev & 0xF0) | data4
                    self.latched = idx
            self._update_tone_freqs_for_dump()
        else:
            self.writes_data += 1
            data7 = byte & 0x7F
            idx = self.latched
            if idx in (0, 2, 4):
                hi = data7 & 0x3F
                lo = self.regs[idx] & 0x0F
                period = (hi << 4) | lo
                if not self.dry_run:
                    self.regs[idx] = (self.regs[idx] & 0xF0) | (period & 0x0F)
                    self.regs[idx+1] = (self.regs[idx+1] & ~0x3) | ((period >> 8) & 0x3)
            elif idx == 6:
                if not self.dry_run:
                    self._set_noise_ctrl(data7 & 0x0F)
                else:
                    self._set_noise_ctrl(data7 & 0x0F)
            elif idx in (1, 3, 5, 7):
                if not self.dry_run:
                    self.regs[idx] = data7 & 0x0F
                    if idx == 7:
                        self.vol[3] = self.regs[idx]
                    else:
                        chan = (idx - 1)//2
                        self.vol[chan] = self.regs[idx]
            self._update_tone_freqs_for_dump()

    def batch(self, ops: List[Tuple]):
        for op in ops:
            kind = op[0]
            if kind == 'tone':
                _, channel, period = op
                lo = period & 0x0F
                hi = (period >> 4) & 0x3F
                self.psg_write(0x80 | ((channel & 0x3) << 5) | 0x0 | lo)
                self.psg_write(hi)
            elif kind == 'vol':
                _, channel, vol = op
                self.psg_write(0x90 | ((channel & 0x3) << 5) | 0x10 | (vol & 0x0F))
            elif kind == 'noise':
                _, mode, rate = op
                rate_map = {'div16':0,'div32':1,'div64':2,'tone2':3}
                white = 1 if (mode == 'white') else 0
                val = (white<<2) | rate_map.get(rate,1)
                self.psg_write(0xE0 | (val & 0x0F))

    def _noise_tick(self):
        bit0 = self.lfsr & 1
        if self.noise_mode == 'white':
            bit1 = (self.lfsr >> 1) & 1
            fb = bit0 ^ bit1
        else:
            fb = bit0
        self.lfsr = (self.lfsr >> 1) | (fb << 14)
        return self.lfsr & 1

    def render_block(self, frames: int) -> 'np.ndarray':
        if np is None:
            self.renders += 1
            self.frames += frames
            return None
        t = np.arange(frames, dtype=np.float32)
        out = np.zeros(frames, dtype=np.float32)
        for i in range(3):
            period = (self.regs[i*2] | ((self.regs[i*2+1] & 0x3) << 8)) & 0x3FF
            if period <= 1:
                wave = np.zeros_like(out)
            else:
                freq = self.sample_rate / (32.0 * period)
                if freq <= 0:
                    wave = np.zeros_like(out)
                else:
                    phase_inc = (2*math.pi*freq)/self.sample_rate
                    phase0 = self.phase[i]
                    ph = phase0 + phase_inc * t
                    wave = np.where(np.sin(ph) >= 0, 1.0, -1.0).astype(np.float32)
                    self.phase[i] = (phase0 + phase_inc*frames) % (2*math.pi)
            gain = self._VOL_TABLE[self.vol[i] & 0xF]
            out += wave * gain
        # Noise
        if self.noise_rate_sel == 'div16':
            step = 16
        elif self.noise_rate_sel == 'div32':
            step = 32
        elif self.noise_rate_sel == 'div64':
            step = 64
        else:
            period2 = (self.regs[4] | ((self.regs[5] & 0x3) << 8)) & 0x3FF
            step = max(1, period2*2)
        noise = np.zeros(frames, dtype=np.float32)
        cnt = 0
        s = 1.0
        for n in range(frames):
            if cnt == 0:
                bit = self._noise_tick()
                s = 1.0 if bit else -1.0
                cnt = step
            noise[n] = s
            cnt -= 1
        out += noise * self._VOL_TABLE[self.vol[3] & 0xF]
        out *= 0.25
        self.renders += 1
        self.frames += frames
        return out

    def dump_state(self, chip_index: int) -> str:
        self._update_tone_freqs_for_dump()
        lines = []
        lines.append(f"chip={chip_index}")
        lines.append(f"latched=R{self.latched}")
        lines.append(' '.join(f"R{i}={self.regs[i]}" for i in range(8)))
        lines.append(' '.join(f"tone{i}_hz={self.derived_freq[i]:.2f}" for i in range(3)))
        lines.append(f"noise_mode={self.noise_mode}")
        lines.append(f"noise_rate={self.noise_rate_sel}")
        lines.append(f"noise_seed=0x{self.noise_seed:04x}")
        lines.append(' '.join(f"v{i}={self.voice_state[i]}" for i in range(3)))
        lines.append(' '.join(f"env{i}={self.env_phase[i]}" for i in range(3)))
        return '\n'.join(lines)

# ------------------------------------------------------------
# Velocity Curves
# ------------------------------------------------------------

def velocity_to_vol4(vel: int, curve: str = 'linear') -> int:
    v = clamp(vel, 1, 127)
    if curve == 'soft':
        x = int(((127 - v) / 127.0) ** 0.7 * 14)
    elif curve == 'hard':
        x = int(((127 - v) / 127.0) ** 1.5 * 14)
    elif curve == 'log':
        x = int(((math.log(128) - math.log(v+1)) / math.log(128)) * 14)
    else:
        x = int((127 - v) / 127.0 * 14)
    return clamp(x, 0, 14)

# ------------------------------------------------------------
# Audio Engine
# ------------------------------------------------------------

class AudioEngine:
    def __init__(self, sample_rate: int, block_frames: int, chips: List[SN76489Chip], pan: str = 'both', master_gain: float = 0.25, dry_run: bool = False, debug_latency: bool = False):
        self.sample_rate = sample_rate
        self.block_frames = block_frames
        self.chips = chips
        self.pan = pan
        self.master_gain = master_gain
        self.stream = None
        self.dry_run = dry_run
        self.debug_latency = debug_latency
        self._lat_last_ts = 0.0
        self._lat_budget = 0

    def _mix_block(self) -> Optional['np.ndarray']:
        if np is None:
            return None
        acc = np.zeros(self.block_frames, dtype=np.float32)
        for ch in self.chips:
            blk = ch.render_block(self.block_frames)
            if blk is not None:
                acc += blk
        acc *= self.master_gain
        if self.pan == 'left':
            stereo = np.stack([acc, np.zeros_like(acc)], axis=1)
        elif self.pan == 'right':
            stereo = np.stack([np.zeros_like(acc), acc], axis=1)
        else:
            stereo = np.stack([acc, acc], axis=1)
        return stereo

    def open(self):
        if self.dry_run:
            return
        if sd is None or np is None:
            return
        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2, dtype='float32', blocksize=self.block_frames)
        self.stream.start()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
            self.stream = None

    def render_blocks(self, blocks: int):
        for _ in range(blocks):
            t0 = time.perf_counter_ns()
            stereo = self._mix_block()
            if (self.stream is not None) and (stereo is not None):
                self.stream.write(stereo)
            else:
                time.sleep(self.block_frames / max(1, self.sample_rate))
            if self.debug_latency:
                now = time.time()
                if now - self._lat_last_ts >= 1.0:
                    self._lat_last_ts = now
                    self._lat_budget = 10
                if self._lat_budget > 0:
                    dt_ms = (time.perf_counter_ns() - t0) / 1e6
                    print(f"LAT: {dt_ms:.3f} ms")
                    self._lat_budget -= 1

# ------------------------------------------------------------
# MIDI Layer
# ------------------------------------------------------------

class MidiLayer:
    def __init__(self, chips: List[SN76489Chip], vel_curve: str, voice_alloc: str, sustain_enabled: bool, debug=False, rate_limit_hz=20):
        self.chips = chips
        self.vel_curve = vel_curve
        self.voice_alloc = voice_alloc
        self.sustain_enabled = sustain_enabled
        self.debug = debug
        self.rate_limit_hz = rate_limit_hz
        self._last_debug_ts = 0.0
        self._dbg_budget = 0
        self.midi = None
        self.sustain_on = [False] * len(chips)
        self.released_while_sustain = [set() for _ in chips]
        self.last_assigned = [-1] * len(chips)
        self.counters = {
            'midi_events_total': 0,
            'note_on_total': 0,
            'note_off_total': 0,
            'pitch_bend_events': 0,
            'voices_used_total': 0,
            'note_ignored_no_voice': 0,
            'env_steps_total': 0,
        }

    def list_ports(self):
        if rtmidi is None:
            print("No MIDI backend (python-rtmidi not installed)")
            return []
        midi_in = rtmidi.MidiIn()
        ports = midi_in.get_ports()
        for i, p in enumerate(ports):
            print(f"{i}: {p}")
        return ports

    def open(self, name_substring: Optional[str] = None):
        if rtmidi is None:
            error_exit("python-rtmidi is required for --midi-in")
        self.midi = rtmidi.MidiIn()
        ports = self.midi.get_ports()
        if not ports:
            error_exit("No MIDI input ports available")
        port_index = 0
        if name_substring and name_substring.lower() != 'auto':
            found = None
            for i, p in enumerate(ports):
                if name_substring.lower() in p.lower():
                    found = i; break
            if found is None:
                error_exit(f"MIDI port not found containing: {name_substring}")
            port_index = found
        self.midi.open_port(port_index)

    def _dbg(self, msg: str):
        if not self.debug:
            return
        now = time.time()
        if now - self._last_debug_ts >= 1.0:
            self._last_debug_ts = now
            self._dbg_budget = self.rate_limit_hz
        if self._dbg_budget > 0:
            print(msg)
            self._dbg_budget -= 1

    def _assign_voice(self, chip_index: int, vol4: int, note: int):
        chip = self.chips[chip_index]
        freq = 440.0 * (2 ** ((note - 69)/12))
        period = int(clamp(round(chip.sample_rate/(32*freq)), 1, 0x3FF))
        assigned = False
        if self.voice_alloc == 'round_robin':
            self.last_assigned[chip_index] = (self.last_assigned[chip_index] + 1) % 3
            v = self.last_assigned[chip_index]
            chip.batch([('tone', v, period), ('vol', v, vol4)])
            chip.voice_state[v] = 'ATTACK'; chip.env_phase[v] = 'ATTACK'
            assigned = True
        else:
            for v in range(3):
                if chip.vol[v] >= 15:
                    chip.batch([('tone', v, period), ('vol', v, vol4)])
                    chip.voice_state[v] = 'ATTACK'; chip.env_phase[v] = 'ATTACK'
                    assigned = True
                    break
        return assigned

    def _all_notes_off_chip(self, chip):
        for v in range(3):
            chip.psg_write(0x90 | ((v & 0x3) << 5) | 0x10 | 0xF)
            chip.voice_state[v] = 'RELEASE'; chip.env_phase[v] = 'RELEASE'

    def run(self):
        if self.midi is None:
            return
        print("MIDI input active. Press Ctrl+C to stop.")
        while True:
            msg = self.midi.get_message()
            if msg:
                data, dt = msg
                self.counters['midi_events_total'] += 1
                status = data[0] & 0xF0
                channel = (data[0] & 0x0F)
                chip_index = channel % max(1, len(self.chips))
                chip = self.chips[chip_index]
                if status == 0x90 and len(data) >= 3:
                    note = data[1]; vel = data[2]
                    if vel == 0:
                        self.counters['note_off_total'] += 1
                        if self.sustain_enabled and self.sustain_on[chip_index]:
                            self.released_while_sustain[chip_index].add(note)
                            self._dbg(f"NOTE_OFF (held by sustain) ch={channel+1} note={note}")
                        else:
                            self._dbg(f"NOTE_OFF ch={channel+1} note={note}")
                            self._all_notes_off_chip(chip)
                    else:
                        self.counters['note_on_total'] += 1
                        vol4 = velocity_to_vol4(vel, self.vel_curve)
                        assigned = self._assign_voice(chip_index, vol4, note)
                        if assigned:
                            self.counters['voices_used_total'] += 1
                            self._dbg(f"NOTE_ON ch={channel+1} note={note} vol4={vol4}")
                        else:
                            self.counters['note_ignored_no_voice'] += 1
                            self._dbg(f"NOTE_IGNORED ch={channel+1} note={note} (no free voice)")
                elif status == 0x80 and len(data) >= 3:
                    note = data[1]
                    self.counters['note_off_total'] += 1
                    if self.sustain_enabled and self.sustain_on[chip_index]:
                        self.released_while_sustain[chip_index].add(note)
                        self._dbg(f"NOTE_OFF (held by sustain) ch={channel+1} note={note}")
                    else:
                        self._dbg(f"NOTE_OFF ch={channel+1} note={note}")
                        self._all_notes_off_chip(chip)
                elif status == 0xB0 and len(data) >= 3 and self.sustain_enabled:
                    cc = data[1]; val = data[2]
                    if cc == 64:
                        on = val >= 64
                        self.sustain_on[chip_index] = on
                        self._dbg(f"SUSTAIN ch={channel+1} val={val}")
                        if not on:
                            if self.released_while_sustain[chip_index]:
                                self._all_notes_off_chip(chip)
                                self.released_while_sustain[chip_index].clear()
                elif status == 0xE0 and len(data) >= 3:
                    self.counters['pitch_bend_events'] += 1
                    self._dbg(f"PITCH_BEND ch={channel+1}")
            else:
                time.sleep(0.001)

# ------------------------------------------------------------
# VGM Player (subset unchanged)
# ------------------------------------------------------------

@dataclass
class VgmCounters:
    vgm_commands_total: int = 0
    vgm_psg_writes_total: int = 0
    vgm_wait_events_total: int = 0
    vgm_wait_samples_total: int = 0
    vgm_loops_total: int = 0


class VgmPlayer:
    def __init__(self, path: str, engine: AudioEngine, chip: SN76489Chip, speed: float = 1.0, loop: bool = False, debug=False):
        self.path = path
        self.engine = engine
        self.chip = chip
        self.speed = speed
        self.loop = loop
        self.data_start = 0
        self.counters = VgmCounters()
        self.debug = debug
        self.debug_budget = 0
        self.last_debug_ts = 0.0

    def _dbg(self, msg: str):
        if not self.debug:
            return
        now = time.time()
        if now - self.last_debug_ts >= 1.0:
            self.last_debug_ts = now
            self.debug_budget = 20
        if self.debug_budget > 0:
            print(msg)
            self.debug_budget -= 1

    def _read_header(self, f):
        head = f.read(0x40)
        if len(head) < 0x40 or head[:4] != VGM_MAGIC:
            error_exit("Invalid VGM header (missing 'Vgm ' magic)")
        data_ofs = int.from_bytes(head[0x34:0x38], 'little', signed=False)
        self.data_start = 0x40 if data_ofs == 0 else 0x34 + data_ofs

    def play(self):
        if not os.path.isfile(self.path):
            error_exit(f"VGM file not found: {self.path}")
        if self.speed <= 0.0:
            error_exit("--vgm-speed must be > 0")
        with open(self.path, 'rb') as f:
            self._read_header(f)
            f.seek(self.data_start)
            while True:
                b = f.read(1)
                if not b:
                    break
                cmd = b[0]
                self.counters.vgm_commands_total += 1
                if cmd == 0x50:
                    dd = f.read(1)
                    if not dd: break
                    self.chip.psg_write(dd[0])
                    self.counters.vgm_psg_writes_total += 1
                elif cmd == 0x61:
                    nn = f.read(2)
                    if len(nn) < 2: break
                    samples = int.from_bytes(nn, 'little', signed=False)
                    self._wait_samples(samples)
                elif cmd == 0x62:
                    self._wait_samples(735)
                elif cmd == 0x63:
                    self._wait_samples(882)
                elif 0x70 <= cmd <= 0x7F:
                    self._wait_samples((cmd & 0x0F) + 1)
                elif cmd == 0x66:
                    if self.loop:
                        self.counters.vgm_loops_total += 1
                        f.seek(self.data_start)
                        continue
                    else:
                        break
                else:
                    off = f.tell() - 1
                    print(f"ERROR: Unsupported VGM command 0x{cmd:02X} at offset 0x{off:08X}")
                    sys.exit(2)

    def _wait_samples(self, samples_vgm: int):
        self.counters.vgm_wait_events_total += 1
        sr = self.engine.sample_rate
        scaled = round(samples_vgm * (sr / 44100.0))
        effective = max(1, round(scaled / self.speed)) if samples_vgm > 0 else 0
        self.counters.vgm_wait_samples_total += effective
        if effective <= 0:
            return
        blocks = effective // self.engine.block_frames
        tail = effective % self.engine.block_frames
        if blocks:
            self.engine.render_blocks(blocks)
        if tail:
            saved = self.engine.block_frames
            self.engine.block_frames = tail
            self.engine.render_blocks(1)
            self.engine.block_frames = saved

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def print_run_config(cfg: Dict[str, str]):
    keys = [
        'mode','test','sample_rate','block_frames','chips','pan','master_gain',
        'attack_ms','decay_ms','sustain_vol','release_ms',
        'noise_mode','noise_rate','noise_seed',
        'midi_in','midi_port',
        'vgm_path','vgm_base_dir','vgm_loop','vgm_speed',
        'dump_regs','counters','debug',
        # v1.6 additions
        'instrument','voice_alloc','vel_curve','sustain','dry_run','debug_latency','save_state','load_state'
    ]
    print('RUN CONFIG:')
    for k in keys:
        print(f"{k}={cfg[k]}")


def list_vgm_dir(base_dir: str) -> Dict[str, str]:
    if not os.path.isdir(base_dir):
        return {}
    files = [f for f in os.listdir(base_dir) if f.lower().endswith('.vgm')]
    files.sort(key=lambda s: s.lower())
    return {fn: os.path.abspath(os.path.join(base_dir, fn)) for fn in files}

# ------------------------------------------------------------
# State save/load
# ------------------------------------------------------------

def save_state(filepath: str, chips: List[SN76489Chip], vgm: Optional[VgmCounters]):
    state = {
        'chips': [
            {
                'regs': c.regs,
                'latched': c.latched,
                'lfsr': c.lfsr,
                'noise_mode': c.noise_mode,
                'noise_rate': c.noise_rate_sel,
                'voice_state': c.voice_state,
                'env_phase': c.env_phase,
                'writes_total': c.writes_total,
                'writes_latch': c.writes_latch,
                'writes_data': c.writes_data,
                'renders': c.renders,
                'frames': c.frames,
            } for c in chips
        ]
    }
    if vgm is not None:
        state['vgm'] = vgm.__dict__
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, sort_keys=True)


def load_state(filepath: str, chips: List[SN76489Chip]) -> Optional[VgmCounters]:
    if not os.path.isfile(filepath):
        error_exit(f"State file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        state = json.load(f)
    for i, c in enumerate(chips):
        if i >= len(state['chips']):
            break
        s = state['chips'][i]
        c.regs = s['regs']
        c.latched = s['latched']
        c.lfsr = s['lfsr']
        c.noise_mode = s['noise_mode']
        c.noise_rate_sel = s['noise_rate']
        c.voice_state = s['voice_state']
        c.env_phase = s['env_phase']
        c.writes_total = s['writes_total']
        c.writes_latch = s['writes_latch']
        c.writes_data = s['writes_data']
        c.renders = s['renders']
        c.frames = s['frames']
    if 'vgm' in state:
        vc = VgmCounters()
        vc.__dict__.update(state['vgm'])
        return vc
    return None

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def run_tests(test_mode: str, engine: AudioEngine, chips: List[SN76489Chip], seconds: float = 2.0):
    if np is None or sd is None:
        print("(Audio backends not available or dry-run; running silent timing simulation)")
    ch0 = chips[0]
    if test_mode == 'beep':
        freq = 440.0
        period = int(max(1, round(ch0.sample_rate/(32*freq))))
        ch0.batch([('tone',0,period), ('vol',0,2)])
    elif test_mode == 'noise':
        ch0.batch([('noise', ch0.noise_mode, ch0.noise_rate_sel), ('vol',3,2)])
    elif test_mode == 'sequence':
        notes = [60, 64, 67]
        for v, note in enumerate(notes):
            freq = 440.0 * (2 ** ((note - 69)/12))
            period = int(max(1, round(ch0.sample_rate/(32*freq))))
            ch0.batch([('tone',v,period), ('vol',v,4)])
    elif test_mode == 'chords':
        notes = [60, 64, 67]
        for v, note in enumerate(notes):
            freq = 440.0 * (2 ** ((note - 69)/12))
            period = int(max(1, round(ch0.sample_rate/(32*freq))))
            ch0.batch([('tone',v,period), ('vol',v,3)])
    elif test_mode == 'sweep':
        t0 = time.time()
        while time.time() - t0 < seconds:
            frac = (time.time() - t0) / seconds
            freq = 2000.0 * (1.0 - frac) + 200.0 * frac
            period = int(max(1, round(ch0.sample_rate/(32*freq))))
            ch0.batch([('tone',0,period), ('vol',0,4)])
            engine.render_blocks(1)
        return
    else:
        print(f"Unknown test mode: {test_mode}")
        return
    total_frames = int(engine.sample_rate * seconds)
    blocks = total_frames // engine.block_frames
    tail = total_frames % engine.block_frames
    if blocks:
        engine.render_blocks(blocks)
    if tail:
        saved = engine.block_frames
        engine.block_frames = tail
        engine.render_blocks(1)
        engine.block_frames = saved

# ------------------------------------------------------------
# CLI / Main
# ------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description='SN76489 Emulator v0.07')
    # Core audio
    p.add_argument('--sample-rate', type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument('--block-frames', type=int, default=DEFAULT_BLOCK_FRAMES)
    p.add_argument('--chips', type=int, default=1)
    p.add_argument('--pan', choices=['left','right','both'], default='both')
    p.add_argument('--master-gain', type=float, default=0.25)
    # Envelope (legacy CLI)
    p.add_argument('--attack-ms', type=float, default=5.0)
    p.add_argument('--decay-ms', type=float, default=60.0)
    p.add_argument('--sustain-vol', type=int, default=8)
    p.add_argument('--release-ms', type=float, default=180.0)
    # Noise
    p.add_argument('--noise-mode', choices=['white','periodic'], default='white')
    p.add_argument('--noise-rate', choices=['div16','div32','div64','tone2'], default='div32')
    p.add_argument('--noise-seed', type=lambda s: int(s, 16) if s.lower().startswith('0x') else int(s), default=0x4000)
    # Debug/inspect
    p.add_argument('--dump-regs', action='store_true')
    p.add_argument('--counters', action='store_true')
    p.add_argument('--debug', action='store_true')
    # Tests
    p.add_argument('--test', choices=['beep','noise','sequence','chords','sweep'])
    p.add_argument('--seconds', type=float, default=2.0)
    # MIDI
    p.add_argument('--midi-list', action='store_true')
    p.add_argument('--midi-in', action='store_true')
    p.add_argument('--midi-port', default='none')
    # VGM
    p.add_argument('--vgm-path', default=None)
    p.add_argument('--vgm-base-dir', default=None)
    p.add_argument('--vgm-loop', action='store_true')
    p.add_argument('--vgm-speed', type=float, default=1.0)
    p.add_argument('--vgm-list', action='store_true')
    # TS v1.6 additions
    p.add_argument('--instrument', default='default')
    p.add_argument('--voice-alloc', choices=['first_free','round_robin'], default='first_free')
    p.add_argument('--vel-curve', choices=['linear','soft','hard','log'], default='linear')
    p.add_argument('--sustain', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--debug-latency', action='store_true')
    p.add_argument('--save-state', dest='save_state_path', default=None)
    p.add_argument('--load-state', dest='load_state_path', default=None)

    args = p.parse_args(argv)

    # Exclusivity
    modes = {'midi': bool(args.midi_in), 'vgm': bool(args.vgm_path), 'test': bool(args.test)}
    if modes['midi'] and modes['vgm']:
        error_exit('Choose either --midi-in OR --vgm-path (mutually exclusive).')
    if modes['midi'] and modes['test']:
        error_exit('Choose either --test OR --midi-in (mutually exclusive).')
    if modes['vgm'] and modes['test']:
        error_exit('Choose either --test OR --vgm-path (mutually exclusive).')

    vgm_base_dir = args.vgm_base_dir if args.vgm_base_dir else APP_DEFAULT_VGM_BASE_DIR

    if args.midi_list:
        if rtmidi is None:
            print('No MIDI backend (python-rtmidi not installed)')
        else:
            midi = rtmidi.MidiIn()
            for i, pstr in enumerate(midi.get_ports()):
                print(f"{i}: {pstr}")
        sys.exit(0)

    inst = load_instrument(args.instrument)

    mode = 'test' if args.test else ('midi' if args.midi_in else ('vgm_list' if args.vgm_list else ('vgm' if args.vgm_path else 'test')))
    test_name = args.test if args.test else 'none'

    cfg = {
        'mode': mode,
        'test': test_name,
        'sample_rate': str(args.sample_rate),
        'block_frames': str(args.block_frames),
        'chips': str(args.chips),
        'pan': args.pan,
        'master_gain': f"{args.master_gain}",
        'attack_ms': f"{inst.get('attack_ms', args.attack_ms)}",
        'decay_ms': f"{inst.get('decay_ms', args.decay_ms)}",
        'sustain_vol': str(inst.get('sustain_vol', args.sustain_vol)),
        'release_ms': f"{inst.get('release_ms', args.release_ms)}",
        'noise_mode': args.noise_mode,
        'noise_rate': args.noise_rate,
        'noise_seed': f"0x{args.noise_seed:04x}",
        'midi_in': '1' if args.midi_in else '0',
        'midi_port': ('auto' if (args.midi_in and args.midi_port in ('none','auto')) else (args.midi_in and args.midi_port or 'none')),
        'vgm_path': args.vgm_path if args.vgm_path else 'none',
        'vgm_base_dir': vgm_base_dir if (args.vgm_list or args.vgm_path) else 'none',
        'vgm_loop': '1' if args.vgm_loop else '0',
        'vgm_speed': f"{args.vgm_speed}",
        'dump_regs': '1' if args.dump_regs else '0',
        'counters': '1' if args.counters else '0',
        'debug': '1' if args.debug else '0',
        'instrument': inst.get('name','default'),
        'voice_alloc': args.voice_alloc,
        'vel_curve': args.vel_curve,
        'sustain': '1' if args.sustain else '0',
        'dry_run': '1' if args.dry_run else '0',
        'debug_latency': '1' if args.debug_latency else '0',
        'save_state': args.save_state_path or 'none',
        'load_state': args.load_state_path or 'none',
    }

    print_run_config(cfg)

    chips: List[SN76489Chip] = []
    N = clamp(args.chips, 1, MAX_CHIPS)
    for i in range(N):
        chips.append(SN76489Chip(sample_rate=args.sample_rate, noise_mode=args.noise_mode, noise_rate=args.noise_rate, noise_seed=args.noise_seed, dry_run=args.dry_run))
        if i > 0:
            chips[i].vol = [15,15,15,15]

    # Apply instrument params to legacy envelope config (shown in dumps)
    args.attack_ms = inst.get('attack_ms', args.attack_ms)
    args.decay_ms = inst.get('decay_ms', args.decay_ms)
    args.sustain_vol = inst.get('sustain_vol', args.sustain_vol)
    args.release_ms = inst.get('release_ms', args.release_ms)

    engine = AudioEngine(args.sample_rate, args.block_frames, chips, args.pan, args.master_gain, dry_run=args.dry_run, debug_latency=args.debug_latency)
    engine.open()

    loaded_vgm_counters = None
    if args.load_state_path:
        loaded_vgm_counters = load_state(args.load_state_path, chips)

    try:
        if args.vgm_list:
            print(f"VGM LIST: {vgm_base_dir}")
            for fn in list_vgm_dir(vgm_base_dir).keys():
                print(fn)
            sys.exit(0)

        if args.vgm_path:
            if not os.path.isfile(args.vgm_path):
                error_exit(f"VGM file not found: {args.vgm_path}")
            player = VgmPlayer(args.vgm_path, engine, chips[0], speed=args.vgm_speed, loop=args.vgm_loop, debug=args.debug)
            player.play()
            if args.counters:
                total = {
                    'writes_total': sum(c.writes_total for c in chips),
                    'latch': sum(c.writes_latch for c in chips),
                    'data': sum(c.writes_data for c in chips),
                    'renders': sum(c.renders for c in chips),
                    'frames': sum(c.frames for c in chips),
                    'midi_events_total': 0,
                    'note_on_total': 0,
                    'note_off_total': 0,
                    'pitch_bend_events': 0,
                    'voices_used_total': 0,
                    'note_ignored_no_voice': 0,
                    'env_steps_total': 0,
                    'vgm_commands_total': player.counters.vgm_commands_total,
                    'vgm_psg_writes_total': player.counters.vgm_psg_writes_total,
                    'vgm_wait_events_total': player.counters.vgm_wait_events_total,
                    'vgm_wait_samples_total': player.counters.vgm_wait_samples_total,
                    'vgm_loops_total': player.counters.vgm_loops_total,
                }
                print(json.dumps(total, indent=2, sort_keys=True))
            if args.dump_regs:
                for i, c in enumerate(chips):
                    print(c.dump_state(i))
            if args.save_state_path:
                save_state(args.save_state_path, chips, player.counters)
            sys.exit(0)

        if args.midi_in:
            midi = MidiLayer(chips, vel_curve=args.vel_curve, voice_alloc=args.voice_alloc, sustain_enabled=args.sustain, debug=args.debug)
            midi.open(None if args.midi_port in ('none','auto') else args.midi_port)
            try:
                midi.run()
            except KeyboardInterrupt:
                if args.counters:
                    agg = midi.counters.copy()
                    agg.update({
                        'writes_total': sum(c.writes_total for c in chips),
                        'latch': sum(c.writes_latch for c in chips),
                        'data': sum(c.writes_data for c in chips),
                        'renders': sum(c.renders for c in chips),
                        'frames': sum(c.frames for c in chips),
                        'vgm_commands_total': 0,
                        'vgm_psg_writes_total': 0,
                        'vgm_wait_events_total': 0,
                        'vgm_wait_samples_total': 0,
                        'vgm_loops_total': 0,
                    })
                    print(json.dumps(agg, indent=2, sort_keys=True))
                if args.dump_regs:
                    for i, c in enumerate(chips):
                        print(c.dump_state(i))
                if args.save_state_path:
                    save_state(args.save_state_path, chips, None)
                sys.exit(0)

        if args.test:
            run_tests(args.test, engine, chips, seconds=args.seconds)
            if args.counters:
                total = {
                    'writes_total': sum(c.writes_total for c in chips),
                    'latch': sum(c.writes_latch for c in chips),
                    'data': sum(c.writes_data for c in chips),
                    'renders': sum(c.renders for c in chips),
                    'frames': sum(c.frames for c in chips),
                    'midi_events_total': 0,
                    'note_on_total': 0,
                    'note_off_total': 0,
                    'pitch_bend_events': 0,
                    'voices_used_total': 0,
                    'note_ignored_no_voice': 0,
                    'env_steps_total': 0,
                    'vgm_commands_total': 0,
                    'vgm_psg_writes_total': 0,
                    'vgm_wait_events_total': 0,
                    'vgm_wait_samples_total': 0,
                    'vgm_loops_total': 0,
                }
                print(json.dumps(total, indent=2, sort_keys=True))
            if args.dump_regs:
                for i, c in enumerate(chips):
                    print(c.dump_state(i))
            if args.save_state_path:
                save_state(args.save_state_path, chips, None)
            sys.exit(0)

        p.print_help()
        sys.exit(0)

    except KeyboardInterrupt:
        if args.counters:
            agg = {
                'writes_total': sum(c.writes_total for c in chips),
                'latch': sum(c.writes_latch for c in chips),
                'data': sum(c.writes_data for c in chips),
                'renders': sum(c.renders for c in chips),
                'frames': sum(c.frames for c in chips),
                'midi_events_total': 0,
                'note_on_total': 0,
                'note_off_total': 0,
                'pitch_bend_events': 0,
                'voices_used_total': 0,
                'note_ignored_no_voice': 0,
                'env_steps_total': 0,
                'vgm_commands_total': 0,
                'vgm_psg_writes_total': 0,
                'vgm_wait_events_total': 0,
                'vgm_wait_samples_total': 0,
                'vgm_loops_total': 0,
            }
            print(json.dumps(agg, indent=2, sort_keys=True))
        if args.dump_regs:
            for i, c in enumerate(chips):
                print(c.dump_state(i))
        if args.save_state_path:
            save_state(args.save_state_path, chips, None)
        sys.exit(0)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
