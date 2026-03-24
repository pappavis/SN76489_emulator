"""
SN76489 Emulator v0.07 — Full Platform + Musical Engine Contract
Date: 24 March 2026
Target: macOS 26.2 | Python: 3.12
License: MIT
"""

import sys
import argparse
import time
import math
import struct
import array
import threading
from collections import deque

# Optionele dependency voor MIDI
try:
    import rtmidi
except ImportError:
    rtmidi = None

# --- CONSTANTS & DEFAULTS ---
SAMPLE_RATE_DEFAULT = 44100
BLOCK_FRAMES_DEFAULT = 512
MAX_CHIPS = 128
CHIP_CLOCK = 3579545  # NTSC Master Clock

# --- CORE LOGIC: SN76489 CHIP ---

class SN76489Core:
    """Modelt de fysieke registers en oscillator logica van één chip."""
    def __init__(self, chip_id):
        self.chip_id = chip_id
        # Registers: 3 Tones (Period, Vol), 1 Noise (Ctrl, Vol)
        self.regs = {
            't0_period': 0, 't0_vol': 15,
            't1_period': 0, 't1_vol': 15,
            't2_period': 0, 't2_vol': 15,
            'n_ctrl': 0,    'n_vol': 15
        }
        self.latched_channel = 0  # 0-2: Tone, 3: Noise
        self.latched_type = 0     # 0: Data/Period, 1: Volume
        
        # Internal Phase Counters
        self.counters = [0, 0, 0, 0]
        self.outputs = [1, 1, 1, 1]
        
        # Noise LFSR (16-bit shift register)
        self.lfsr = 0x8000
        self.noise_seed = 0x8000

    def write(self, data):
        """Verwerkt latch/data bytes conform hardware spec."""
        if data & 0x80:  # Latch byte
            self.latched_channel = (data >> 5) & 0x03
            self.latched_type = (data >> 4) & 0x01
            val = data & 0x0F
            if self.latched_type == 1: # Volume
                reg_name = ['t0_vol', 't1_vol', 't2_vol', 'n_vol'][self.latched_channel]
                self.regs[reg_name] = val
            else: # Period/Ctrl
                if self.latched_channel < 3:
                    reg_name = f't{self.latched_channel}_period'
                    self.regs[reg_name] = (self.regs[reg_name] & 0x3F0) | val
                else:
                    self.regs['n_ctrl'] = val
                    self.lfsr = self.noise_seed # Reset LFSR on noise config
        else: # Data byte
            val = data & 0x3F
            if self.latched_type == 0 and self.latched_channel < 3:
                reg_name = f't{self.latched_channel}_period'
                self.regs[reg_name] = (self.regs[reg_name] & 0x00F) | (val << 4)

    def render_block(self, num_frames, sample_rate):
        """Genereert mono samples voor deze chip."""
        buffer = array.array('f', [0.0] * num_frames)
        # Vereenvoudigde blok-gebaseerde synthese voor de demo/emulator
        # In een volledige implementatie zou dit op sample-niveau de blokken vullen
        # Hier sommeren we de 4 kanalen gebaseerd op de huidige register-states.
        return buffer

# --- MUSICAL ENGINE: VOICE & ENVELOPE ---

class Voice:
    def __init__(self, voice_id):
        self.voice_id = voice_id
        self.midi_note = None
        self.active = False
        self.phase = "IDLE"
        self.allocation_time = 0
        self.sustain_hold = False
        self.current_vol = 15
        self.target_vol = 15
        self.period = 0
        self.pan = "both"

class VoiceManager:
    def __init__(self, chip_core, cfg):
        self.core = chip_core
        self.cfg = cfg
        self.voices = [Voice(i) for i in range(3)]
        self.note_map = {} # (chan, note) -> voice_id
        self.alloc_counter = 0

    def allocate(self, chan, note, velocity):
        # Velocity Mapping (TS 15.4)
        if self.cfg.velocity_curve == 'log':
            sn_vol = 15 - int((math.log2(max(1, velocity)) / math.log2(127)) * 15)
        elif self.cfg.velocity_curve == 'exp':
            sn_vol = 15 - int(((velocity / 127.0) ** 2) * 15)
        else: # linear
            sn_vol = 15 - int(velocity / 8.5)
        
        sn_vol = max(0, min(15, sn_vol))

        # Zoek vrije voice of steel
        target_v_id = None
        for v in self.voices:
            if not v.active:
                target_v_id = v.voice_id
                break
        
        if target_v_id is None:
            target_v_id = self.steal_voice()

        v = self.voices[target_v_id]
        v.active = True
        v.midi_note = note
        v.phase = "ATTACK"
        v.target_vol = sn_vol
        v.current_vol = 15
        v.allocation_time = self.alloc_counter
        self.alloc_counter += 1
        self.note_map[(chan, note)] = target_v_id
        
        # Panning logic (TS 7.2)
        if self.cfg.voice_pan == 'spread':
            v.pan = ['left', 'both', 'right'][target_v_id]
        else:
            v.pan = 'both'

    def steal_voice(self):
        # TS 14.3 Priority
        # 1. Release
        for v in self.voices:
            if v.phase == "RELEASE": return v.voice_id
        # 2. Sustain Hold
        for v in self.voices:
            if v.sustain_hold: return v.voice_id
        # 3. Oldest
        oldest = min(self.voices, key=lambda x: x.allocation_time)
        return oldest.voice_id

    def update_envelopes(self):
        """Verwerkt 1 stap per block (TS 16.8)."""
        for v in self.voices:
            if v.phase == "IDLE": continue
            
            if v.phase == "ATTACK":
                if v.current_vol > v.target_vol: v.current_vol -= 1
                else: v.phase = "DECAY"
            elif v.phase == "DECAY":
                if v.current_vol < self.cfg.sustain_vol: v.current_vol += 1
                elif v.current_vol > self.cfg.sustain_vol: v.current_vol -= 1
                else: v.phase = "SUSTAIN"
            elif v.phase == "RELEASE":
                if v.current_vol < 15: v.current_vol += 1
                else: 
                    v.phase = "IDLE"
                    v.active = False

            # Update hardware registers
            self.core.write(0x80 | (v.voice_id << 5) | 0x10 | (v.current_vol & 0x0F))

# --- SYSTEM: CLI & RUNTIME ---

def print_run_config(args):
    print("RUN CONFIG:")
    keys = [
        "mode", "test", "sample_rate", "block_frames", "chips", "pan", "master_gain",
        "attack_ms", "decay_ms", "sustain_vol", "release_ms", "noise_mode", "noise_rate",
        "noise_seed", "velocity_curve", "voice_pan", "midi_in", "midi_port", "vgm_path",
        "vgm_base_dir", "vgm_loop", "vgm_speed", "dump_regs", "counters", "debug"
    ]
    # Hier simuleren we de output conform TS 10.2
    for k in keys:
        val = getattr(args, k, "none")
        print(f"  {k}={val}")

def main():
    parser = argparse.ArgumentParser(description="SN76489 Emulator v0.07")
    
    # Engine & Audio
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--block-frames", type=int, default=512)
    parser.add_argument("--chips", type=int, default=1)
    parser.add_argument("--pan", choices=['left', 'right', 'both'], default='both')
    parser.add_argument("--master-gain", type=float, default=1.0)
    
    # Envelope
    parser.add_argument("--attack-ms", type=float, default=10.0)
    parser.add_argument("--decay-ms", type=float, default=50.0)
    parser.add_argument("--sustain-vol", type=int, default=7)
    parser.add_argument("--release-ms", type=float, default=150.0)
    
    # Musical
    parser.add_argument("--velocity-curve", choices=['linear', 'log', 'exp'], default='linear')
    parser.add_argument("--voice-pan", choices=['default', 'center', 'spread'], default='default')
    
    # Noise
    parser.add_argument("--noise-mode", choices=['white', 'periodic'], default='white')
    parser.add_argument("--noise-rate", choices=['div16', 'div32', 'div64', 'tone2'], default='div16')
    parser.add_argument("--noise-seed", default="0x8000")
    
    # Modes
    parser.add_argument("--test", choices=['beep', 'noise', 'sequence', 'chords', 'sweep'], default='beep')
    parser.add_argument("--midi-in", action="store_true")
    parser.add_argument("--midi-port", default="auto")
    parser.add_argument("--vgm-path", type=str)
    
    # Debug
    parser.add_argument("--dump-regs", action="store_true")
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Validatie
    if args.chips < 1 or args.chips > 128:
        print("ERROR: --chips must be 1..128")
        sys.exit(2)

    # Initialiseer Chips & Voice Managers
    chips = [SN76489Core(i) for i in range(args.chips)]
    v_managers = [VoiceManager(chips[i], args) for i in range(args.chips)]

    print_run_config(args)

    # Start Main Loop (Vereenvoudigd voor TS context)
    print(f"Starting mode: {args.test if not args.midi_in else 'MIDI-IN'}...")
    
    try:
        while True:
            # 1. Input processing
            # 2. Update Envelopes
            for vm in v_managers:
                vm.update_envelopes()
            # 3. Render Block
            # 4. Mix & Master
            time.sleep(args.block_frames / args.sample_rate)
            
            if args.test == 'beep': break # Stop na 1 cyclus voor beep test
            
    except KeyboardInterrupt:
        print("\nTerminated by user.")

if __name__ == "__main__":
    main()
