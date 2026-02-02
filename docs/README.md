# SN76489 Emulator (Python, MacOS) — v0.06

**Target:** MacOS 26.2 • Python 3.12  
**Single-file artifact:** `sn76489_emulator.py`  
**Core rule:** All sound is produced via SN76489-style register writes (no direct DSP shortcuts).

## Features (v0.06)
- SN76489 PSG core emulation: 3 tone + 1 noise
- Multi-chip engine: `--chips 1..128`
- Stereo routing: `--pan left|right|both`
- Mixer + `--master-gain`
- Deterministic noise: `--noise-rate`, `--noise-seed`
- ADSR-lite envelope via volume writes (used by MIDI/sequence)
- MIDI input on MacOS via CoreMIDI (python-rtmidi)
- VGM playback subset:
  - PSG write: `0x50 dd`
  - waits: `0x61`, `0x62`, `0x63`, `0x70..0x7F`
  - end: `0x66`
- Debug contracts:
  - RUN CONFIG echo (stable key order)
  - `--dump-regs` golden-friendly output
  - `--counters` includes VGM metrics
  - `--debug` rate-limited trace

## Install
### Python deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install sounddevice numpy
pip install python-rtmidi
