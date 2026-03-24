# SN76489 Emulator (Python, macOS) — v0.07

Single-file SN76489 PSG emulator for macOS with:
- 3 tone + 1 noise channel per chip
- multi-chip engine (1..128)
- stereo routing + per-voice pan
- ADSR-lite via register-style volume steps
- MIDI input (CoreMIDI via python-rtmidi)
- VGM playback subset
- deterministic debug/counter output

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy sounddevice
pip install python-rtmidi
brew install portaudio
```

## Quick start
```bash
python sn76489_emulator.py --test beep
python sn76489_emulator.py --test sequence --velocity-curve log --debug
python sn76489_emulator.py --test chords --pan both --voice-pan spread --dump-regs
python sn76489_emulator.py --midi-list
python sn76489_emulator.py --vgm-list --vgm-base-dir "/path/to/vgms"
```

## Musical engine (v0.07)
- deterministic voice allocation + stealing
- velocity curves: linear / log / exp
- sustain pedal (CC64)
- pitch bend (+/-2 semitone)
- per-voice stereo pan

## VGM playback
Supported commands:
- 0x50 dd
- 0x61 nn nn
- 0x62
- 0x63
- 0x70..0x7F
- 0x66

## Commit
```bash
git commit -m "v0.07: fix split-render mixing, note cleanup and counters; keep MIDI/VGM/tests stable"
```
