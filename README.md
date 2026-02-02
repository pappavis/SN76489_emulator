# SN76489 Emulator (Python, macOS) — v0.06

**One-liner (manager-level):** A small, test-driven audio engine that emulates the classic SN76489 sound chip on macOS and plays chip-music (VGM) and live MIDI with deterministic, repeatable behavior.

---

## What is this?

This project is a **software emulator** of the **SN76489 PSG (Programmable Sound Generator)** — the classic 8-bit era sound chip found in multiple retro systems.

It runs as a **single Python file** (`sn76489_emulator.py`) and outputs audio directly to your Mac speakers.

**Key idea:** all audio is produced via **SN76489-style register writes** (no “shortcut” oscillators). This makes the emulator useful as both:
- a playable chip-synth engine (tests + MIDI input), and
- a reference implementation for later hardware bridging (real SN76489 chips).

---

## Who is this for?

- **Retro audio / chiptune builders** who want to play or validate SN76489 behavior quickly.
- **Developers** who want a clean baseline to compare against other implementations (including Copilot-generated code).
- **Hardware hobbyists** building SN76489 synth hardware and needing a “golden” reference engine.
- **DAW users** (Logic / Ableton on macOS) who want to drive SN76489-style sound via MIDI input.

---

## Features (v0.06)

### SN76489 core + engine
- SN76489 emulation: **3 tone channels + 1 noise channel**
- Noise modes: **white / periodic**
- Noise rate selection: `div16 | div32 | div64 | tone2`
- Deterministic noise seed support for regression checks
- Mixer with `--master-gain`
- Stereo routing: `--pan left|right|both`
- Multi-chip engine: `--chips 1..128`

### Runtime modes (CLI-first)
- Tests:
  - `--test beep`
  - `--test noise`
  - `--test sequence`
  - `--test chords`
  - `--test sweep`
- Debug / inspection:
  - `--dump-regs` (stable “golden output” format)
  - `--counters` (chip + voice + VGM metrics)
  - `--debug` (rate-limited trace)
- **RUN CONFIG** parameter echo block (always printed before playback modes)

### MIDI (macOS)
- Live MIDI input via CoreMIDI (through `python-rtmidi`)
- MIDI channel → chip mapping (wraps by modulo chips)
- Supported events:
  - Note On / Note Off
  - Velocity → 4-bit volume mapping
  - Pitch Bend (±2 semitone)
  - CC64 sustain (basic)

### VGM playback (macOS)
Plays `.vgm` files using a strict supported command subset:
- PSG write: `0x50 dd`
- Wait: `0x61 nn nn`, `0x62`, `0x63`, `0x70..0x7F`
- End: `0x66`

VGM CLI:
- `--vgm-list`
- `--vgm-path <file>`
- `--vgm-loop`
- `--vgm-speed <factor>`

---

## Installation (macOS 26.2 / Python 3.12)

### 1) Create venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2) Audio dependencies
```bash
pip install numpy sounddevice python-rtmidi
```

MacOS if sounddevice fails due to PortAudio:
```bash
brew install portaudio
```

# Quick start
## Beep
```bash
python sn76489_emulator.py --test beep
```

## 2) Multi-chip + panning sanity
```bash
python sn76489_emulator.py --test beep --chips 2 --pan left
python sn76489_emulator.py --test beep --chips 2 --pan right
```

## 3) Noise determinism check (run twice, should sound identical)
```bash
python sn76489_emulator.py --test noise --noise-mode white --noise-rate div32 --noise-seed 0x4000
python sn76489_emulator.py --test noise --noise-mode white --noise-rate div32 --noise-seed 0x4000
```

## 4) Play VGM

### List directory:
```bash
python sn76489_emulator.py --vgm-list --vgm-base-dir "/path/to/vgm/dir"
```

### Play once:
```bash
python sn76489_emulator.py --vgm-path "/path/to/song.vgm"
```

### Loop + speed:
```bash
python sn76489_emulator.py --vgm-path "/path/to/song.vgm" --vgm-loop --vgm-speed 1.0
```

### With counters + registers:
```bash
python sn76489_emulator.py --vgm-path "/path/to/song.vgm" --counters --dump-regs
```

### 5) MIDI (optional)
List ports:
```bash
python sn76489_emulator.py --midi-list
```


Auto-open port 0:
```bash
python sn76489_emulator.py --midi-in --chips 2 --pan both --master-gain 0.25
```

Choose a port by substring:
```bash
python sn76489_emulator.py --midi-in --midi-port "IAC" --chips 1
```

## Operational contract (important)

### Mutual exclusivity (hard fail)

These modes are mutually exclusive:
	•	--test ...
	•	--midi-in
	•	--vgm-path ...
	•	--vgm-list
	•	--midi-list

If you combine exclusive modes, the program exits with:
	•	ERROR: ...
	•	exit code 2

RUN CONFIG echo

Before playback starts, the emulator prints a stable multi-line RUN CONFIG: block that echoes effective parameters.
This provides “visual verification” and supports regression snapshots.

# How this was built (development cycle)

This project was built with a deliberate iterative loop:
##	1.	Idea / need
	•	Fast SN76489 emulation with real register behavior
	•	Repeatable audio tests on macOS

##	2.	Functional Specification (FS)
	•	What must exist (modes, outputs, acceptance criteria)

##	3.	Technical Specification (TS)
	•	Exact contracts (CLI, debug formats, timing model, command subsets)

##	4.	Test harness
	•	Beep/noise/sequence/chords/sweep tests + counters + dumps
    
##	5.	Release
	•	Known-good tagged states, rollback-friendly
	•	v0.06 adds stable VGM playback + strict debug contracts

This approach keeps the code Copilot-checkable: the spec is explicit enough that implementations don’t need guesswork.

⸻

# Credits / references
	•	SN76489 PSG behavior: community knowledge + classic chip documentation conventions.
	•	Python audio backend: sounddevice (PortAudio).
	•	MIDI backend: python-rtmidi (CoreMIDI on macOS).
	•	<b>Michiel Erasmus</b> for iterative test-driven feedback loops and ruthless sanity checks.
