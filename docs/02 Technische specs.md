# Tegniese Spesifikasie (TS) v1.5.1  
## SN76489 Emulator v0.06 – Volledige Tegniese Kontrak (MacOS, Python) — Tightened

Datum: 2 Februarie 2026  
Gebaseer op: Funksionele Spesifikasie (FS) v1.6 (goedgekeur)  
Target platform: MacOS 26.2  
Python: 3.12  
Artefak: Eén Python-bestand (`sn76489_emulator.py`)  
Status: Tegniese kontrak – volledig, geen implementasie-kode

---

## 0. Doel en Reikwydte van TS v1.5.1

TS v1.5.1 beskryf die volledige tegniese ontwerp van die SN76489 emulator/engine soos dit bestaan ná v0.05 en met die toevoeging van VGM playback (v0.06), met kontrak-verstrakking sodat geen komponent (mens of Copilot) hoef te raai nie.

Hierdie dokument:
- vervang nie vorige TS’e nie
- inkorporeer TS v1.4 volledig
- inkorporeer TS v1.5 volledig
- verstrak (tighten) spesifiek:
  - debug parameter echo (format + inhoud)
  - CLI konflik-matriks (mutual exclusivity + invalid kombinasies)
  - VGM parser subset (presiese command bytes)
  - VGM wait/timing model (presies)
  - `--vgm-list` output kontrak (presies)
  - error message format (presies, copy/paste-baar)
  - commit style (ASCII, code block)

TS v1.5.1 is die tegniese kontrak vir v0.06 implementasie.

---

## 1. Hoëvlak Argitektuur

Die emulator bestaan logies uit:

1. CLI / Runtime mode selector  
2. Audio engine (stream/block renderer)  
3. SN76489 chip core(s) (1–128 chips)  
4. Voice allocation + envelope engine (polyfonie per chip)  
5. MIDI input layer (CoreMIDI via python-rtmidi)  
6. VGM playback layer (file parser + scheduler)  
7. Debug & inspectie layer (dump/counters/debug/echo)

Alles bly binne één Python-bestand, maar met streng skeiding van verantwoordelikhede.

---

## 2. SN76489 Core Emulasie

### 2.1 Chip Model

Elke SN76489-chip implementeer:
- 3 tone-kanale
- 1 noise-kanaal
- 8 interne registers
- latch/data write gedrag

Registers (logiese map):
- Tone 0 period (10-bit)
- Tone 1 period (10-bit)
- Tone 2 period (10-bit)
- Noise control
- Volume registers (4-bit) per kanaal

Volume betekenis:
- 0 = maksimum volume
- 15 = mute

Alle klank ontstaan uitsluitlik deur register writes.

### 2.2 Noise Generator

- White noise
- Periodic noise
- Rate seleksies:
  - div16
  - div32
  - div64
  - tone2-linked
- LFSR implementasie
- Opsionele seed (`--noise-seed`) vir determinisme

---

## 3. Audio Rendering & Timing

### 3.1 Sample Rate
- Konfigureerbaar
- Default: 44100 Hz

### 3.2 Block Rendering
- Audio word gegenereer in blokke (`--block-frames`)
- Geen sample-accurate scheduling vereis nie
- Block-accurate timing is voldoende

---

## 4. Multi-chip Argitektuur

- 1 tot 128 SN76489 chips
- Elke chip:
  - eie registers
  - eie voices
  - eie envelopes
  - eie counters

Mixer:
- Lineêr O(N)
- Geen gedeelde mutable state
- Per-chip gain (opsioneel)
- Global master gain (verpligtend)

---

## 5. Stereo Routing

- Routing opsies:
  - left
  - right
  - both (dual-mono)
- Routing geld per run (global) of per chip (implementasie-keuse)
- Normatief vir v0.06: global routing via `--pan {left|right|both}`

---

## 6. Voice Allocation (Polyfonie)

Elke chip het presies 3 stemme:
- voice 0 → tone0
- voice 1 → tone1
- voice 2 → tone2

Stem-state:
- IDLE
- ATTACK
- DECAY
- SUSTAIN
- RELEASE

### 6.1 Allocation Algoritme (normatief)

By MIDI NOTE_ON:
1. Soek eerste vrye stem (tone0 → tone1 → tone2)
2. Indien geen stem vry is nie:
   - NOTE_ON word geïgnoreer
   - Geen voice stealing in v0.06

---

## 7. Envelope Engine (ADSR-lite)

### 7.1 Beginsels
- Envelopes word slegs via volume register writes geïmplementeer
- Geen DSP amplitude envelopes
- 4-bit volume domein

### 7.2 Parameters (CLI)
- `--attack-ms`
- `--decay-ms`
- `--sustain-vol` (0–15)
- `--release-ms`

### 7.3 Timing Model
- Envelope stappe word block-accurate geëvalueer
- Elke volume verandering = register write
- Afronding is aanvaarbaar

---

## 8. Pitch Bend

- MIDI Pitch Bend (14-bit)
- Middelpunt = 8192
- Range: ±2 semitone (vast vir v0.06)
- Pitch bend beïnvloed slegs aktiewe stemme
- Geen vibrato / LFO

---

## 9. Runtime Modes / CLI (VOLLEDIG)

### 9.1 Test Modes
- `--test beep`
- `--test noise`
- `--test sequence`
- `--test chords`
- `--test sweep`

Elke test:
- Moet parameters echo (sien §10)
- Moet hoorbare klank lewer
- Moet stabiel wees
- Moet skoon stop

### 9.2 Debug / Inspectie
- `--dump-regs`
- `--counters`
- `--debug` (rate-limited)

Nota: `--noise-mode` en `--noise-rate` en `--noise-seed` bestaan reeds vir tests en moet in RUN CONFIG ge-echo word (sien §10).

---

## 10. Debug Output Kontrak (TIGHTENED)

### 10.1 Parameter Echo – presiese format (verpligtend)

Wanneer die program in een van hierdie modes run:
- `--test <mode>`
- `--midi-in`
- `--vgm-path <file>`
- `--vgm-list`

MOET die program ’n konsekwente, multi-line “RUN CONFIG” blok print voor playback begin.

#### 10.1.1 RUN CONFIG blok – presiese vorm

Eerste reël:
- `RUN CONFIG:`

Daarna presies hierdie sleutels (in hierdie orde), met `=`:

1. `mode=<test|midi|vgm|vgm_list>`
2. `test=<beep|noise|sequence|chords|sweep|none>`
3. `sample_rate=<int>`
4. `block_frames=<int>`
5. `chips=<int>`
6. `pan=<left|right|both>`
7. `master_gain=<float>`
8. `attack_ms=<float>`
9. `decay_ms=<float>`
10. `sustain_vol=<int>`
11. `release_ms=<float>`
12. `noise_mode=<white|periodic>`
13. `noise_rate=<div16|div32|div64|tone2>`
14. `noise_seed=<0x....>`
15. `midi_in=<0|1>`
16. `midi_port=<string|auto|none>`
17. `vgm_path=<string|none>`
18. `vgm_base_dir=<string|none>`
19. `vgm_loop=<0|1>`
20. `vgm_speed=<float>`
21. `dump_regs=<0|1>`
22. `counters=<0|1>`
23. `debug=<0|1>`

#### 10.1.2 Voorbeeld (test)

```text
RUN CONFIG:
mode=test
test=chords
sample_rate=44100
block_frames=512
chips=1
pan=both
master_gain=0.25
attack_ms=5.0
decay_ms=60.0
sustain_vol=8
release_ms=180.0
noise_mode=white
noise_rate=div32
noise_seed=0x4000
midi_in=0
midi_port=none
vgm_path=none
vgm_base_dir=none
vgm_loop=0
vgm_speed=1.0
dump_regs=1
counters=1
debug=0

10.2 --dump-regs kontrak
	•	Golden-friendly:
	•	stabiele sleutelorde
	•	geen timestamps
	•	Moet minstens bevat:
	•	chip index
	•	latched register
	•	registers R0..R7
	•	derived tone freqs
	•	noise mode/rate/seed
	•	voice allocation state per stem
	•	envelope phase per stem (state)

10.3 --counters kontrak
	•	Per chip:
	•	writes_total, latch, data
	•	renders, frames
	•	Totale:
	•	midi_events_total, note_on_total, note_off_total, pitch_bend_events
	•	voices_used_total, note_ignored_no_voice
	•	env_steps_total
	•	VGM totale (sien §14 en §17)

10.4 --debug kontrak (rate-limited)
	•	Mens-leesbare trace
	•	Geen onbounded spam
	•	Normatief:
	•	maksimum 20 debug lines per sekonde (soft limit) of “print elke N events”
	•	Mag toon:
	•	NOTE_ON/OFF
	•	stem-toewysing
	•	envelope transitions (opsioneel)
	•	VGM write + wait (opsioneel)

⸻

11. MIDI Input (VOLLEDIG)

11.1 MIDI Bron
	•	CoreMIDI (MacOS)
	•	Realtime MIDI input (python-rtmidi)

11.2 MIDI Mapping
	•	MIDI channel → chip mapping
	•	Default:
	•	channel 1 → chip 0
	•	channel 2 → chip 1
	•	ens.
	•	Indien --chips < N kleiner is as nodige channels:
	•	mapping wraps modulo chips (channel→(channel-1)%chips)

11.3 Ondersteunde Events
	•	Note On
	•	Note Off
	•	Velocity → volume mapping (4-bit)
	•	Pitch Bend (14-bit)
	•	Opsioneel: CC64 sustain

11.4 MIDI CLI
	•	--midi-list:
	•	print alle input ports
	•	exit 0
	•	--midi-in:
	•	open port “auto” (port 0) as --midi-port nie gegee is nie
	•	run until Ctrl+C
	•	--midi-port <name_substring>:
	•	kies eerste port wat substring match (case-insensitive)
	•	hard fail as niks match nie

⸻

12. CLI Konflik-Matriks (TIGHTENED)

12.1 Eksklusiewe modes (hard fail)

Die volgende kombinasies is ongeldig en MOET exit met fout:
	1.	--midi-in saam met --vgm-path
	2.	--midi-in saam met --test <mode>
	3.	--vgm-path saam met --test <mode>

12.2 Konsekwente foutboodskap

Normatief error prefix:
	•	ERROR:

Voorbeelde:
	•	ERROR: Choose either --midi-in OR --vgm-path (mutually exclusive).
	•	ERROR: Choose either --test OR --vgm-path (mutually exclusive).
	•	ERROR: Choose either --test OR --midi-in (mutually exclusive).

Normatief: exit code 2.

⸻

13. VGM Playback (NUUT in v0.06)

13.1 CLI Flags (VOLLEDIG)
	•	--vgm-path <file>
	•	--vgm-base-dir <path>
	•	--vgm-loop
	•	--vgm-speed <factor>
	•	--vgm-list

13.2 --vgm-base-dir default
	•	Indien nie gegee nie: gebruik app-config constant
	•	App-config default (FS v1.6):
	•	/Volumes/data1/Yandex.Disk.localized/michiele/Arduino/PCB Ontwerp/KiCAD/github/SN76489-synth-midi/src/tmp/src/

13.3 --vgm-list output kontrak (TIGHTENED)

Wanneer --vgm-list run:
	•	Print RUN CONFIG blok (mode=vgm_list)
	•	Print daarna:
	•	VGM LIST: <base_dir>
	•	dan elke file op eie lyn

Output format per lyn:
	•	<filename> (net file name, geen absolute path)
	•	sorteer alfabeties (case-insensitive)

Voorbeeld:
VGM LIST: /path/to/dir
01_intro.vgm
02_level1.vgm
boss_theme.vgm

Opsioneel interne datastruktuur:
	•	vgm_index: Dict[str, str]
	•	key = filename
	•	value = absolute path

13.4 --vgm-path gedrag
	•	Indien file nie bestaan nie:
	•	ERROR: VGM file not found: <path>
	•	exit code 2
	•	Indien file bestaan:
	•	start playback onmiddellik
	•	Normatief: VGM writes gaan na chip 0 (sien §16)

13.5 --vgm-loop gedrag
	•	As --vgm-loop teenwoordig:
	•	wanneer end-of-data bereik word:
	•	loop terug na command stream start
	•	increment vgm_loops_total
	•	As absent:
	•	stop playback skoon by end-of-data

13.6 --vgm-speed gedrag
	•	Float > 0 (hard fail anders)
	•	Default: 1.0
	•	Aangewend op wait/delay tyd:
	•	wait_samples_effective = round(wait_samples / vgm_speed)
	•	minimum 1 sample vir nonzero waits

⸻

14. VGM Parser Subset (TIGHTENED)

14.1 Header handling

Minimum:
	•	Validate magic string: b'Vgm '
	•	Read data offset:
	•	if data offset field = 0, default to 0x40
	•	else: data_start = 0x34 + data_offset
	•	Ignore onbekende header fields, maar moenie crash nie

14.2 Command bytes (normatief vir v0.06)

v0.06 ondersteun presies hierdie command types:
	1.	PSG write:
	•	0x50 <dd>
	2.	Wait n samples:
	•	0x61 <nn> <nn> (little endian 16-bit)
	3.	Wait 735:
	•	0x62
	4.	Wait 882:
	•	0x63
	5.	Short wait:
	•	0x70..0x7F => (cmd & 0x0F) + 1 samples
	6.	End:
	•	0x66

Alles anders:
	•	hard fail (sien §14.4)

14.3 Wait samples basis
	•	VGM wait commands is in “samples @ 44100Hz basis”
	•	Implementasie moet waits interpreteer vir engine sample_rate

Normatief:
	•	As engine sample_rate != 44100:
	•	wait_samples_engine = round(wait_samples_vgm * (engine_sr / 44100.0))
	•	Daarna apply --vgm-speed:
	•	wait_samples_effective = round(wait_samples_engine / vgm_speed)

Minimum:
	•	clamp minimum 1 sample vir nonzero waits

14.4 Unknown command error format (TIGHTENED)

As onbekende command byte gevind word:
	•	Print:
	•	ERROR: Unsupported VGM command 0xXX at offset 0xYYYYYYYY
	•	Exit code 2

Offset = file position in hex, 0-based.

⸻

15. Timing Model (VGM Scheduling)

15.1 Playback loop

Normatief:
	1.	read cmd byte
	2.	if PSG write: apply SN write onmiddellik
	3.	if wait: render audio vir wait_samples_effective in blocks
	4.	if end:
	•	if loop: seek to data_start, loops++
	•	else: stop playback

15.2 Ctrl+C gedrag

In loop playback moet Ctrl+C:
	•	stop audio stream
	•	print final counters/dump (as flags set)
	•	exit clean (exit code 0)

⸻

16. Routing + Multi-chip gedrag tydens VGM

Normatief v0.06:
	•	VGM PSG writes (0x50 dd) gaan na chip 0 se SN write()
	•	--chips > 1:
	•	chips 1..N bestaan
	•	hulle kry geen VGM writes
	•	implementasie mag hulle default mute (volume 15)

⸻

17. Debug & Counters (VGM)

By --counters moet VGM metrics bevat:
	•	vgm_commands_total
	•	vgm_psg_writes_total
	•	vgm_wait_events_total
	•	vgm_wait_samples_total
	•	vgm_loops_total

Definisies:
	•	commands_total tel alle commands (ook waits)
	•	psg_writes_total tel 0x50 writes
	•	wait_events_total tel elke wait command event
	•	wait_samples_total tel som van effektiewe wait samples (ná SR scaling + speed factor)

⸻

18. Sanity Checks (VERPLIGTEND)

18.1 Regressie (v0.05 tests MOET bly werk)
	1.	Audio path:

	•	python sn76489_emulator.py --test beep

	2.	Multi-chip + panning:

	•	python sn76489_emulator.py --test beep --chips 2 --pan left
	•	python sn76489_emulator.py --test beep --chips 2 --pan right

	3.	Noise determinisme:

	•	python sn76489_emulator.py --test noise --noise-mode white --noise-rate div32 --noise-seed 0x4000
	•	run 2×, output moet identies klink

	4.	Sequence + ADSR-lite:

	•	python sn76489_emulator.py --test sequence --attack-ms 5 --decay-ms 80 --sustain-vol 8 --release-ms 120

	5.	Chords:

	•	python sn76489_emulator.py --test chords --attack-ms 5 --decay-ms 60 --sustain-vol 8 --release-ms 180

	6.	Sweep:

	•	python sn76489_emulator.py --test sweep --seconds 2

18.2 VGM checks (v0.06)
	1.	List dir:

	•	python sn76489_emulator.py --vgm-list --vgm-base-dir <path>

	2.	Play once:

	•	python sn76489_emulator.py --vgm-path <file>

	3.	Loop:

	•	python sn76489_emulator.py --vgm-path <file> --vgm-loop

	4.	Speed:

	•	python sn76489_emulator.py --vgm-path <file> --vgm-speed 0.5
	•	python sn76489_emulator.py --vgm-path <file> --vgm-speed 2.0

	5.	Counters/dump:

	•	python sn76489_emulator.py --vgm-path <file> --counters --dump-regs

Pass kriteria:
	•	hoorbare klank
	•	geen crash/hang
	•	counters toon VGM metrics
	•	Ctrl+C stop skoon (vir loop)

Fail kriteria:
	•	geen geluid
	•	unsupported command errors op bekende goeie VGM (dan subset te klein)
	•	determinisme breek

⸻

19. Github Kontrak

19.1 README

README moet dokumenteer:
	•	installasie
	•	tests
	•	MIDI
	•	VGM flags (path/base-dir/list/loop/speed)
	•	sanity checklist (copy/paste)

19.2 Commit Style (ASCII + code block)

Vaste commit message styl vir v0.06:
git commit -m "v0.06: add VGM playback (path/list/loop/speed) with counters; keep MIDI+tests stable"
20. Bewus Uitgesluit (v0.06)
	•	Cycle-perfect timing
	•	Volledige VGM spec / alle commands
	•	Multi-chip VGM routing
	•	Seek/rewind
	•	DSP effects
	•	VST/AU

⸻

21. Vervolgkeuse

ies een:
	1.	TS v1.5.1 verfyn
→ “TS pas aan: …”
	2.	TS v1.5.1 goedkeur
→ Ek genereer SN76489 Emulator v0.06 code (nog steeds één Python-bestand), plus:

	•	GitHub README (v0.06)
	•	Install notes
	•	Sanity check checklist (copy/paste)
	•	Rollback hint na v0.05-known-good

	3.	Stop hier / refleksie


TS v1.5.1 goedkeur
