# Funksionele Spesifikasie (FS) v1.6  
## SN76489 PSG Emulasie in Python (MacOS)

**Datum:** 2 Februarie 2026  
**Status:** FS v1.6 (verfyn ná goedgekeurde v0.04 + v0.05)  
**Doelplatform:** MacOS 26.2  
**Python:** 3.12  
**Huidige status:** v0.05 stabiel ✅  
- Multi-chip (1–128)  
- Stereo routing (left/right/both)  
- Mixer + gain model  
- CLI: `--chips`, `--pan`, `--master-gain`, `--block-frames`  
- Debug: `--dump-regs`, `--counters`, `--debug` (rate-limited)  
- v0.05: polyfonie + ADSR-lite + pitch bend + `--test chords` + `--test sweep`

---

## 0. Konteks & Status

SN76489 Emulator is ’n **register-gedrewe** PSG engine/emulator in Python, gebou as ’n iteratiewe reeks milestones.

Belangrikste reël:
- Alle klank ontstaan uitsluitlik via **SN76489 register writes**.
- Geen direkte DSP sine generators as “shortcut” vir tone nie.

---

## 1. Doel

Bou ’n **SN76489 PSG emulasie/engine** in Python wat:

- Op MacOS loop
- Live audio na computer speakers uitstuur
- As ’n speelbare instrument kan optree (tests, sequencing, MIDI, later DAW)
- Multi-chip layering en stereo routing ondersteun
- Iteratief uitbrei:  
  **debug → sequencing → multi-chip → file playback (MIDI/VGM) → DAW/MIDI integrasie**

---

## 2. Scope

### 2.1 In scope (huidige + volgende iterasies)

#### Kern-emulasie
- 3× tone + 1× noise per chip
- register latch/data gedrag
- volume, noise modes, mixing
- mono per chip → stereo via routing

#### Engine / platform
- Multi-chip bank (1–128)
- Stereo routing: left/right/both
- Mixer + gain model
- Buffer rendering

#### Runtime modes / CLI
- Tests:
  - `--test beep`
  - `--test noise`
  - `--test sequence`
  - `--test chords`
  - `--test sweep`
- Debug/inspectie:
  - `--dump-regs`
  - `--counters`
  - `--debug` (rate-limited)

#### File playback (NUUT: v0.06 kandidaat)
- VGM file playback (SN76489 writes + delays)
- App-config base path vir VGM biblioteek
- CLI: spesifieke VGM file path speel onmiddellik

---

### 2.2 Out of scope (vir nou)
- VST/AU plugin (native DAW plugin)
- Cycle-perfect timing
- Effects (reverb/filter)
- Multi-file refactor (nog uitgestel)

---

## 3. Gebruikersverhaal (User Story)

> “Ek wil SN76489 as ’n engine hê wat eers in CLI tests stabiel is,  
> en dan kan uitbrei na file playback (VGM), en na DAW/MIDI gebruik as ’n virtuele instrument.”

---

## 4. Funksionele Vereistes

### 4.1 SN76489 Core
- Tone: 3 kanale, 10-bit period, 4-bit volume
- Noise: white/periodic, rate select, volume
- Register writes:
  - latch/data bytes
  - tone period updates
  - noise ctrl updates
  - volume updates
- Output:
  - mono PCM per chip

### 4.2 Multi-chip Engine
- Tot 128 onafhanklike chips
- Per chip: eie registers/state/counters
- Mixer: lineêr O(N)
- Per-chip gain + master gain

### 4.3 Stereo Routing
- Per chip of default global routing:
  - left / right / both
- Dual-mono (both) stuur selfde mono na L+R

### 4.4 Debug & Inspectie
- `--dump-regs` bly “golden output” format (stabiel vir regression)
- `--counters` toon per-chip en totaal stats

---

## 5. Test- en Run Modi (belangrik)

Die program moet verskillende run modes hê via CLI-argumente.

### 5.1 Beep Test
- `--test beep`
- Speel ’n vaste toon (bv. 440 Hz)
- Doel: audio path + tone register writes

### 5.2 Noise Test
- `--test noise`
- Toets white/periodic + rate/seed
- Doel: LFSR gedrag + determinisme

### 5.3 Mini Sequencer
- `--test sequence`
- Speel ’n vaste reeks note (bv. C–E–G–C)
- Doel: melodiese playback basis

### 5.4 Chords Test
- `--test chords`
- Speel 3 note gelyk (tone0/tone1/tone2)
- Doel: multi-channel mixing + clipping detection

### 5.5 Sweep Test
- `--test sweep`
- Pitch sweep (bv. 220→880 Hz)
- Doel: period write stabiliteit

---

## 6. Debug & Inspectie Modi

### 6.1 Register Dump
- CLI: `--dump-regs`
- Toon:
  - alle 8 SN76489 registers
  - latched register
  - afgeleide state (tone periods/freqs, volumes)
  - (v0.05+) voice allocation + envelope states

### 6.2 Debug Counters
- CLI: `--counters`
- Toon:
  - register writes
  - render calls
  - audio frames
  - MIDI events
  - envelope steps
  - (later) VGM commands processed + wait ticks

---

## 7. Kwaliteit-verbeterings

### 7.1 CLI: `--master-gain`
- Maklike volume trim sonder code edits
- Voorkom clipping by multi-channel/multi-chip

### 7.2 CLI: `--block-frames`
- Toets stutters vs latency
- Handig vir VGM playback later

### 7.3 “Golden” debug outputs
- `--dump-regs` output in vaste format (regression-friendly)

### 7.4 Seed control vir noise
- CLI: `--noise-seed 0x4000`
- Deterministiese tests

### 7.5 Noise rate control
- CLI: `--noise-rate {div16,div32,div64,tone2}`

---

## 8. Toekomstige Modules (later) — **verfyn**

### 8.1 MIDI-bestand Playback (later)
- Lees MIDI file met 1 instrument track
- Note on/off → SN76489 register writes
- File-based playback (nie realtime)
- Doel: reproduseer eenvoudige melodieë vanuit MIDI files

### 8.2 VGM Playback (NUUT, v0.06 kandidaat)
Doel: speel bestaande retro game music (VGM) deur die SN76489 emulator.

Funksionele vereistes:
- Lees VGM file vanaf disk
- Ondersteun:
  - SN76489 write commands
  - wait/delay commands (timing)
- Playback loop:
  - parse command → apply register write(s) → render wagtyd → herhaal
- CLI ondersteuning:
  - `--vgm-path <absolute_or_relative_path>`
    - begin onmiddellik met afspeel
  - `--vgm-base-dir <path>`
    - default location (app config) waar VGM library woon

Default voorbeeld (app-config instelbaar):
- `/Volumes/data1/Yandex.Disk.localized/michiele/Arduino/PCB Ontwerp/KiCAD/github/SN76489-synth-midi/src/tmp/src/`

Belangrik:
- v0.06 timing mag “audio-rate accurate” wees (nie cycle-perfect nie)
- Debug counters moet VGM metrics toon (commands, waits)

### 8.3 SN76489 as DAW Instrument (verfyn MIDI gedeelte)
Doel: engine kan MIDI ontvang en speel sodat Ableton/Logic MIDI kan stuur na die engine.

Vereistes:
- MIDI input via CoreMIDI (MacOS)
- MIDI channel → chip mapping
- Minimum:
  - Note On/Off
  - Velocity → volume mapping
  - Pitch Bend (v0.05)
  - (opsioneel later) sustain pedal CC64

Uitdruklik nie:
- AU/VST plugin in hierdie fase
- Multi-thread timing correction

---

## 9. Github (NUUT)

### 9.1 README vir breë publiek
Die repo moet ’n README hê wat:
- verduidelik wat die projek is (SN76489 emulator / engine)
- hoe om te installeer (pip deps)
- hoe om tests te run (beep/noise/sequence/chords/sweep)
- hoe om MIDI te gebruik
- hoe VGM playback later gaan werk (roadmap)

### 9.2 Git commit comment in tegniese specs
Elke release moet ’n standaard commit message hê (in TS opgeneem).

Voorbeeld vir v0.05:
- `git commit -m "v0.05: polyphonic voices + ADSR-lite + pitch bend; add chords/sweep tests"`

Vir v0.06 (VGM fokus) sal ons later ’n soortgelyke “one-liner” definieer.

---

## 10. Aanvaaringskriteria (vir volgende iterasie — v0.06 fokus op VGM)

Sukses vir v0.06 as:
- `--vgm-path` speel ’n VGM file hoorbaar af
- `--vgm-base-dir` werk as default library location
- Debug counters toon VGM commands processed + waits
- Geen regressie: v0.05 tests werk nog

---

## Review / Vervolgkeuses

Kies een:

1) **FS verfyn**  
   → “FS pas aan: …”

2) **FS goedkeur**  
   → Ek skryf **Tegniese Spesifikasie (TS v1.5)** vir v0.06 (VGM playback + CLI + timing model + debug output + sanity checks)

3) Ná TS v1.5  
   → Ek vra of jy **code v0.06** wil genereer (nog steeds één Python-bestand)

---