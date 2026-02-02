TS v1.4 is goedgekeurd.

# Tegniese Spesifikasie (TS) v1.4  
## SN76489 Emulator v0.05 – Muzikaliteit & Speelbaarheid

Datum: 2 Februarie 2026  
Gebaseer op: Funksionele Spesifikasie (FS) v1.6 (goedgekeur)  
Target platform: MacOS 26.2  
Python: 3.12  
Artefak: Eén Python-bestand (sn76489_emulator.py)  
Status: Ontwerp-kontrak (geen implementasie-kode)

---

## 0. Doel van v0.05

v0.05 brei SN76489 uit van ’n stabiele emulator na ’n **speelbare musikale engine**, sonder om die register-gedrewe model te verlaat.

Hooffokus:
1. Polyfoniese voice allocation (tone0 / tone1 / tone2)
2. ADSR-lite envelope via volume register writes
3. Pitch bend ondersteuning (beperk, SN76489-realistisch)
4. Verbeterde MIDI speelbaarheid
5. Uitbreiding van test- en debug-modes

Geen regressie teen v0.04 word aanvaar nie.

---

## 1. Voice Allocation (Polyfonie per Chip)

### 1.1 Ontwerpdoel

Gebruik **alle drie tone-kanale** van ’n SN76489-chip om:
- akkoorde te speel
- melodie + begeleiding gelyk te speel
- MIDI-ervaring instrumenteel te maak

Elke chip het **presies 3 stemme**:
- voice 0 → tone0
- voice 1 → tone1
- voice 2 → tone2

---

### 1.2 Stem-lewensiklus

Elke stem het die volgende state:
- IDLE (vry)
- ATTACK
- DECAY
- SUSTAIN
- RELEASE

’n Stem is **beset** vanaf NOTE_ON tot einde van RELEASE.

---

### 1.3 Voice Allocation Algoritme

Normatief vir v0.05:

1. By MIDI NOTE_ON:
   - soek **eerste vrye stem** (tone0 → tone1 → tone2)
2. Indien geen stem vry is nie:
   - NOTE_ON word **ge-ignoreer**
   - Geen voice stealing in v0.05

Rede:
- voorspelbaarheid
- eenvoud
- SN76489-histories realisties

Voice stealing kan later bygevoeg word (v0.06+).

---

### 1.4 MIDI Note → Stem Binding

Elke stem hou:
- huidige MIDI note
- huidige period
- envelope state

By NOTE_OFF:
- slegs die stem wat daardie note speel, gaan na RELEASE

---

## 2. Envelope Engine (ADSR-lite)

### 2.1 Ontwerpdoel

Maak note:
- minder clicky
- meer musikale
- meer “speelbaar”

Alles bly **register-gedrewe**:
- slegs volume register writes (R1 / R3 / R5)

Geen DSP-envelopes, geen floats in amplitude domein.

---

### 2.2 Envelope Parameters

CLI-parameters (v0.05):

- `--attack-ms <ms>`
- `--decay-ms <ms>`
- `--sustain-vol <0–15>`
- `--release-ms <ms>`

Volume-domein:
- 0 = hardste
- 15 = mute

---

### 2.3 Envelope Fases

#### Attack
- Volume: 15 → target_volume
- Tyd: attack-ms
- Stapgrootte: 1 volume-eenheid

#### Decay
- Volume: target_volume → sustain_volume
- Tyd: decay-ms

#### Sustain
- Volume: sustain_volume
- Duur: totdat NOTE_OFF ontvang word

#### Release
- Volume: huidige → 15
- Tyd: release-ms
- Daarna: stem word IDLE

---

### 2.4 Timing Model

- Envelope word **block-accurate** ge-evalueer
- Elke volume-verandering = één register write
- Geen sample-accurate vereiste

Stap-interval:
- bereken uit ms → samples → blocks
- afronding is aanvaarbaar

---

## 3. Pitch Bend

### 3.1 Ontwerpdoel

Basiese expressie vir MIDI controllers:
- subtiele buiging van toonhoogte
- geen vibrato/LFO

---

### 3.2 MIDI Pitch Bend Input

- MIDI Pitch Bend is 14-bit (0–16383)
- Middelpunt = 8192 (geen bend)

---

### 3.3 Bend Range

Normatief vir v0.05:
- ±2 semitone (konfigureerbaar later)

Mapping:
- bend_value → semitone_offset
- semitone_offset → frequency multiplier
- frequency → nuwe tone period

---

### 3.4 Beperkings

- Geen sample-accurate sweeps
- Geen per-stem vibrato
- Pitch bend beïnvloed **slegs aktiewe stemme**

---

## 4. MIDI Speelbaarheid

### 4.1 Verbeterings bo v0.04

- Akkoorde moontlik
- Note kan gehou word (sustain fase)
- Minder hoorbare artefakte
- Meer voorspelbare response

---

### 4.2 Sustain Pedal (v0.05)

- MIDI CC64 (sustain pedal):
  - OPTIONAL in v0.05
  - Indien geïmplementeer:
    - NOTE_OFF vertraag RELEASE
    - Stem bly in SUSTAIN tot pedal los

(Volledige sustain-logika kan later verfyn word.)

---

## 5. Runtime Modes / CLI Uitbreiding

### 5.1 Test modes

- `--test beep`
- `--test noise`
- `--test sequence`
- `--test chords`
  - speel 3 note gelyk (tone0/1/2)
- `--test sweep`
  - pitch sweep op een stem

---

### 5.2 Debug / Inspectie

- `--dump-regs`
  - registers
  - afgeleide frekwensies
  - voice allocation
  - envelope state per stem

- `--counters`
  - voices_used
  - env_steps
  - pitch_bend_events
  - MIDI events

- `--debug` (rate-limited)
  - NOTE_ON / NOTE_OFF
  - stem-toewysing
  - envelope transitions

---

## 6. Sanity Checks (v0.05)

Bestaande v0.04 checks **bly geldig**, plus:

1. Akkoord test:
   - 3 note gelyk
   - geen clipping
2. Sustain test:
   - note hou sonder volume-spronge
3. Release test:
   - gladde afsterf
4. Pitch bend test:
   - hoorbaar maar stabiel
5. Regressie:
   - v0.04 tests bly slaag

---

## 7. Bewus Uitgesluit (v0.05)

- Voice stealing
- ADSR-modulasie
- LFO/vibrato
- Effects
- Sample-accurate scheduling
- VST/AU plugin

---

## 8. Aanvaaringskriteria vir v0.05

v0.05 is suksesvol as:

- Polyfonie oor 3 tone-kanale werk
- ADSR-lite maak klank merkbaar musikaler
- Pitch bend werk sonder glitches
- MIDI speel voel instrumenteel
- Geen regressie teen v0.04

---

## 9. Vervolgkeuse

Kies een:

1. **TS v1.4 verfyn**  
   → “TS pas aan: …”

2. **TS v1.4 goedkeur**  
   → Ek genereer **SN76489 Emulator v0.05 code**  
     (nog steeds één Python-bestand)

---
