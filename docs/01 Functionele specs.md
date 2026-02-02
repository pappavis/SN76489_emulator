# Funksionele Spesifikasie (FS) v1.6  
## SN76489 PSG Emulasie in Python (MacOS)

Datum: 2 Februarie 2026  
Status: FS v1.6 (uitgebreid, kontrak-dokument)  
Doelplatform: MacOS 26.2  
Python: 3.12  
Huidige stabiele implementasie: v0.05 ✅  

---

## 0. Konteks & Agtergrond

Hierdie projek is ’n **register-gedrewe SN76489 PSG emulasie / engine** in Python.

Belangrike ontwerpbesluit (nie onderhandelbaar nie):
- Alle klank ontstaan **uitsluitlik** deur SN76489-register writes
- Geen direkte DSP tone generators as plaasvervanger vir chipgedrag
- Die emulator dien as:
  - klankgenerator
  - toetsinstrument
  - referensie-implementasie
  - basis vir toekomstige hardware-integrasie

Die projek word iteratief ontwikkel met streng skeiding tussen:
- Funksionele spesifikasie (FS)
- Tegniese spesifikasie (TS)
- Implementasie (code)

---

## 1. Doel

Bou ’n SN76489 PSG engine wat:

- Op MacOS loop
- CLI-first en toetsbaar is
- Deterministies is (reproseerbare output)
- Kan funksioneer as:
  - emulator
  - musikale instrument
  - MIDI-klankbron
  - VGM player
- Iteratief uitbrei volgens vaste fases:
  
  **debug → sequencing → polyfonie → MIDI → VGM → DAW → hardware**

---

## 2. Scope

### 2.1 In scope (huidig + volgende iterasies)

#### Kern-emulasie
- 3× tone-kanale
- 1× noise-kanaal
- Latch/data registergedrag
- Volume, noise modes, clock-dividers
- Mono PCM per chip

#### Engine / platform
- Multi-chip ondersteuning (1–128 SN76489s)
- Stereo routing (left / right / both)
- Mixer + master gain model
- Buffer-gebaseerde audio rendering

#### Runtime modes / CLI
- Tests:
  - `--test beep`
  - `--test noise`
  - `--test sequence`
  - `--test chords`
  - `--test sweep`
- Debug:
  - `--dump-regs`
  - `--counters`
  - `--debug` (rate-limited)

#### MIDI
- CoreMIDI input (MacOS)
- MIDI channel → chip mapping
- Note On / Note Off
- Velocity → volume mapping
- Pitch bend (v0.05)
- Optioneel: sustain pedal (CC64)

#### File playback
- MIDI file playback (later)
- **VGM file playback** (v0.06 fokus)

---

### 2.2 Uitdruklik buite scope (vir nou)

- VST/AU plugins
- Cycle-perfect timing
- DSP effects (reverb, delay, filters)
- Multi-file refactor
- Grafiese UI

---

## 3. Runtime Modes / CLI

CLI is die **primêre beheerlaag** vir:
- ontwikkeling
- regressie
- Copilot-validasie
- dokumentasie

### 3.1 Test modes

- `--test beep`  
  Basiese sanity test (audio path + tone register)

- `--test noise`  
  Noise kanaal, LFSR, rate en seed

- `--test sequence`  
  Vasgestelde note-reeks (melodie)

- `--test chords`  
  3 stemme gelyk (tone0/1/2)

- `--test sweep`  
  Period sweep (bv. 220 → 880 Hz)

---

### 3.2 Debug / Inspectie modes

- `--dump-regs`
  - Alle 8 SN76489 registers
  - Latched register
  - Afgeleide state (frekwensies, volumes)
  - Voice allocation (v0.05+)
  - Envelope states

- `--counters`
  - Register writes
  - Render calls
  - Audio frames
  - MIDI events
  - Envelope steps
  - (later) VGM commands / waits

- `--debug`
  - Mens-leesbare debug output
  - Rate-limited
  - Geen realtime spam

---

## 4. NUUT — Debug Output Gedrag (ADD)

### 4.1 Parameter Echo (verpligtend)

Wanneer `sn76489_emulator.py` met **`--test` parameters** gerun word:

- Alle relevante CLI parameters **MOET eksplisiet na console ge-echo word**
- Doel:
  - visuele bevestiging
  - log-vergelyking
  - regressie-verifikasie

Voorbeeld:
TEST MODE: sequence
chips=1 pan=both
attack-ms=5 decay-ms=80 sustain-vol=8 release-ms=120
block-frames=512 master-gain=0.25

Geen parameter mag “stil” wees tydens tests nie.

---

## 5. SN76489 Core Funksionele Vereistes

- 3 tone-kanale
  - 10-bit period
  - 4-bit volume
- 1 noise-kanaal
  - white / periodic
  - rate: div16/div32/div64/tone2
- Register writes:
  - latch + data bytes
  - tone period updates
  - noise ctrl updates
  - volume updates
- Output:
  - mono PCM per chip

---

## 6. Multi-chip Argitektuur

- 1 tot 128 onafhanklike SN76489s
- Elke chip het:
  - eie registers
  - eie voices
  - eie counters
- Mixer:
  - lineêr O(N)
  - geen gedeelde state
- Per-chip gain + master gain

---

## 7. Stereo Routing

- Per run of per chip:
  - `left`
  - `right`
  - `both`
- `both` = dual-mono (selfde mono na L+R)

---

## 8. Kwaliteit-verbeterings (bestaand)

- `--master-gain`
- `--block-frames`
- `--noise-seed`
- `--noise-rate`
- “Golden” `--dump-regs` output

---

## 9. MIDI (verfyn)

### 9.1 MIDI input
- CoreMIDI
- Channel → chip mapping
- Note On / Off
- Velocity → volume
- Pitch bend
- Optioneel CC64 sustain

### 9.2 NUUT — Eksklusiwiteit (ADD)
- `--midi-in` en `--vgm-path` is **wedersyds eksklusief**
- Program moet hard faal met duidelike foutboodskap indien albei gebruik word

---

## 10. VGM Playback (v0.06 fokus)

### 10.1 Basiese vereistes
- Lees VGM file vanaf disk
- Ondersteun:
  - SN76489 write commands
  - wait / delay commands
- Audio-rate accurate timing is voldoende

---

### 10.2 VGM CLI (ADD)

- `--vgm-path <file>`
  - speel gespesifiseerde VGM onmiddellik

- `--vgm-base-dir <path>`
  - default VGM library directory
  - voorbeeld:
    ```
    /Volumes/data1/Yandex.Disk.localized/michiele/Arduino/PCB Ontwerp/KiCAD/github/SN76489-synth-midi/src/tmp/src/
    ```

- `--vgm-loop`
  - loop playback
  - default: speel 1 keer

- `--vgm-speed <factor>`
  - 1.0 = normaal
  - <1.0 = stadiger
  - >1.0 = vinniger
  - slegs vir debug/analise

- `--vgm-list`
  - lys alle `.vgm` files in `--vgm-base-dir`
  - output na console
  - opsioneel beskikbaar as interne datastruktuur

---

## 11. Debug & Counters (uitgebrei)

Counters moet ook bevat:
- VGM commands processed
- VGM wait ticks
- Loop count

Alles sigbaar via `--counters`.

---

## 12. Github

### 12.1 README (breë publiek)
README moet beskryf:
- wat die projek is
- hoe om te installeer
- hoe om tests te run
- hoe om MIDI te gebruik
- VGM roadmap

### 12.2 Git commit messages (kontrak)
Elke release het ’n vasgestelde commit style.

Voorbeeld:
git commit -m “v0.05: polyphonic voices + ADSR-lite + pitch bend; add chords/sweep tests”
---

## 13. Aanvaaringskriteria (v0.06 – VGM)

- `--vgm-path` speel hoorbare klank
- `--vgm-loop` werk
- `--vgm-speed` beïnvloed tempo
- `--vgm-list` wys files
- Debug counters wys VGM statistiek
- Geen regressie teen v0.05

---

## 14. Vervolgkeuses

1) FS verder verfyn  
2) FS goedkeur → TS v1.5 (VGM playback)  
3) Pas-op-die-plaas / cleanup

---