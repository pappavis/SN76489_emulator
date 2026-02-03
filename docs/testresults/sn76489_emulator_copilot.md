A3F9C7B1


**v0.07 – Implementasie is gereed** ✅  
Ek het die **enkele Python-lêer** vir v0.07 volgens **TS v1.6** gegenereer, plus ’n kort README vir die nuwe vlaggies. **Laai dit hier af:**

*   **sn76489\_emulator.py (v0.07)** – een-lêer emulator met al die nuwe funksies (instrumente, velocity curves, voice-alloc, sustain, dry-run, latency, state save/load)  
    [sn76489\_emulator.py](blob:https://m365.cloud.microsoft/8ee2c6c1-60ec-48e9-8eb5-3f174a0ad096)

*   **README\_v0\_07.md** – oorsig & vinnige begin vir v0.07  
    [README\_v0\_07.md](blob:https://m365.cloud.microsoft/142d6963-50a2-4ab1-ab12-78b83c7dc971)

> **Nota:** RUN CONFIG-blok behou die **oorspronklike 23 sleutels** en **voeg die nuwe TS v1.6-sleutels aan die einde by** om golden-diffs en terugwaartse vergelyking stabiel te hou.

***

## Wat is nuut in v0.07 (volgens TS v1.6)

*   **Instrument-profiele** (`--instrument <name>`)  
    Laai JSON uit `instruments/<name>.json` met ADSR-waardes, ens. As dit ontbreek: val terug na **default**. (Implementasie: `load_instrument()`.)
*   **Velocity curves** (`--vel-curve {linear|soft|hard|log}`)  
    Mappings na 4-bit volume per curve (funksie `velocity_to_vol4`).
*   **Voice allocation modes** (`--voice-alloc {first_free|round_robin}`)  
    Nuwe round-robin opsie via per-chip `last_assigned`-aanwysers.
*   **Sustain pedal (CC64)** (`--sustain`)  
    NOTE\_OFF word gehou terwyl sustain aan is; by loslaat word opgehoopte NOTE\_OFF’s toegepas.
*   **Dry-run** (`--dry-run`)  
    Geen audio, **geen effektiewe** registermutasies (maar **writes word steeds getel** vir debug); goed vir klanklose validasie.
*   **Latency debug** (`--debug-latency`)  
    Druk **rate-limited** `LAT: <ms>` (\~10/s) tydens block rendering.
*   **State snapshot/restore** (`--save-state`, `--load-state`)  
    Stoor/lees alle chip-state + (indien relevant) VGM counters as JSON.

**Alles bo-op v0.06** bly in plek: VGM subset + foutformate, toets-modes, MIDI, counters, `--vgm-list` ens.

***

## Vinnige begin (kopie/plak)

```bash
# Installeer (dieselfde as v0.06)
python3 -m pip install numpy sounddevice python-rtmidi

# Basiese sanity
python3 sn76489_emulator.py --test beep

# MIDI met velocity-curve, round-robin en sustain
python3 sn76489_emulator.py --midi-in --vel-curve log --voice-alloc round_robin --sustain

# VGM (loop + latency logs)
python3 sn76489_emulator.py --vgm-path /pad/na/song.vgm --vgm-loop --debug-latency

# Dry-run (geen klank, geen effektiewe writes) – vir CI/golden-vergelykings
python3 sn76489_emulator.py --test chords --dry-run --counters --dump-regs

# State save/load
python3 sn76489_emulator.py --test sequence --save-state state.json
python3 sn76489_emulator.py --load-state state.json --dump-regs
```

***

## RUN CONFIG – sleutelorde (uitgebrei)

Die oorspronklike **23 sleutels** (v0.06) kom **eers**, dan die v1.6-nuwe:

    ... 23 sleutels ...
    instrument=<name>
    voice_alloc=<first_free|round_robin>
    vel_curve=<linear|soft|hard|log>
    sustain=<0|1>
    dry_run=<0|1>
    debug_latency=<0|1>
    save_state=<string|none>
    load_state=<string|none>

***

## Belangrike ontwerpnotas

*   **Dry-run-gedrag:**
    *   `psg_write()` tel steeds `writes_total/latch/data`, **maar** mutasies op `regs/volumes` word oorgeslaan. Dit hou debug- en “wat sou gebeur”-gedrag zichtbaar sonder om state te verander.
*   **VGM:** ongewysig (subset presies soos v0.06, foutboodskap presies, wagte skaal vs. SR + `--vgm-speed`).
*   **Voice-alloc:** `first_free` bly die backwards-compatible default; `round_robin` is nuttig vir eksperimente, maar kan notas “chop”.

***

## Commit-lyn (ASCII, TS v1.6)

```bash
git commit -m "v0.07: add instrument profiles, velocity curves, voice-alloc modes, dry-run, latency debug"
```

***

## Volgende aksies (opsioneel)

*   Ek kan ’n **uitgebreide README (hoof README)** saamstel wat v0.05→v0.07 dek, plus volledige sanity-checks en voorbeeld-instrument JSON’e.
*   Wil jy hê ek moet ’n **klein `instruments/chip_lead.json`** voorbeeld saam genereer?

***

## Bronne / Verwysings

*   Jou **FS v1.6** (hierdie gesprek) — primêre kontrak vir v0.07.
*   Interne konteks (Teams) oor jou SN76489-werk en repo-skakels wat die projekkonteks bevestig: “Retro chip synths from 1980 in Python” (Teams-boodskappe; interne verwysings). [\[us-prod.as...rosoft.com\]](https://us-prod.asyncgw.teams.microsoft.com/v1/objects/0-wus-d6-442c0e21afe20a59054cf20682fb4b74/views/original/sn76489_emulator.py), [\[us-prod.as...rosoft.com\]](https://us-prod.asyncgw.teams.microsoft.com/v1/objects/0-wus-d11-77660fb21ecd12bdc20d16cf40d494a4/views/original/README_v0_07.md), [\[Fun Stuff | Teams\]](https://teams.microsoft.com/l/message/19:2729154d04cc4b3d96d1491fba952a57@thread.skype/1770069501335?tenantId=e3b48527-4cbe-42a2-b4d2-11b3cc7a86fc&groupId=ffa3f3cf-8a3a-46e1-a7ff-0bb41b1be8f7&parentMessageId=1770069501335&teamName=Global%20IT%20Operations&channelName=Fun%20Stuff&createdTime=1770069501335), [\[Erasmus, M...el in chat | Teams\]](https://teams.microsoft.com/l/message/19:2d408c5a-e28e-4386-b5ae-9e3d0c386df5_913b7ce0-89c5-499d-9163-75709da2a874@unq.gbl.spaces/1770069026698?context=%7B%22contextType%22:%22chat%22%7D)

> Artefakte geskep met die Copilot Python-omgewing:
>
> *   [sn76489\_emulator.py (v0.07)](blob:https://m365.cloud.microsoft/8ee2c6c1-60ec-48e9-8eb5-3f174a0ad096)
> *   [README\_v0\_07.md](blob:https://m365.cloud.microsoft/142d6963-50a2-4ab1-ab12-78b83c7dc971)

Laat weet as ek die **voorbeeld-instrument JSON** moet bysit of die **sanity-check checklist** vir v0.07 moet uitbrei met dry-run en state save/load gevalle.




A3F9C7B1

Lekker — jy het nou:

```bash
python3 sn76489_emulator.py --test beep
```

Dit is **die korrekte kommando** om die nuwe **v0.07** implementasie te toets. Hier is wat **moet gebeur** wanneer jy dit run:

***

# ✅ Wat jy behoort te sien

### **1. RUN CONFIG blok**

Dit druk al die parameters, insluitend die nuwe TS v1.6-sleutels:

    RUN CONFIG:
    mode=test
    test=beep
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
    dump_regs=0
    counters=0
    debug=0
    instrument=default
    voice_alloc=first_free
    vel_curve=linear
    sustain=0
    dry_run=0
    debug_latency=0
    save_state=none
    load_state=none

### **2. Hoorbare 440 Hz "beep"**

Die test harness doen:

*   Bereken periode vir 440 Hz
*   Batch-write:
    *   tone0 → 440 Hz
    *   vol0 → 2 (≈ *hard, helder beep*)
*   Render vir `--seconds` (default 2 sekondes)

***

# 🔊 Geen klank gehoor?

Hier is die drie mees algemene redes *tydens MAC uitvoering*:

### **1. sounddevice nie beskikbaar?**

Run:

```bash
python3 - << 'EOF'
import sounddevice as sd
print(sd.query_devices())
EOF
```

As dit faal → installeer:

```bash
pip install sounddevice numpy
```

### **2. ‘device busy’ (CoreAudio)**

*   Probeer die Mac herstart
*   Maak Chrome tabs toe wat audio hou
*   Probeer:

```bash
python3 sn76489_emulator.py --test beep --block-frames 256
```

### **3. Volume te laag**

Gebruik `--master-gain`:

```bash
python3 sn76489_emulator.py --test beep --master-gain 0.8
```

***

# 🎧 Opsioneel: Debug / regs-dump

```bash
python3 sn76489_emulator.py --test beep --dump-regs
```

Jy sal sien:

    chip=0
    latched=R0
    R0=2 R1=0 R2=... etc
    tone0_hz=440.22 ...
    noise_mode=white
    ...

***

# 🔍 Wil jy hê ek moet:

1.  **Bevestig of die beep korrek gegenereer is op jou masjien?**
2.  **Help troubleshoot as jy geen klank hoor nie?**
3.  **Direk 'n klein instrument JSON voorbeeld toks maak (bv. chip\_lead.json)?**

Laat weet — ek is reg!


A3F9C7B1

Hier is ’n **geldige, voorstel-klaar** instrument JSON‑lêer wat perfek aansluit by **v0.07** en **TS v1.6**.  
Dit is geskik vir `instruments/chip_lead.json`.

***

# 🎹 **Voorbeeld: `chip_lead.json`**

```json
{
  "name": "chip_lead",
  "volume_curve": "linear",
  "attack_ms": 4.0,
  "decay_ms": 60.0,
  "sustain_vol": 7,
  "release_ms": 150.0,
  "detune_cents": 0.0,
  "bend_range_semitones": 2.0
}
```

***

# 📘 Uitleg van die waardes

### **name**

Die interne naam wat in `RUN CONFIG` weerspieël word wanneer jy `--instrument chip_lead` gebruik.

### **volume\_curve**

Ondersteunde curves:

*   `linear`
*   `soft`
*   `hard`
*   `log`

Dit beïnvloed hoe MIDI‑velocity → 4‑bit volume gemap word.

### **ADSR parameters**

*(Emuleer nie ’n DSP‑envelope nie; stuur net volume‑register writes volgens jou engine‑model.)*

*   **attack\_ms:** Hoe vinnig die volume oplig na ’n Note On
*   **decay\_ms:** Hoe vinnig dit sak tot sustain
*   **sustain\_vol:** 0–15 (maar 8 = “musikaal bruikbaar”)
*   **release\_ms:** Saktempo na Note Off

### **detune\_cents**

0 = geen detune  
Positiewe waardes maak toon hoër, negatiewe laer (deur periode effens te verander).

### **bend\_range\_semitones**

Hoe groot ’n MIDI pitch‑bend die toon mag skuif (± semitone‑reeks).

***

# 🎵 Opsionele “warm pad” instrument

As jy ’n alternatief wil hê:

```json
{
  "name": "warm_pad",
  "volume_curve": "soft",
  "attack_ms": 40.0,
  "decay_ms": 120.0,
  "sustain_vol": 10,
  "release_ms": 300.0,
  "detune_cents": 3.5,
  "bend_range_semitones": 1.0
}
```

***

# 📂 Waar om die lêer te plaas

    instruments/chip_lead.json

Dan gebruik jy dit so:

```bash
python3 sn76489_emulator.py --test chords --instrument chip_lead
```

***

Wil jy hê ek moet **3–5 standaard instrumente** saamstel (lead, bass, pad, arpeggio, noise-drum)?
