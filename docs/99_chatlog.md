```
24-mrt-2026. 
Welke items zijn nu op de roadmap
```

Goed dat je dit even scherp trekt — want op dit punt kan het makkelijk rommelig worden.

Hier is jouw huidige roadmap (objectief, opgeschoond) op basis van alles wat je gebouwd en goedgekeurd hebt:

⸻

🧭 SN76489 Roadmap — Status 24-mrt-2026

✅ Wat is al DONE (stabiel, bewezen)

🔹 v0.01 → v0.06 (COMPLEET)

Fundament + volwassen engine
	•	SN76489 core (3 tone + noise)
	•	Audio output (MacOS)
	•	Test harness:
	•	beep / noise / sequence / chords / sweep
	•	Debug & inspectie:
	•	--dump-regs (golden output)
	•	--counters
	•	--debug
	•	Multi-chip engine (1–128 chips)
	•	Stereo routing (L / R / both)
	•	Mixer + master gain
	•	Deterministische noise (rate + seed)
	•	Envelope engine (ADSR-lite)
	•	MIDI input (CoreMIDI)
	•	Pitch bend
	•	CLI contract (mutual exclusivity + RUN CONFIG)

⸻

🔹 v0.06 (CURRENT STATE = ⭐️)

Volledige playback engine
	•	VGM playback:
	•	--vgm-path
	•	--vgm-list
	•	--vgm-loop
	•	--vgm-speed
	•	VGM parser (subset, strict)
	•	Timing model (sample → block scaling)
	•	VGM counters & debug
	•	Ctrl+C clean shutdown
	•	Known-good release + rollback strategy

👉 Dit is nu je stabiele “platform”

⸻

🚧 Wat staat NU op de roadmap (nog NIET gebouwd)

Dit is de echte backlog — opgesplitst per pad.

⸻

🔵 Pad A — Muzikaliteit & Speelbaarheid (meest logisch volgende stap)

Status: 0% gestart
	•	Voice stealing (wanneer 3 voices vol zijn)
	•	Velocity curves (beter dan lineair → 4-bit volume)
	•	Sustain (CC64) gedrag verfijnen
	•	Per-voice panning (ipv global)
	•	Glide / portamento (period smoothing, geen DSP)
	•	MIDI speelgevoel verbeteren (latency perception, feel)

👉 Doel: dit ding voelt als een instrument

⸻

🟣 Pad B — Retro / VGM verdieping

Status: 10% (basis staat)
	•	Meer VGM commands ondersteunen
	•	“Soft skip” mode (debug vs strict)
	•	Seek / restart / loop-points
	•	Accurate wait accumulator (lange files)
	•	Performance bij grote VGM files
	•	Golden audio regression (vergelijk output over versies)

👉 Doel: meer compatibiliteit + betrouwbaarheid

⸻

🟢 Pad C — AI / Vergelijkings-harnas

Status: 0%
	•	Zelfde TS → Copilot / ChatGPT / Gemini implementaties
	•	Automatische test runner (CLI script)
	•	Diff op:
	•	--dump-regs
	•	--counters
	•	Documenteren waar AI faalt vs correct is
	•	“Golden spec → multiple implementations” experiment

👉 Doel: uniek thought leadership + engineering credibility

⸻

🟠 Pad D — Hardware bridge

Status: 0%
	•	Python engine → referentie model
	•	ESP32 / Arduino SN76489 aansturing
	•	Timing mapping (software → hardware)
	•	Scope metingen (Rigol DHO804)
	•	Audio vergelijking (emulator vs echte chip)

👉 Doel: van software → fysieke synth

⸻

🧠 Belangrijk inzicht (eerlijk)

Je zit nu op een kritisch kantelpunt:

👉 Alles tot v0.06 = platform bouwen
👉 Alles daarna = positionering kiezen

Als je nu alles tegelijk doet → project verwatert
Als je 1 pad kiest → project wordt serieus sterk

⸻

🎯 Mijn advies (recht vooruit)

Als je kijkt naar jouw gedrag + doelen:

👉 Start met Pad A (Muzikaliteit)
Waarom:
	•	direct hoorbaar resultaat
	•	sluit aan op je muziek/looper context
	•	maakt demo’s veel sterker
	•	basis voor DAW usage

Daarna:
👉 Pad B of D

⸻

👉 Jouw volgende stap

Beantwoord gewoon:

👉 “A” / “B” / “C” / “D”

Dan doe ik:
	•	FS v1.7 (super strak, geen ruis)
	•	volledig GitHub-ready
	•	exact scoped voor v0.07

En dan gaan we weer dezelfde flow doen:
👉 FS → TS → code → sanity → release 🚀


```
Start met Pad A (Muzikaliteit)
```

Goed. We gaan Pad A pakken.

Pad A — wat nu op de roadmap staat

Voor FS v1.7 / v0.07 is de shortlist:
	•	Voice stealing wanneer 3 voices bezet zijn
	•	Betere velocity-curves naar 4-bit volume
	•	Sustain/CC64 gedrag verfijnen
	•	Per-voice panning in plaats van alleen global pan
	•	Glide / portamento zonder DSP-shortcuts
	•	MIDI speelgevoel verbeteren

Mijn voorstel voor v0.07 scope

Ik zou dit niet allemaal tegelijk doen. De strakste v0.07 is:
	1.	Voice stealing policy
	2.	Velocity curve opties
	3.	Sustain pedal gedrag
	4.	Per-voice panning

En glide/portamento parkeren voor v0.08, anders wordt de TS onnodig breed.

Waarom:
	•	dit levert direct beter speelgevoel op
	•	blijft nog beheersbaar in één Python-bestand
	•	minder kans dat v0.06 stabiliteit sneuvelt

Volgende stap

Ik kan nu FS v1.7 uitschrijven in GitHub .md formaat voor Pad A / v0.07.

```
verfijningen aan FS1.7.2
Herlees FS1.7.0, FS 1.7.1 en FS 1.7.2.
Rapport eer welke items je wass vergeten.
Voeg de vergeten items toe aan FS 1.7.2 zoals pitch bending.
VOor je output genereer herlees weer FS 1.7.2 en sanity check en meld wat je was vegeten. 
Vraag mij om bevestiging of ik de geheel FS opnieuw wil genereer ~6500 woorden, of ik wil doorgaan naar TS.
```

Goed — dit is precies hoe je dit professioneel doet.
Ik heb nu FS v1.7.0, v1.7.1 en v1.7.2 opnieuw naast elkaar gelegd en puur gekeken naar:
	•	wat erin zat → en verdwenen is
	•	wat impliciet was → maar nu expliciet moet
	•	wat nog steeds ontbreekt voor implementatie

⸻

🔍 1. Wat je was vergeten (objectief, cross-version)

Dit zijn echte gaten die uit de vergelijking naar voren komen:

⸻

❗ 1.1 Pitch bend is verdwenen (regressie)

In eerdere iteraties (v1.6 / TS v1.4):
	•	pitch bend was aanwezig

In v1.7.2:
→ volledig weg

👉 Dit is een functionele regressie, geen detail.

Moet terug:
	•	14-bit MIDI pitch bend
	•	±2 semitone range
	•	per-voice toegepast
	•	register-driven (period recalculatie)

⸻

❗ 1.2 Envelope gedrag ontbreekt nog steeds

Alle versies:
	•	noemen ADSR-lite

Maar:
→ geen enkele versie definieert:
	•	hoe volume verandert
	•	hoe vaak
	•	hoe snel

👉 Zonder dit:
	•	geen consistent geluid
	•	geen reproduceerbare tests

⸻

❗ 1.3 State transitions niet vastgelegd

Je hebt fases:
	•	ATTACK / DECAY / SUSTAIN / RELEASE

Maar nergens:
	•	wanneer overgang gebeurt

👉 Dit is state machine incompleet

⸻

❗ 1.4 Timing model ontbreekt volledig

Alle FS versies missen:

👉 Wanneer gebeurt updates:
	•	per sample?
	•	per block?
	•	per MIDI event?

Dit raakt:
	•	envelope
	•	pitch bend
	•	sustain

⸻

❗ 1.5 Velocity=0 edge case ontbreekt

MIDI standaard:
	•	NOTE_ON velocity 0 = NOTE_OFF

Niet gedefinieerd → inconsistent gedrag

⸻

❗ 1.6 Per-chip isolatie impliciet, niet expliciet

Je hebt chip_id, maar:

👉 Niet gezegd:
	•	voice allocation is per chip
	•	stealing nooit cross-chip

⸻

❗ 1.7 Envelope + voice stealing interactie ontbreekt

Wat gebeurt als:
	•	voice in ATTACK wordt gestolen?

Niet gedefinieerd.

⸻

❗ 1.8 Debug coverage voor envelope ontbreekt

Je debugt:
	•	voice assign
	•	steal
	•	sustain

Maar niet:
	•	envelope phase transitions

👉 lastig debuggen

⸻

🧠 2. Wat je goed hebt gedaan (blijft staan)

FS v1.7.2 is nog steeds sterk op:
	•	voice mapping ✔️
	•	deterministic stealing ✔️
	•	sustain lifecycle ✔️
	•	pan model ✔️
	•	CLI contract ✔️

👉 Dat blijft ongewijzigd

⸻

🧩 3. Wat moet worden toegevoegd aan FS v1.7.2

Dit is de minimale set om compleet te zijn:

⸻

➕ 3.1 Pitch Bend (toevoegen)

Vereisten
	•	MIDI 14-bit (0–16383)
	•	center = 8192
	•	range = ±2 semitone

Gedrag
	•	per voice toegepast
	•	herbereken tone period
	•	geen DSP

⸻

➕ 3.2 Envelope Model (toevoegen)

Minimum definitie
	•	volume verandert in stappen (register writes)
	•	geen instant jumps (behalve NOTE_ON start)

Fases
	•	ATTACK → volume 15 → target
	•	DECAY → target → sustain
	•	SUSTAIN → constant
	•	RELEASE → current → 15

⸻

➕ 3.3 State transitions (toevoegen)
	•	NOTE_ON → ATTACK
	•	ATTACK → DECAY (na target bereik)
	•	DECAY → SUSTAIN
	•	NOTE_OFF → RELEASE (of sustain_hold)
	•	RELEASE → IDLE (volume = 15)

⸻

➕ 3.4 Timing model (toevoegen)

Normatief:
	•	block-based evaluation
	•	envelope updates per render block
	•	geen sample-accurate requirement

⸻

➕ 3.5 Velocity=0 (toevoegen)
	•	NOTE_ON velocity 0 = NOTE_OFF

⸻

➕ 3.6 Per-chip isolatie (toevoegen)
	•	voice allocation per chip
	•	geen cross-chip stealing

⸻

➕ 3.7 Debug uitbreiding (toevoegen)
	•	ENVELOPE_PHASE_CHANGE
	•	PITCH_BEND_UPDATE

⸻

🔍 4. Sanity check op huidige FS v1.7.2

Na herlezing:

👉 FS v1.7.2 is:
	•	logisch ✔️
	•	consistent ✔️
	•	deterministic ✔️

Maar:

👉 Nog niet volledig gedrag-gedefinieerd

⸻

🎯 5. Eindoordeel

👉 Je zit nu op:
	•	95% architectuur compleet
	•	maar 85% gedragsdefinitie compleet

⸻

🚀 6. Wat nu?

Ik ga nog NIET direct herschrijven (zoals je vroeg).

👉 Eerst jouw keuze:

⸻

Wil je dat ik:

1️⃣ Volledige FS v1.7.3 genereer (~6500 woorden, volledig tight, GitHub-ready)

→ inclusief:
	•	pitch bend
	•	envelope model
	•	timing
	•	zero ambiguity

⸻

2️⃣ Direct doorgaan naar TS v1.6

→ sneller, maar minder strak

⸻

3️⃣ Alleen specifieke toevoegingen laten genereren

⸻

👉 Antwoord gewoon:

“1” / “2” / “3”


```
1️⃣ Volledige FS v1.7.3 genereer (~6500 woorden, volledig tight, GitHub-ready)

→ inclusief:
	•	pitch bend
	•	envelope model
	•	timing
	•	zero ambiguity

VOor je output genereer herlees weer FS 1.7.3 en sanity check en meld wat je was vegeten. 

```

```
verfijningen aan FS1.7.2
Herlees TS 1.4, TS 1.5 en TS 1.6.
Rapport eer welke items je was vergeten.
Voeg de vergeten items toe aan TS 1.6 die je was vergeten
Voor je output genereer herlees weer FS 1.8 en sanity check opnieuw en meld wat je was vergeten. 
Vraag mij om bevestiging of ik de geheel TS opnieuw wil genereer ~6500 woorden, of ik wil doorgaan naar de volgende stap?

1️⃣ Genereer volledige TS v1.6.1 (~6500 woorden, volledige merge, GitHub-ready)

24-mrt-2026 16:55.
code review + regressie verslag + verbetervoorstelle

```

Hier is die verbeterde v0.07 code, plus die README, sanity checklist en commit message.

Bestanden:
	•	sn76489_emulator_v007.py￼
	•	README_v0.07.md￼
	•	SANITY_CHECKLIST_v0.07.md￼

Wat ek gefix het
	•	seconds nou netjies in Config
	•	split-render hack vervang met render_split() per chip
	•	active_notes cleanup by RELEASE -> IDLE
	•	counters nou weer sinvol
	•	pitch bend pad vereenvoudig
	•	v0.06 platformfeatures behou

Aanbevole commit message

git commit -m "v0.07: fix split-render mixing, note cleanup and counters; keep MIDI/VGM/tests stable"

Ek sou nou eers die sanity checklist run voor jy v0.07-known-good tag.



