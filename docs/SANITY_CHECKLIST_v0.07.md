# v0.07 sanity checklist

```bash
python sn76489_emulator.py --test beep

python sn76489_emulator.py --test noise --noise-mode white --noise-rate div32 --noise-seed 0x4000
python sn76489_emulator.py --test noise --noise-mode white --noise-rate div32 --noise-seed 0x4000

python sn76489_emulator.py --test sequence --velocity-curve linear --debug
python sn76489_emulator.py --test sequence --velocity-curve log --debug
python sn76489_emulator.py --test sequence --velocity-curve exp --debug

python sn76489_emulator.py --test chords --pan both --voice-pan spread --dump-regs
python sn76489_emulator.py --test chords --pan both --voice-pan center --dump-regs

python sn76489_emulator.py --midi-list
python sn76489_emulator.py --vgm-list --vgm-base-dir "/path/to/vgms"
python sn76489_emulator.py --vgm-path "/path/to/song.vgm" --counters --dump-regs
python sn76489_emulator.py --vgm-path "/path/to/song.vgm" --vgm-loop --vgm-speed 0.5
python sn76489_emulator.py --vgm-path "/path/to/song.vgm" --vgm-loop --vgm-speed 2.0
```


ja graag!
```
24-mrt-2026 17:10
één finale code-patch ronde op die huidige sn76489_emulator_v007.py, gefokus op:
	•	render/mixer robuustheid
	•	counters korrektheid
	•	finale cleanup vir release

Dan gee ek jou daarna:
	•	die gepatchte finale code
	•	’n release README
	•	’n finale sanity checklist
	•	en ’n recommended tag/commit flow.
```
