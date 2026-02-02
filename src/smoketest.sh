echo "Starting smoketest for SN76489 emulator..."
python sn76489_emulator.py --test beep
python sn76489_emulator.py --test beep --chips 2 --pan left
python sn76489_emulator.py --test beep --chips 2 --pan right
python sn76489_emulator.py --test noise --noise-mode white --noise-seed 0x4000 --noise-rate div32 --seconds 1
python sn76489_emulator.py --test noise --noise-mode white --noise-seed 0x4000 --noise-rate div32 --seconds 1
python sn76489_emulator.py --test sequence --attack-ms 5 --decay-ms 80 --sustain-vol 8 --release-ms 140
python sn76489_emulator.py --test chords --attack-ms 5 --decay-ms 60 --sustain-vol 8 --release-ms 180
python sn76489_emulator.py --test sweep --seconds 2
python sn76489_emulator.py --midi-list
echo "# stuur note-on/off vanuit Ableton/Logic, stop met Ctrl+C"
python sn76489_emulator.py --midi-in

echo "Smoketest completed."
