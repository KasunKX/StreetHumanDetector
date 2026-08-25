# Raspberry Pi human detector

This application detects people with the lightweight MobileNet-SSD model and
sets physical header pin **11** (BCM **GPIO17**) active while a person is visible.
At startup, the output pulses for 0.5 seconds as a wiring and output self-test.

## Wiring

Pin 11 is a 3.3 V logic output. Connect an LED through a 330 ohm resistor for a
basic test. For a relay, lamp, motor, or other load, use a suitable transistor or
3.3 V-compatible relay module and a flyback diode where required. **Do not power
a load directly from a GPIO pin.** Join the module ground to a Raspberry Pi GND.

## Install on Raspberry Pi OS

```bash
sudo apt update
sudo apt install -y python3-opencv python3-gpiozero python3-picamera2
python3 download_models.py
```

Picamera2 is used automatically when available; otherwise the program opens USB
camera 0. Start the detector without a desktop preview:

```bash
python3 main.py
```

Useful test commands:

```bash
# Preview boxes on a connected display
python3 main.py --preview

# USB webcam and no real GPIO (development/test mode)
python3 main.py --camera usb --no-gpio --preview

# Active-low relay input and a 2 second output hold time
python3 main.py --active-low --off-delay 2

# Disable the startup pulse
python3 main.py --startup-blink 0
```

Run `python3 main.py --help` for all settings. Stop with Ctrl+C; shutdown always
returns the output to its inactive state.
