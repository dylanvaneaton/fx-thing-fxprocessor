# This file contains everything necessary to start this program and ensure jack is running with correct settings.
import time
import re
import sys
import subprocess
from config import SAMPLE_RATE, BUFFER_SIZE


def choose_alsa_device():
    result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)

    # parse out cards from aplay -l output
    devices = []
    for line in result.stdout.splitlines():
        match = re.match(r"^card (\d+):.*\[(.+?)\].*device (\d+):", line)
        if match:
            card_num = match.group(1)
            card_name = match.group(2)
            device_num = match.group(3)
            devices.append((card_num, device_num, card_name))

    if not devices:
        print("No ALSA devices found")
        return None

    print("\nChoose an ALSA Device:")
    for i, (card_num, device_num, card_name) in enumerate(devices):
        print(f"  [{i}] hw:{card_num},{device_num} — {card_name}")

    while True:
        choice = input("\nSelect device number: ").strip()
        if choice.isdigit() and int(choice) < len(devices):
            card_num, device_num, _ = devices[int(choice)]
            return f"hw:{card_num},{device_num}"
        print(f"Invalid choice, enter a number between 0 and {len(devices) - 1}")


def get_jack_pid():
    result = subprocess.run(["pgrep", "jackd"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def stop_jack(pid):
    for attempt in range(1, 4):
        result = subprocess.run(["kill", pid])
        if result.returncode != 0:
            print(
                f"Failed to send kill signal to JACK (pid {pid}), attempt {attempt}/3"
            )
        else:
            time.sleep(1)
            if get_jack_pid() is None:
                print("JACK stopped successfully")
                return True
            print(f"JACK still running after kill signal, attempt {attempt}/3")

    print("Failed to stop JACK after 3 attempts, proceeding anyway")
    return False


def start_jack():
    # Shows the user a list of alsa playback devices and prompts a choice out of the list.
    device = choose_alsa_device()
    subprocess.Popen(
        [
            "jackd",
            "-d",
            "alsa",
            "-d",
            device,
            "-r",
            str(SAMPLE_RATE),
            "-p",
            str(BUFFER_SIZE),
            "-n",
            "3",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    if get_jack_pid() is None:
        print("JACK failed to start, exiting")
        sys.exit()
    print("JACK started")


def ensure_jack_running():
    pid = get_jack_pid()
    if pid:
        print(f"JACK already running (pid {pid}), stopping...")
        stop_jack(pid)
    else:
        print("JACK not running, proceeding with start")
    start_jack()


def choose_jack_inport():
    result = subprocess.run(["jack_lsp"], capture_output=True, text=True)

    capture_ports = [
        line.strip()
        for line in result.stdout.splitlines()
        if re.search(r"capture", line, re.IGNORECASE)
    ]

    if not capture_ports:
        print("No JACK capture ports found.")
        return None

    print("\nAvailable JACK Capture Ports:")
    for i, port in enumerate(capture_ports):
        print(f"  [{i}] {port}")

    while True:
        choice = input("\nSelect input number: ").strip()
        if choice.isdigit() and int(choice) < len(capture_ports):
            chosen_capture_port = capture_ports[int(choice)]
            print(f"Selected: {chosen_capture_port}")
            return chosen_capture_port
        print(f"Invalid choice, enter a number between 0 and {len(capture_ports) - 1}")
