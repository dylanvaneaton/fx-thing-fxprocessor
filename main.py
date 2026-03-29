import jack
import time
import re
import sys
import subprocess
from pedalboard import Pedalboard, Reverb

SAMPLE_RATE = 48000
CHUNK_SIZE = 256


def main():

    # After determining if jack is open and if so closing it for a restart, it then prompts the user for a device from a list to start jack on.
    ensure_jack_running()

    # enstantiates the pedalboard object. effects are put into it in a list, and the list from start to end is the signal chain from start to end.
    board = Pedalboard([Reverb()])

    # enstantiate a jack client, which is how we get digital audio from a physical sound card
    client = jack.Client("fx-thing-fxprocessor")  # enstantiates jack client

    # registers an input to our jack client. without this, the program cannot be connected through jack to the physical sound device input
    inport = client.inports.register("input_1")
    # same as above but for the output. this is where the program will output the processed audio so it can be connected to the physical out through jack
    outport = client.outports.register("output_1")

    @client.set_process_callback  # as far as i understand this means the function below it will be set as the jack client callback function, similar to declaring the function then immediately passing it into the name of the decorator.
    def process(frames):
        # this gets the audio as a numpy array. it is one dimensional, and represents one buffer period of sound.
        audio = inport.get_array()

        # pedalboard expects a two dimensional array arranged as (channel, samples). jack does not have fundementally store multiple channels as one array, hence the diffefrence.
        audio_2d = audio.reshape(1, -1)

        processed = board(audio_2d, SAMPLE_RATE, reset=False)

        # though i don't quite understand the difference, this copies the processed array into the outport array. the : is telling numpy to copy the values of the processed array into the outport array instead of assigning the outport array to a new array with the new info because apparently that will not work.
        outport.get_array()[:] = processed[0]

    @client.set_shutdown_callback
    def shutdown(status, reason):
        print(f"JACK has Stopped. Reason:{reason}")

    # this activates the client and keeps it running until the with is exited.
    with client:
        # this connects the jack port for the mic on the focusrite to the inport in this client.
        client.connect("system:capture_1", inport)

        # this connects the outport of our client to the left channel on the focusrite
        client.connect(outport, "system:playback_1")

        # same as above, but left channel, so it plays out of the stereo out instead of just left
        client.connect(outport, "system:playback_2")

        # when enter is pressed it will exit out of the block
        input("Processing running. Press enter to exit.")


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
            str(CHUNK_SIZE),
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


if __name__ == "__main__":
    main()
