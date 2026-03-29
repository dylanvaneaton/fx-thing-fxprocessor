import jack
import start
from pedalboard import Reverb
from config import SAMPLE_RATE


def main():

    # After determining if jack is open and if so closing it for a restart, it then prompts the user for a device from a list to start jack on.
    start.ensure_jack_running()
    # Choose jack capture to use for the input
    chosen_jack_inport = start.choose_jack_inport()

    # enstantiate a jack client, which is how we get digital audio from a physical sound card
    client = jack.Client("fx-thing-fxprocessor")  # enstantiates jack client

    # enstantiate a reverb to be used in the callback
    reverb = Reverb()

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

        processed = reverb.process(audio_2d, SAMPLE_RATE, reset=False)

        # though i don't quite understand the difference, this copies the processed array into the outport array. the : is telling numpy to copy the values of the processed array into the outport array instead of assigning the outport array to a new array with the new info because apparently that will not work.
        outport.get_array()[:] = processed[0]

    @client.set_shutdown_callback
    def shutdown(status, reason):
        print(f"JACK has Stopped. Reason:{reason}")

    # this activates the client and keeps it running until the with is exited.
    with client:
        # this connects the jack port for the mic on the focusrite to the inport in this client.
        client.connect(chosen_jack_inport, inport)

        # this connects the outport of our client to the left channel on the focusrite
        client.connect(outport, "system:playback_1")

        # same as above, but left channel, so it plays out of the stereo out instead of just left
        client.connect(outport, "system:playback_2")

        # when enter is pressed it will exit out of the block
        input("Processing running. Press enter to exit.")


if __name__ == "__main__":
    main()
