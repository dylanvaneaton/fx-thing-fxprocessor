import jack
from pedalboard import Pedalboard, Reverb

SAMPLE_RATE = 48000
CHUNK_SIZE = 256


def main():
    board = Pedalboard([Reverb()])
    client = jack.Client("fx-thing-fxprocessor")  # enstantiates jack client

    inport = client.inports.register("input_1")
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
        input("Processing running. Press enter to exit.")


if __name__ == "__main__":
    main()
