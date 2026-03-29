import jack
import start
from pedalboard import (
    Compressor,
    Gain,
    Reverb,
    HighpassFilter,
    LowpassFilter,
)
from config import SAMPLE_RATE, BUFFER_SIZE
import json
from pathlib import Path
from typing import Any, Literal, TypedDict, Callable
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import numpy as np

NodeHandleType = Literal["number", "boolean", "audio"]


graph_lock = threading.Lock()


class GraphReloader(FileSystemEventHandler):
    def __init__(self, runtime: GraphRuntime):
        super().__init__()
        self.runtime = runtime

    def on_modified(self, event):
        if event.src_path.endswith("effects.json"):
            print("Reloading effects.json...")
            new_graph = get_graph()
            new_effects = instantiate_effects(new_graph)
            with graph_lock:
                self.runtime.graph = new_graph
                self.runtime.effects = new_effects


class NodeHandle(TypedDict):
    name: str
    type: NodeHandleType


class GraphNode(TypedDict, total=False):
    id: str
    type: str
    data: dict[str, Any]


class GraphEdge(TypedDict):
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class Graph(TypedDict):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    processingOrder: list[str]


FxModuleFn = Callable[[GraphNode, dict[str, Any], Any], dict[str, Any]]


# setup (once)
@dataclass
class GraphRuntime:
    graph: Graph
    node_functions: dict[str, FxModuleFn]
    resolved: dict[str, Any] = field(default_factory=dict)
    effects: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


# --- Effect Instantiation ---


def instantiate_effects(graph: Graph) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for node in graph["nodes"]:
        node_id = node["id"]
        node_type = node["type"]
        if node_type == "Gain":
            effects[node_id] = Gain()
        if node_type == "Compressor":
            effects[node_id] = Compressor()
        if node_type == "Reverb":
            effects[node_id] = Reverb()
        if node_type == "LowPass":
            effects[node_id] = LowpassFilter()
        if node_type == "HighPass":
            effects[node_id] = HighpassFilter()

    return effects


# --- Core Algorithm ---


def resolve_handle_key(node_id: str, handle_name: str) -> str:
    return f"{node_id}::{handle_name}"


def gather_inputs(
    node_id: str,
    edges: list[GraphEdge],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for edge in edges:
        if edge["target"] == node_id:
            source_key = resolve_handle_key(edge["source"], edge["sourceHandle"])
            inputs[edge["targetHandle"]] = resolved.get(source_key)
    return inputs


def process_node(node: GraphNode, runtime: GraphRuntime) -> None:
    node_type = node["type"]
    fx_fn = runtime.node_functions.get(node_type)
    if fx_fn is None:
        raise ValueError(f"No fx module registered for node type: '{node_type}'")

    inputs = gather_inputs(node["id"], runtime.graph["edges"], runtime.resolved)

    # Inject context into Input nodes
    if node_type == "Input":
        inputs.update(runtime.context)

    effect = runtime.effects.get(node["id"])
    outputs = fx_fn(node, inputs, effect)
    for handle_name, value in outputs.items():
        key = resolve_handle_key(node["id"], handle_name)
        runtime.resolved[key] = value


def tick(runtime: GraphRuntime) -> dict[str, Any]:
    runtime.resolved = {}
    node_index = {node["id"]: node for node in runtime.graph["nodes"]}
    for node_id in runtime.graph["processingOrder"]:
        node = node_index.get(node_id)
        if node is None:
            raise ValueError(f"processingOrder references unknown node id: '{node_id}'")
        process_node(node, runtime)
    return runtime.resolved


# FX ---------------------------------------------------


def input_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    return {"output": inputs["audio"]}


def constant_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    data = node.get("data", {})
    value: float = float(data.get("value", 0.0))
    return {"output": value}


def gain_node(node: GraphNode, inputs: dict[str, Any], effect: Gain) -> dict[str, Any]:
    db = inputs.get("db")
    if db is None:
        raise ValueError(f"gain_node missing 'db'. inputs: {inputs}")
    effect.gain_db = float(db)
    return {"output": effect.process(inputs["input"], SAMPLE_RATE, reset=False)}


def output_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    return {"output": inputs["input"]}


def highpass_node(
    node: GraphNode, inputs: dict[str, Any], effect: HighpassFilter
) -> dict[str, Any]:
    cutoff = inputs.get("cutoff (hz)")
    effect.cutoff_frequency_hz = float(cutoff)  # type: ignore
    return {"output": effect.process(inputs["input"], SAMPLE_RATE, reset=False)}


def lowpass_node(
    node: GraphNode, inputs: dict[str, Any], effect: LowpassFilter
) -> dict[str, Any]:
    cutoff = inputs.get("cutoff (hz)")
    effect.cutoff_frequency_hz = float(cutoff)  # type: ignore
    return {"output": effect.process(inputs["input"], SAMPLE_RATE, reset=False)}


def reverb_node(
    node: GraphNode, inputs: dict[str, Any], effect: Reverb
) -> dict[str, Any]:
    effect.room_size = inputs.get("roomSize")  # type: ignore
    effect.damping = inputs.get("damping")  # type: ignore
    effect.wet_level = inputs.get("wetLevel")  # type: ignore
    effect.dry_level = inputs.get("dryLevel")  # type: ignore
    effect.width = inputs.get("width")  # type: ignore
    return {"output": effect.process(inputs["input"], SAMPLE_RATE, reset=False)}


def compressor_node(
    node: GraphNode, inputs: dict[str, Any], effect: Compressor
) -> dict[str, Any]:
    effect.threshold_db = inputs.get("threshold (db)")  # type: ignore
    effect.ratio = inputs.get("ratio (x:1)")  # type: ignore
    effect.attack_ms = inputs.get("attack (ms)")  # type: ignore
    effect.release_ms = inputs.get("release (ms)")  # type: ignore
    return {"output": effect.process(inputs["input"], SAMPLE_RATE, reset=False)}


# Math / Basic ----------------------------------------------------------------


def audioparam_rms_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:

    audio = inputs.get("input")

    if audio is None:
        return {"output": None, "value": 0.0}
    value = float(np.sqrt(np.mean(audio**2)))
    return {"output": value}


def audioparam_peak_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:

    audio = inputs.get("input")

    if audio is None:
        return {"output": None, "value": 0.0}

    value = float(np.max(np.abs(audio)))
    return {"output": value}


def mix_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    a = inputs.get("input 1")
    b = inputs.get("input 2")
    a_level: float = float(inputs.get("a_level") or 1.0)
    b_level: float = float(inputs.get("b_level") or 1.0)

    if a is None and b is None:
        return {"output": np.zeros((1, 1), dtype=np.float32)}
    if a is None:
        return {"output": b * b_level}
    if b is None:
        return {"output": a * a_level}

    return {"output": np.clip((a * a_level) + (b * b_level), -1.0, 1.0)}


def add_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    return {"output": inputs["number 1"] + inputs["number 2"]}


def multiply_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    return {"output": inputs["number 1"] * inputs["number 2"]}


def subtract_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    return {"output": inputs["number 1"] - inputs["number 2"]}


def divide_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    divisor = inputs["number 2"]
    if divisor == 0:
        raise ValueError("divide_node: division by zero")
    return {"output": inputs["number 1"] / divisor}


def exponent_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    return {"output": inputs["input"] ** inputs["exponent"]}


# Math / Comparison -----------------------------------------------------------


def greater_than_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    return {"output": inputs["this"] > inputs["isGreaterThan"]}


def less_than_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    return {"output": inputs["this"] < inputs["isLessThan"]}


# Math / Logic ----------------------------------------------------------------


def and_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    return {"output": bool(inputs["condition 1"]) and bool(inputs["condition 2"])}


def or_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    return {"output": bool(inputs["condition 1"]) or bool(inputs["condition 2"])}


def not_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    return {"output": not bool(inputs["condition"])}


# Math / Scaling --------------------------------------------------------------


def normalize_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    span = inputs["maximum"] - inputs["minimum"]
    if span == 0:
        return {"output": 0}
    return {"output": (inputs["input"] - inputs["minimum"]) / span}


def floor_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    return {"output": max(inputs["input"], inputs["floor"])}


def ceiling_node(
    node: GraphNode, inputs: dict[str, Any], effect: None
) -> dict[str, Any]:
    return {"output": min(inputs["input"], inputs["ceiling"])}


def sine_node(node: GraphNode, inputs: dict[str, Any], effect: None) -> dict[str, Any]:
    data = node.get("data", {})
    amplitude: float = float(inputs.get("amplitude", 1.0))
    frequency: float = float(inputs.get("frequency (hz)", 1.0))
    buffer_size: int = int(data.get("buffer_size", BUFFER_SIZE))

    phase: float = float(data.get("phase", 0.0))

    t = np.arange(buffer_size) / SAMPLE_RATE
    sine = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    new_phase = (phase + 2 * np.pi * frequency * buffer_size / SAMPLE_RATE) % (
        2 * np.pi
    )
    data["phase"] = new_phase

    audio = sine.astype(np.float32).reshape(1, -1)  # (1, buffer_size) for pedalboard
    value = float(sine[-1])  # scalar for param modulation

    return {
        "output": audio,  # wire to audio inputs
        "raw": value,  # wire to param inputs like db, cutoff etc.
    }


def get_graph() -> Graph:
    effects_path = Path(__file__).parent.parent / "effects.json"
    with effects_path.open() as f:
        return json.load(f)


node_functions: dict[str, FxModuleFn] = {
    "Input": input_node,
    "Output": output_node,
    "Constant": constant_node,
    # FX
    "Gain": gain_node,
    "Reverb": reverb_node,
    "Compressor": compressor_node,
    "HighPass": highpass_node,
    "LowPass": lowpass_node,
    # Math / Basic
    "Add": add_node,
    "Multiply": multiply_node,
    "Subtract": subtract_node,
    "Divide": divide_node,
    "Exponent": exponent_node,
    # Math / Comparison
    "GreaterThan": greater_than_node,
    "LessThan": less_than_node,
    # Math / Logic
    "And": and_node,
    "Or": or_node,
    "Not": not_node,
    # Math / Scaling
    "Normalize": normalize_node,
    "Floor": floor_node,
    "Ceiling": ceiling_node,
    "AudioToRms": audioparam_rms_node,
    "AudioToPeak": audioparam_peak_node,
    "SineWave": sine_node,
    "Mixer": mix_node,
}


def main():

    graph = get_graph()

    # setup (once)
    runtime = GraphRuntime(
        graph=graph,
        node_functions=node_functions,
        effects=instantiate_effects(graph),
    )

    # After determining if jack is open and if so closing it for a restart, it then prompts the user for a device from a list to start jack on.
    start.ensure_jack_running()
    # Choose jack capture to use for the input
    chosen_jack_inport = start.choose_jack_inport()

    # enstantiate a jack client, which is how we get digital audio from a physical sound card
    client = jack.Client("fx-thing-fxprocessor")  # enstantiates jack client

    # registers an input to our jack client. without this, the program cannot be connected through jack to the physical sound device input
    inport = client.inports.register("input_1")
    # same as above but for the output. this is where the program will output the processed audio so it can be connected to the physical out through jack
    outport = client.outports.register("output_1")

    @client.set_process_callback
    def process(frames):
        # Get and reshape the JACK buffer
        audio_jack = inport.get_array()
        audio = audio_jack.reshape(1, -1)

        # Inject into the runtime context so Input nodes can access it
        runtime.context = {"audio": audio}

        # Run the graph
        with graph_lock:
            resolved = tick(runtime)

        # Pull the output back out — find the output node's audio
        output_audio = resolved.get("Output::output")
        if output_audio is not None:
            outport.get_array()[:] = output_audio[0]

    @client.set_shutdown_callback
    def shutdown(status, reason):
        print(f"JACK has Stopped. Reason:{reason}")

    # this activates the client and keeps it running until the with is exited.
    with client:
        client.connect(chosen_jack_inport, inport)
        client.connect(outport, "system:playback_1")
        client.connect(outport, "system:playback_2")

        observer = Observer()
        observer.schedule(
            GraphReloader(runtime),
            path=str(Path(__file__).parent.parent),
            recursive=False,
        )
        observer.start()

        try:
            while True:
                user_input = (
                    input(
                        "Type change or c to change jack input, or type exit to close.\n> "
                    )
                    .strip()
                    .lower()
                )
                if user_input == "exit":
                    break
                elif user_input in ("change", "c"):
                    client.disconnect(chosen_jack_inport, inport)
                    chosen_jack_inport = start.choose_jack_inport()
                    client.connect(chosen_jack_inport, inport)
        finally:
            observer.stop()
            observer.join()


# # takes in the stream of audio from jack and prepares it for pedalboard.
# def input_node(jack_inport):
#     # this gets the audio as a numpy array. it is one dimensional, and represents one buffer period of sound.
#     audio_jack = jack_inport.get_array()
#     # pedalboard expects a two dimensional array arranged as (channel, samples). jack does not have fundementally store multiple channels as one array, hence the diffefrence.
#     audio = audio_jack.reshape(1, -1)
#     return audio


if __name__ == "__main__":
    main()
