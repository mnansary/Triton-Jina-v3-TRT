import onnx_graphsurgeon as gs
import numpy as np
import onnx
import argparse
import sys

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Hardcode the task_id in a Jina-v3 ONNX model.")
    parser.add_argument("--input", required=True, help="Path to the source ONNX model.")
    parser.add_argument("--output", required=True, help="Path for the new ONNX model with the hardcoded task_id.")
    parser.add_argument("--task_id", type=int, required=True, help="The integer value for the task_id to hardcode.")
    return parser.parse_args()

# --- Main Logic ---
def main():
    args = get_args()
    print(f"[INFO] Loading ONNX model from: {args.input}")
    graph = gs.import_onnx(onnx.load(args.input))

    if graph is None:
        print("[ERROR] Failed to load the ONNX graph.", file=sys.stderr)
        sys.exit(1)

    # Find the task_id tensor in the graph's inputs
    task_id_input = next((inp for inp in graph.inputs if inp.name == "task_id"), None)

    if not task_id_input:
        print("[ERROR] Could not find 'task_id' in the model's inputs. Is this the correct model or has it been processed already?", file=sys.stderr)
        sys.exit(1)

    # Create the constant to replace the task_id input
    task_id_constant = gs.Constant(
        name="hardcoded_task_id_const", 
        values=np.array(args.task_id, dtype=np.int64)
    )

    # Replace every usage of the task_id_input with our new constant
    for node in graph.nodes:
        for i, inp in enumerate(node.inputs):
            if inp == task_id_input:
                node.inputs[i] = task_id_constant

    # Remove the original task_id from the graph's input list
    graph.inputs.remove(task_id_input)
    graph.cleanup()

    # Export the graph to ONNX ModelProto
    model_proto = gs.export_onnx(graph)

    # --- FIX 1: Force ONNX opset (main domain) to 14 ---
    for opset in model_proto.opset_import:
        if opset.domain == "":
            print(f"[INFO] Found ONNX opset version: {opset.version}. Forcing it to 14.")
            opset.version = 14

    # --- FIX 2: Force IR version to Triton-supported level (10) ---
    print(f"[INFO] Original IR version: {model_proto.ir_version}. Forcing it to 10.")
    model_proto.ir_version = 10

    # Save modified model
    onnx.save(model_proto, args.output)

    print(f"[SUCCESS] Model saved to {args.output} with task_id hardcoded to {args.task_id}.")
    print("New model inputs:", [inp.name for inp in graph.inputs])

if __name__ == "__main__":
    main()
