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

    # The task_id is a scalar, so we create a NumPy array for the constant
    # The shape should be () but ONNX requires at least a 1-element array for a constant.
    task_id_constant = gs.Constant(name="hardcoded_task_id_const", values=np.array(args.task_id, dtype=np.int64))

    # Replace every usage of the task_id_input with our new constant
    # This is more robust as multiple nodes could theoretically use this input.
    for node in graph.nodes:
        for i, inp in enumerate(node.inputs):
            if inp == task_id_input:
                node.inputs[i] = task_id_constant

    # Remove the original task_id from the graph's input list
    graph.inputs.remove(task_id_input)

    # Clean up any disconnected nodes from the graph
    graph.cleanup()

    # Export the modified graph to a new ONNX file
    onnx.save(gs.export_onnx(graph), args.output)
    print(f"[SUCCESS] Model saved to {args.output} with task_id hardcoded to {args.task_id}.")
    print("New model inputs:", [inp.name for inp in graph.inputs])

if __name__ == "__main__":
    main()