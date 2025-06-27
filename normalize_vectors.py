from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import torch as t
import os
import argparse

def normalize_vectors(model_size: str, is_base: bool, n_layers: int, behaviors):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        any_found = 0
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in behaviors:
            vec_path = get_vector_path(behavior, layer, get_model_path(model_size, is_base=is_base))
            if not os.path.exists(vec_path):
                            print(f"Warning: Vector file not found at {vec_path}")
                            continue
            else:
                any_found = 1
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        if any_found == 0:
            print("No vector files found")
            continue
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in behaviors:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in behaviors:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behaviors",
        nargs="+",
        type=str,
        default=ALL_BEHAVIORS,
    )
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")

    
    args = parser.parse_args()
    size_to_layers = {
        '7b': 32,
        '13b': 36
    }
    normalize_vectors(args.model_size, args.use_base_model, size_to_layers[args.model_size], args.behaviors)
    # normalize_vectors("7b", True, 32)
    # normalize_vectors("7b", False, 32, args.behaviors)
    # normalize_vectors("13b", False, 36, args.behaviors)
