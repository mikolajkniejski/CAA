import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_params_from_filename(filename):
    """Extract layer and multiplier from filename"""
    # Example: results_layer=10_multiplier=0.0_behavior=utilitarian_type=ab_use_base_model=True_model_size=7b.json
    parts = filename.split('_')
    layer = None
    multiplier = None
    type = None
    use_base_model = None
    model_size = None

    for part in parts:
        if part.startswith('layer='):
            layer = int(part.split('=')[1])
        elif part.startswith('multiplier='):
            multiplier = float(part.split('=')[1])
        elif part.startswith('behavior='):
            behavior = str(part.split('=')[1])
        elif part.startswith('type='):
            type = str(part.split('=')[1])
        elif part.startswith('use_base_model='):
            use_base_model = str(part.split('=')[1])
        elif part.startswith('model_size='):
            model_size = str(part.split('=')[1])

    return layer, multiplier, type, use_base_model, model_size

def calculate_ratio_stats(data):
    # Are we biased towards a or b?
    res = []
    for item in data:
        a_prob = item.get('a_prob', 0)
        b_prob = item.get('b_prob', 0)
        if a_prob - b_prob > 0:
            res.append(1)
        else:
            res.append(0)

    proportion = sum(res) / len(res)

    return proportion


def main(behavior):
    """Analyze all utilitarian result files"""

    results_dir = Path(f'results/{behavior}')

    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist!")
        return {}, [], []


    analysis_results = {}


    # Process each JSON file
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            layer, multiplier, type, use_base_model, model_size = extract_params_from_filename(filename)

            if layer is None or multiplier is None:
                print(f"Could not extract parameters from {filename}")
                continue


            filepath = results_dir / filename

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                stats = calculate_ratio_stats(data)
                analysis_results[(layer, multiplier, use_base_model)] = stats

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return analysis_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    # parser.add_argument("--use_base_model", action="store_true", default=False)
    # parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=['altruistic'])

    args = parser.parse_args()
    for behavior in args.behaviors:
        print(main(behavior))
