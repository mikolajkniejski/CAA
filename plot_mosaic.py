"""
Script to analyze utilitarian results and create visualizations

Usage:
python analyze_utilitarian_results.py
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def set_plotting_settings():
    """Set matplotlib plotting settings inspired by utils/helpers.py"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            pass

    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 12,
        "figure.autolayout": True,
        'figure.dpi': 300,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

def extract_params_from_filename(filename):
    """Extract layer and multiplier from filename"""
    # Example: results_layer=10_multiplier=0.0_behavior=utilitarian_type=ab_use_base_model=True_model_size=7b.json
    parts = filename.split('_')
    layer = None
    multiplier = None

    for part in parts:
        if part.startswith('layer='):
            layer = int(part.split('=')[1])
        elif part.startswith('multiplier='):
            multiplier = float(part.split('=')[1])

    return layer, multiplier

def calculate_ratio_stats(data):
    """Calculate statistics for prob(matching_behavior)/prob(not_matching_behavior) ratios"""
    ratios = []
    matching_probs = []
    not_matching_probs = []

    for item in data:
        a_prob = item.get('a_prob', 0)
        b_prob = item.get('b_prob', 0)
        answer_matching = item.get('answer_matching_behavior', '')
        answer_not_matching = item.get('answer_not_matching_behavior', '')

        # Determine which probability corresponds to matching behavior
        if answer_matching == "(A)":
            matching_prob = a_prob
            not_matching_prob = b_prob
        elif answer_matching == "(B)":
            matching_prob = b_prob
            not_matching_prob = a_prob
        else:
            print(f"Warning: Unexpected answer_matching_behavior format: {answer_matching}")
            continue

        matching_probs.append(matching_prob)
        not_matching_probs.append(not_matching_prob)

        # Avoid division by zero
        if not_matching_prob != 0:
            ratio = matching_prob / not_matching_prob
            ratios.append(ratio)

    if not ratios:
        return {
            'mean_ratio': float('nan'),
            'std_ratio': float('nan'),
            'range_ratio': float('nan'),
            'count': 0,
            'mean_matching_prob': 0,
            'mean_not_matching_prob': 0,
            'matching_is_a_count': 0,
            'matching_is_b_count': 0
        }

    mean_ratio = sum(ratios) / len(ratios)
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    range_ratio = max_ratio - min_ratio

    # Calculate standard deviation
    if len(ratios) > 1:
        variance = sum((r - mean_ratio) ** 2 for r in ratios) / (len(ratios) - 1)
        std_ratio = variance ** 0.5
    else:
        std_ratio = 0.0

    # Count how many times A vs B was the matching behavior
    matching_is_a_count = sum(1 for item in data if item.get('answer_matching_behavior') == "(A)")
    matching_is_b_count = sum(1 for item in data if item.get('answer_matching_behavior') == "(B)")

    return {
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'range_ratio': range_ratio,
        'count': len(ratios),
        'mean_matching_prob': sum(matching_probs) / len(matching_probs) if matching_probs else 0,
        'mean_not_matching_prob': sum(not_matching_probs) / len(not_matching_probs) if not_matching_probs else 0,
        'matching_is_a_count': matching_is_a_count,
        'matching_is_b_count': matching_is_b_count
    }

def analyze_utilitarian_results():
    """Analyze all utilitarian result files"""
    results_dir = Path('results/altruistic')

    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist!")
        return {}, [], []

    # Store results: {(layer, multiplier): stats}
    analysis_results = {}

    # Get all unique layers and multipliers
    all_layers = set()
    all_multipliers = set()

    # Process each JSON file
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            layer, multiplier = extract_params_from_filename(filename)

            if layer is None or multiplier is None:
                print(f"Could not extract parameters from {filename}")
                continue

            all_layers.add(layer)
            all_multipliers.add(multiplier)

            filepath = results_dir / filename

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                stats = calculate_ratio_stats(data)
                analysis_results[(layer, multiplier)] = stats

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return analysis_results, sorted(all_layers), sorted(all_multipliers)

def print_table(results, layers, multipliers):
    """Print a formatted table"""
    print("\nMEAN prob(matching)/prob(not_matching) RATIOS BY LAYER AND MULTIPLIER")
    print("=" * 100)

    # Print header
    header = f"{'Layer':>5}"
    for mult in multipliers:
        header += f"{mult:>10.1f}"
    print(header)
    print("-" * len(header))

    # Print rows
    for layer in layers:
        row = f"{layer:>5}"
        for mult in multipliers:
            if (layer, mult) in results:
                mean_ratio = results[(layer, mult)]['mean_ratio']
                if str(mean_ratio) == 'nan':
                    row += f"{'N/A':>10}"
                else:
                    row += f"{mean_ratio:>10.4f}"
            else:
                row += f"{'---':>10}"
        print(row)

def print_behavior_distribution(results):
    """Print information about A vs B as matching behavior"""
    print("\nBEHAVIOR DISTRIBUTION ANALYSIS")
    print("=" * 50)

    total_a_matching = 0
    total_b_matching = 0

    for stats in results.values():
        total_a_matching += stats['matching_is_a_count']
        total_b_matching += stats['matching_is_b_count']

    total_questions = total_a_matching + total_b_matching
    if total_questions > 0:
        print(f"Total questions analyzed: {total_questions}")
        print(f"Times A was matching behavior: {total_a_matching} ({100*total_a_matching/total_questions:.1f}%)")
        print(f"Times B was matching behavior: {total_b_matching} ({100*total_b_matching/total_questions:.1f}%)")

    # Show sample from one file to verify
    sample_key = list(results.keys())[0] if results else None
    if sample_key:
        sample_stats = results[sample_key]
        print(f"\nSample from Layer {sample_key[0]}, Multiplier {sample_key[1]}:")
        print(f"  A was matching: {sample_stats['matching_is_a_count']} times")
        print(f"  B was matching: {sample_stats['matching_is_b_count']} times")

def create_heatmap(results, layers, multipliers):
    """Create a heatmap visualization"""
    set_plotting_settings()

    # Create matrix for heatmap
    matrix = np.full((len(layers), len(multipliers)), np.nan)

    for i, layer in enumerate(layers):
        for j, mult in enumerate(multipliers):
            if (layer, mult) in results:
                ratio = results[(layer, mult)]['mean_ratio']
                if str(ratio) != 'nan':
                    matrix[i, j] = ratio

    plt.figure(figsize=(12, 6))
    im = plt.imshow(matrix, cmap='RdYlBu_r', aspect='auto')

    # Set ticks and labels
    plt.xticks(range(len(multipliers)), [f"{m:.1f}" for m in multipliers])
    plt.yticks(range(len(layers)), layers)
    plt.xlabel('Multiplier')
    plt.ylabel('Layer')
    plt.title('Mean prob(matching)/prob(not_matching) Ratios Heatmap\n(Utilitarian Behavior)', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Mean Ratio', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(multipliers)):
            if not np.isnan(matrix[i, j]):
                text = plt.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig('utilitarian_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_layer_comparison(results, layers, multipliers):
    """Plot comparison across layers for different multipliers"""
    set_plotting_settings()

    plt.figure(figsize=(10, 6))

    # Select interesting multipliers for plotting
    plot_multipliers = [-1.0, 0.0, 1.0, 2.0, 5.0]
    plot_multipliers = [m for m in plot_multipliers if m in multipliers]

    for mult in plot_multipliers:
        ratios = []
        plot_layers = []

        for layer in layers:
            if (layer, mult) in results:
                ratio = results[(layer, mult)]['mean_ratio']
                if str(ratio) != 'nan':
                    ratios.append(ratio)
                    plot_layers.append(layer)

        if ratios:
            plt.plot(plot_layers, ratios, marker='o', linewidth=2,
                    markersize=8, label=f'Multiplier {mult}')

    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal probability')
    plt.xlabel('Layer')
    plt.ylabel('Mean prob(matching)/prob(not_matching) Ratio')
    plt.title('Utilitarian Behavior: Ratio Trends Across Layers', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('utilitarian_layer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiplier_effects(results, layers, multipliers):
    """Plot effects of different multipliers for each layer"""
    set_plotting_settings()

    fig, axes = plt.subplots(1, len(layers), figsize=(15, 5))
    if len(layers) == 1:
        axes = [axes]

    for idx, layer in enumerate(layers):
        ax = axes[idx]

        ratios = []
        plot_multipliers = []

        for mult in multipliers:
            if (layer, mult) in results:
                ratio = results[(layer, mult)]['mean_ratio']
                if str(ratio) != 'nan':
                    ratios.append(ratio)
                    plot_multipliers.append(mult)

        if ratios:
            ax.plot(plot_multipliers, ratios, marker='o', linewidth=2,
                   markersize=8, color='#377eb8')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7,
                      label='Equal probability')

        ax.set_xlabel('Multiplier')
        ax.set_ylabel('Mean Ratio')
        ax.set_title(f'Layer {layer}')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend()

    plt.suptitle('Utilitarian Behavior: Multiplier Effects by Layer', fontsize=16)
    plt.tight_layout()
    plt.savefig('utilitarian_multiplier_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(results, layers, multipliers):
    """Save detailed results to CSV files"""

    # Detailed results
    with open('utilitarian_analysis_detailed.csv', 'w') as f:
        f.write("layer,multiplier,mean_ratio,std_ratio,range_ratio,count,mean_matching_prob,mean_not_matching_prob,matching_is_a_count,matching_is_b_count\n")
        for (layer, mult), stats in results.items():
            f.write(f"{layer},{mult},{stats['mean_ratio']},{stats['std_ratio']},{stats['range_ratio']},"
                   f"{stats['count']},{stats['mean_matching_prob']},{stats['mean_not_matching_prob']},"
                   f"{stats['matching_is_a_count']},{stats['matching_is_b_count']}\n")

    # Pivot table format
    with open('utilitarian_analysis_pivot.csv', 'w') as f:
        # Header
        f.write("layer")
        for mult in multipliers:
            f.write(f",{mult}")
        f.write("\n")

        # Data rows
        for layer in layers:
            f.write(f"{layer}")
            for mult in multipliers:
                if (layer, mult) in results:
                    ratio = results[(layer, mult)]['mean_ratio']
                    if str(ratio) != 'nan':
                        f.write(f",{ratio:.6f}")
                    else:
                        f.write(",NaN")
                else:
                    f.write(",")
            f.write("\n")

def main():
    print("Analyzing utilitarian results...")
    print("=" * 50)

    # Analyze the results
    results, layers, multipliers = analyze_utilitarian_results()

    if not results:
        print("No results found!")
        return

    print(f"Processed {len(results)} combinations")
    print(f"Layers: {layers}")
    print(f"Multipliers: {multipliers}")

    # Show behavior distribution
    print_behavior_distribution(results)

    # Print the table
    print_table(results, layers, multipliers)

    # Create visualizations
    print("\nCreating visualizations...")
    create_heatmap(results, layers, multipliers)
    plot_layer_comparison(results, layers, multipliers)
    plot_multiplier_effects(results, layers, multipliers)

    # Save detailed results
    save_detailed_results(results, layers, multipliers)

    # Show summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    valid_means = [stats['mean_ratio'] for stats in results.values()
                   if str(stats['mean_ratio']) != 'nan']

    if valid_means:
        overall_mean = sum(valid_means) / len(valid_means)
        overall_std = (sum((x - overall_mean) ** 2 for x in valid_means) / len(valid_means)) ** 0.5

        print(f"Overall mean ratio: {overall_mean:.4f}")
        print(f"Overall std ratio: {overall_std:.4f}")
        print(f"\nInterpretation:")
        print(f"  - Ratio > 1.0: Model prefers matching behavior (utilitarian)")
        print(f"  - Ratio < 1.0: Model prefers non-matching behavior (non-utilitarian)")
        print(f"  - Ratio = 1.0: Equal preference")

        # Find min/max
        max_ratio = max(valid_means)
        min_ratio = min(valid_means)

        # Find which layer/multiplier has min/max
        max_key = None
        min_key = None
        for key, stats in results.items():
            if stats['mean_ratio'] == max_ratio:
                max_key = key
            if stats['mean_ratio'] == min_ratio:
                min_key = key

        if max_key:
            print(f"\nHighest ratio: {max_ratio:.4f} (Layer {max_key[0]}, Multiplier {max_key[1]})")
        if min_key:
            print(f"Lowest ratio: {min_ratio:.4f} (Layer {min_key[0]}, Multiplier {min_key[1]})")

    print(f"\nFiles saved:")
    print(f"  - utilitarian_analysis_detailed.csv")
    print(f"  - utilitarian_analysis_pivot.csv")
    print(f"  - utilitarian_heatmap.png")
    print(f"  - utilitarian_layer_comparison.png")
    print(f"  - utilitarian_multiplier_effects.png")

if __name__ == "__main__":
    main()
