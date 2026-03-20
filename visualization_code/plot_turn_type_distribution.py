import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def load_turn_type_data(directory):
    """Load all turn type labels from JSON files in the given directory."""
    turn_type_labels = []

    json_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.json'):
                json_files.append(os.path.join(root, file_name))
    print(f"Found {len(json_files)} JSON files in {directory}")

    # Load turn type labels from each file
    for file_path in json_files:
        json_file = os.path.relpath(file_path, directory)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Each file contains a list of records
                if isinstance(data, list):
                    for record in data:
                        if 'turn_type_label' in record and record['turn_type_label'] is not None:
                            turn_type_labels.append(record['turn_type_label'])
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return turn_type_labels


def plot_turn_type_distribution(turn_type_labels, output_path=None):
    """Plot the distribution of turn type labels."""

    # Count occurrences of each turn type label
    turn_type_counts = Counter(turn_type_labels)

    # Sort by frequency (descending)
    sorted_turn_types = sorted(turn_type_counts.items(), key=lambda x: x[1], reverse=True)
    turn_types = [t[0] for t in sorted_turn_types]
    counts = [t[1] for t in sorted_turn_types]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    bars = ax.bar(range(len(turn_types)), counts, color='steelblue', edgecolor='black', alpha=0.7)

    # Customize the plot
    ax.set_xlabel('Turn Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Turn Type Distribution from turn_type_labeled', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(turn_types)))
    ax.set_xticklabels(turn_types, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()

    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.savefig('/mnt/d/flow_of_argument/turn_type_distribution_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to /mnt/d/flow_of_argument/turn_type_distribution_plot.png")

    plt.show()

    # Print statistics
    print("\n" + "=" * 50)
    print("TURN TYPE DISTRIBUTION STATISTICS")
    print("=" * 50)
    print(f"Total records: {len(turn_type_labels)}")
    print(f"Unique turn types: {len(set(turn_type_labels))}")
    print("\nDistribution (sorted by frequency):")
    for turn_type, count in sorted_turn_types:
        percentage = (count / len(turn_type_labels)) * 100
        print(f"  {turn_type:<20s}: {count:6d} ({percentage:6.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    # Path to the turn_type_labeled directory
    data_dir = "data/turn_type_labeled"
    output_dir = "results"

    # Load turn type data
    turn_type_labels = load_turn_type_data(data_dir)

    if turn_type_labels:
        # Plot the distribution
        os.makedirs(output_dir, exist_ok=True)
        plot_turn_type_distribution(
            turn_type_labels,
            output_path=os.path.join(output_dir, "turn_type_distribution_plot.png")
        )
    else:
        print("No turn type data found!")
