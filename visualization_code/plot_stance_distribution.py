import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def load_stance_data(directory):
    """Load all stance values from JSON files in the given directory."""
    stance_values = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    # Load stance values from each file
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Each file contains a list of records
                if isinstance(data, list):
                    for record in data:
                        if 'stance_5pt' in record:
                            stance_values.append(record['stance_5pt'])
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return stance_values

def plot_stance_distribution(stance_values, output_path=None):
    """Plot the distribution of stance values."""
    
    # Count occurrences of each stance value
    stance_counts = Counter(stance_values)
    
    # Sort by stance value
    sorted_stances = sorted(stance_counts.items())
    stances = [s[0] for s in sorted_stances]
    counts = [s[1] for s in sorted_stances]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(stances, counts, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Stance Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Stance Distribution from stance_labeled/512', fontsize=14, fontweight='bold')
    ax.set_xticks(stances)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.savefig('/mnt/d/flow_of_argument/stance_distribution_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to /mnt/d/flow_of_argument/stance_distribution_plot.png")
    
    # Print statistics
    print("\n" + "="*50)
    print("STANCE DISTRIBUTION STATISTICS")
    print("="*50)
    print(f"Total records: {len(stance_values)}")
    print(f"Unique stance values: {sorted(set(stance_values))}")
    print("\nDistribution:")
    for stance, count in sorted_stances:
        percentage = (count / len(stance_values)) * 100
        print(f"  Stance {stance}: {count:6d} ({percentage:6.2f}%)")
    print("="*50)

if __name__ == "__main__":
    # Path to the stance_labeled/512 directory
    data_dir = '/mnt/d/flow_of_argument/data/stance_labeled/512'
    
    # Load stance data
    stance_values = load_stance_data(data_dir)
    
    if stance_values:
        # Plot the distribution
        plot_stance_distribution(stance_values, output_path="results/stance_distribution_plot.png")
    else:
        print("No stance data found!")
