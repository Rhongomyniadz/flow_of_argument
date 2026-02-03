import json
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def load_conversation_moves_data(directory):
    """Load all conversation move labels from JSON files in the given directory."""
    move_labels = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    # Load conversation move labels from each file
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Each file contains a list of records
                if isinstance(data, list):
                    for record in data:
                        if 'conversation_move_label' in record and record['conversation_move_label'] is not None:
                            move_labels.append(record['conversation_move_label'])
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return move_labels

def plot_conversation_moves_distribution(move_labels, output_path=None):
    """Plot the distribution of conversation move labels."""
    
    # Count occurrences of each move label
    move_counts = Counter(move_labels)
    
    # Sort by frequency (descending)
    sorted_moves = sorted(move_counts.items(), key=lambda x: x[1], reverse=True)
    moves = [m[0] for m in sorted_moves]
    counts = [m[1] for m in sorted_moves]
    
    # Create the plot with larger figure size for better readability
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar chart
    bars = ax.bar(range(len(moves)), counts, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Conversation Move', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Conversation Moves Distribution from conversation_moves_labeled', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(moves)))
    ax.set_xticklabels(moves, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.savefig('/mnt/d/flow_of_argument/conversation_moves_distribution_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to /mnt/d/flow_of_argument/conversation_moves_distribution_plot.png")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("CONVERSATION MOVES DISTRIBUTION STATISTICS")
    print("="*60)
    print(f"Total records: {len(move_labels)}")
    print(f"Unique conversation moves: {len(set(move_labels))}")
    print("\nDistribution (sorted by frequency):")
    for move, count in sorted_moves:
        percentage = (count / len(move_labels)) * 100
        print(f"  {move:<40s}: {count:6d} ({percentage:6.2f}%)")
    print("="*60)

if __name__ == "__main__":
    # Path to the conversation_moves_labeled directory
    data_dir = '/mnt/d/flow_of_argument/data/conversation_moves_labeled'
    
    # Load conversation move data
    move_labels = load_conversation_moves_data(data_dir)
    
    if move_labels:
        # Plot the distribution
        plot_conversation_moves_distribution(move_labels, output_path="results/conversation_moves_distribution_plot.png")
    else:
        print("No conversation move data found!")
