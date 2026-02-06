#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import List, Dict, Any

def merge_turns_by_speaker(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive turns by the same speaker.
    Even offsets for Speaker A, odd offsets for Speaker B.
    """
    if not turns:
        return turns
    
    merged_turns = []
    current_turn = None
    
    for turn in turns:
        if current_turn is None:
            # Start a new turn group
            current_turn = turn.copy()
        elif turn.get("speaker_id") == current_turn.get("speaker_id"):
            # Same speaker - merge turns
            # Concatenate turn text
            current_turn["turn_text"] = (
                current_turn.get("turn_text", "") + " " + turn.get("turn_text", "")
            ).strip()
            
            # Merge explicit_propositions
            if "explicit_propositions" in turn:
                if "explicit_propositions" not in current_turn:
                    current_turn["explicit_propositions"] = []
                current_turn["explicit_propositions"].extend(turn["explicit_propositions"])
            
            # Merge assumptions
            if "assumptions" in turn:
                if "assumptions" not in current_turn:
                    current_turn["assumptions"] = []
                current_turn["assumptions"].extend(turn["assumptions"])
        else:
            # Different speaker - save the merged turn and start a new one
            merged_turns.append(current_turn)
            current_turn = turn.copy()
    
    # Don't forget the last turn
    if current_turn is not None:
        merged_turns.append(current_turn)
    
    # Update turn indices: 0 (Speaker A), 1 (Speaker B), 2 (Speaker A), etc.
    for idx, turn in enumerate(merged_turns):
        turn["turn_idx"] = idx
    
    return merged_turns


def process_json_file(file_path: str) -> None:
    """Process a single JSON file and merge turns by speaker."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            turns = json.load(f)
        
        # Merge turns by speaker
        merged_turns = merge_turns_by_speaker(turns)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_turns, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Processed {file_path}: {len(turns)} turns -> {len(merged_turns)} merged turns")
    
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")


def main():
    """Process all JSON files in the stance_labeled/512 directory."""
    stance_labeled_dir = "/mnt/d/flow_of_argument/data/stance_labeled/512"
    
    if not os.path.isdir(stance_labeled_dir):
        print(f"Error: Directory not found: {stance_labeled_dir}")
        return
    
    # Get all JSON files
    json_files = sorted(Path(stance_labeled_dir).glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {stance_labeled_dir}")
        return
    
    print(f"Processing {len(json_files)} JSON files...\n")
    
    for json_file in json_files:
        process_json_file(str(json_file))
    
    print(f"\n✓ All files processed!")


if __name__ == "__main__":
    main()
