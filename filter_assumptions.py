import json
import os
from pathlib import Path
from typing import List, Dict
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assumption-filter")

class AssumptionFilter:
    def __init__(self):
        # Problematic patterns that indicate low quality
        self.low_quality_patterns = [
            r"^(the|this|that|there|it)\s+(?:is|was|are|were)\s+\w+$",
            r"^\d+\.\s*\w+$",
        ]
        
        self.min_words = 4
        self.max_words = 50

    def is_valid_assumption(self, assumption: str) -> bool:
        """Check if an assumption meets quality criteria."""
        assumption = assumption.strip().strip('.,')
        
        if not assumption:
            return False
            
        word_count = len(assumption.split())
        if word_count < self.min_words or word_count > self.max_words:
            return False
            
        # Check for low-quality patterns
        for pattern in self.low_quality_patterns:
            if re.match(pattern, assumption, re.IGNORECASE):
                return False
        
        # Check for overly generic single words in parentheses
        if re.match(r'^\([a-z]+\)$', assumption, re.IGNORECASE):
            return False
            
        return True

    def filter_assumptions(self, assumptions: List[str]) -> List[str]:
        """Filter a list of assumptions, removing low-quality ones."""
        filtered = []
        for assumption in assumptions:
            # Handle lists that might be nested or have extra formatting
            if isinstance(assumption, (list, tuple)):
                assumption = ' '.join(str(x) for x in assumption)
            
            # Convert to string in case it's not
            assumption = str(assumption).strip()
            
            if self.is_valid_assumption(assumption):
                filtered.append(assumption)
                
        return filtered

def process_files(input_dir: Path, output_dir: Path):
    """Process all JSON files in the input directory."""
    filter = AssumptionFilter()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for json_file in input_dir.glob('*.json'):
        log.info(f"Processing {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_assumptions = 0
        filtered_assumptions = 0
        
        # Process each turn
        for turn in data:
            if 'assumptions' in turn:
                original_assumptions = turn['assumptions']
                total_assumptions += len(original_assumptions)
                
                # Filter assumptions
                turn['assumptions'] = filter.filter_assumptions(original_assumptions)
                filtered_assumptions += len(turn['assumptions'])
        
        # Save filtered results
        output_file = output_dir / f"filtered_{json_file.name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        log.info(f"Filtered {total_assumptions - filtered_assumptions} low-quality assumptions "
                f"({filtered_assumptions}/{total_assumptions} remaining)")

def main():
    input_dir = Path('results/covid')
    output_dir = Path('results/covid_filtered')
    
    if not input_dir.exists():
        log.error(f"Input directory {input_dir} does not exist!")
        return
        
    process_files(input_dir, output_dir)
    log.info("Filtering complete. Results saved in results/covid_filtered/")

if __name__ == "__main__":
    main()
