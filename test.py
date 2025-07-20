import gzip
import json

cluster_speaker_turn_path = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/speakerTurnData.jsonl.gz'
local_speaker_turn_path = 'data/speakerTurnData.jsonl.gz'
cluster_episode_level_path = '/shared/3/datasets/podcasts/SPoRC/processed/mayJune/v1/episodeLevelData.jsonl.gz'
local_episode_level_path = 'data/episodeLevelData.jsonl.gz'

def print_sample(file_path):
    """
    Opens a gzipped JSONL file, reads the first two lines,
    parses each JSON line, and prints them in a pretty format.
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        line = f.readline().strip()
        if not line:
            print("No more lines to read.")
        
        try:
            sample = json.loads(line)
            print(f"Sample:")
            print(json.dumps(sample, indent=4, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON line: {e}")
            
            
print_sample(cluster_episode_level_path)


from itertools import islice

with open("doc_topics.txt", "r", encoding="utf-8") as f:
    for line in islice(f, 5):
        print(line.rstrip())
