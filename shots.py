import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def extract_shot_events(event_data):
    """
    Extract shot events from a match event data
    Returns a list of dictionaries containing shot information
    """
    shots = []
    for event in event_data:
        if event.get('type', {}).get('name') == 'Shot':
            # We'll expand this function later to extract more shot features
            shots.append(event)
    return shots

def process_event_files():
    # Path to events directory
    events_dir = Path(__file__).parent / 'data' / 'data' / 'events'
    all_shots = []
    
    # Get list of all JSON files in events directory
    event_files = list(events_dir.glob('*.json'))
    
    # Iterate through each event file with progress bar
    for event_file in tqdm(event_files, desc="Processing match events"):
        try:
            with open(event_file, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Extract shots from this match
            match_shots = extract_shot_events(match_data)
            
            # Add match_id to each shot event
            match_id = event_file.stem  # filename without extension
            for shot in match_shots:
                shot['match_id'] = match_id
            
            all_shots.extend(match_shots)
            
        except Exception as e:
            print(f"Error processing {event_file}: {str(e)}")
    
    return all_shots

if __name__ == "__main__":
    # Process all event files and collect shots
    shots_data = process_event_files()
    
    # Convert to DataFrame
    shots_df = pd.DataFrame(shots_data)
    
    # Basic info about collected data
    print(f"\nTotal shots collected: {len(shots_df)}")
    
    # Save to CSV (we'll improve this later)
    shots_df.to_csv('shots_data.csv', index=False)
    print("Data saved to shots_data.csv")
