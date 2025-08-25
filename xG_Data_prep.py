import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ast import literal_eval


df_shots = pd.read_csv('shots_data.csv')

df_shots['shot'] = df_shots['shot'].apply(literal_eval)

# Filter for shots from open play
df_shots_open_play = df_shots[
    df_shots['shot'].apply(lambda x: (x.get('type', {}).get('name') == 'Open Play'))
]

# Replace NaN values with False in under_pressure and out columns(Remember to drop 'out' later)

df_shots_open_play['under_pressure'] = df_shots_open_play['under_pressure'].fillna(False)
df_shots_open_play['out'] = df_shots_open_play['out'].fillna(False)


# x,y extraction

# # Extract start coordinates from location column
df_shots_open_play['start_x'] = df_shots_open_play['location'].apply(lambda x: literal_eval(x)[0] if isinstance(x, str) else x[0])
df_shots_open_play['start_y'] = df_shots_open_play['location'].apply(lambda x: literal_eval(x)[1] if isinstance(x, str) else x[1])

# Extract end coordinates from shot dictionary
df_shots_open_play['end_x'] = df_shots_open_play['shot'].apply(lambda x: x.get('end_location', [0, 0, 0])[0])
df_shots_open_play['end_y'] = df_shots_open_play['shot'].apply(lambda x: x.get('end_location', [0, 0, 0])[1])



# create distance and angle to goal.


# Goal center coordinates
goal_center_x = 120
goal_center_y = 40

# Calculate distance using Euclidean distance formula
df_shots_open_play['distance_to_goal'] = np.sqrt(
    (df_shots_open_play['start_x'] - goal_center_x)**2 + 
    (df_shots_open_play['start_y'] - goal_center_y)**2
)

# Goal posts coordinates
post_1_x, post_1_y = 120, 36  # Bottom post
post_2_x, post_2_y = 120, 44  # Top post

# Calculate vectors from shot to posts
df_shots_open_play['angle_to_goal'] = df_shots_open_play.apply(
    lambda row: np.degrees(
        np.arccos(
            np.dot(
                [post_1_x - row['start_x'], post_1_y - row['start_y']],
                [post_2_x - row['start_x'], post_2_y - row['start_y']]
            ) / (
                np.sqrt((post_1_x - row['start_x'])**2 + (post_1_y - row['start_y'])**2) *
                np.sqrt((post_2_x - row['start_x'])**2 + (post_2_y - row['start_y'])**2)
            )
        )
    ),
    axis=1
)



# extract first_time, technique, body part, and player_pattern

df_shots_open_play['first_time'] = df_shots_open_play['shot'].apply(lambda x: x.get('first_time', False))
df_shots_open_play['technique'] = df_shots_open_play['shot'].apply(lambda x: x.get('technique', {}).get('name', ''))
df_shots_open_play['body_part'] = df_shots_open_play['shot'].apply(lambda x: x.get('body_part', {}).get('name', ''))

df_shots_open_play['play_pattern_name'] = df_shots_open_play['play_pattern'].apply(
    lambda x: literal_eval(x)['name'] if isinstance(x, str) else x['name']
)


# from freeze_frame: create number of opponents/teammates between the ball and goal

def is_between_shot_and_goal(player_loc, shot_loc, goal_center=(120, 40), buffer=10):
    """Check if player is in a cone between shot and goal"""
    # Convert to numpy arrays for vector operations
    player = np.array(player_loc)
    shot = np.array(shot_loc)
    goal = np.array(goal_center)
    
    # Vector from shot to goal and its length
    shot_to_goal = goal - shot
    shot_to_goal_length = np.linalg.norm(shot_to_goal)
    
    # Vector from shot to player and its length
    shot_to_player = player - shot
    shot_to_player_length = np.linalg.norm(shot_to_player)
    
    # If player is behind shot, ignore
    if shot_to_player_length == 0 or np.dot(shot_to_goal, shot_to_player) < 0:
        return False
        
    # Calculate perpendicular distance to shot-goal line
    cross_product = np.abs(np.cross(shot_to_goal, shot_to_player))
    perpendicular_distance = cross_product / shot_to_goal_length
    
    return perpendicular_distance < buffer

# Extract defensive coverage features
def get_defensive_coverage(row):
    shot_loc = [row['start_x'], row['start_y']]
    freeze_frame = row['shot'].get('freeze_frame', [])
    
    defenders_between = 0
    teammates_between = 0
    goalkeeper_in_frame = False
    
    for player in freeze_frame:
        if is_between_shot_and_goal(player['location'], shot_loc):
            if player['teammate']:
                teammates_between += 1
            else:
                if player['position']['name'] == 'Goalkeeper':
                    goalkeeper_in_frame = True
                defenders_between += 1
    
    return pd.Series({
        'defenders_in_path': defenders_between,
        'teammates_in_path': teammates_between,
        'goalkeeper_in_path': goalkeeper_in_frame
    })

# Apply the function to create new features
df_shots_open_play[['defenders_in_path', 'teammates_in_path', 'goalkeeper_in_path']] = \
    df_shots_open_play.apply(get_defensive_coverage, axis=1)




df_shots_open_play['is_goal'] = df_shots_open_play['shot'].apply(
    lambda x: x.get('outcome', {}).get('name') == 'Goal'
)



columns = ['period','possession','start_x','start_y','end_x','end_y','play_pattern_name','is_goal',
           'duration', 'distance_to_goal','angle_to_goal','first_time',
           'technique', 'body_part', 'defenders_in_path', 'teammates_in_path',
           'goalkeeper_in_path','under_pressure']

df_cleaned_shots = df_final = df_shots_open_play[columns].copy()
df_cleaned_shots.to_csv('shots_data_clean.csv', index=False)

