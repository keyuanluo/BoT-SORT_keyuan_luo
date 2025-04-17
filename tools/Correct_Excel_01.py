import pandas as pd
import numpy as np
import os  # Newly added import

# Duplicate imports retained from the original script for consistency
import pandas as pd
import numpy as np
import os  # Newly added import (duplicate)

def fill_missing_frames(df, track_id_list=None):
    """Fill missing frames for specified track IDs by linear interpolation."""
    # Extract frame numbers, assuming image_name format is '000370.png'
    df['frame_number'] = df['image_name'].str.extract('(\\d+)').astype(int)

    # If track_id_list is not specified, process all track_ids
    if track_id_list is None:
        track_id_list = df['track_id'].unique()

    # DataFrame to store all filled data
    filled_df = pd.DataFrame()

    for track_id in track_id_list:
        # Filter current track_id and sort by frame number
        track_df = df[df['track_id'] == track_id].sort_values('frame_number').reset_index(drop=True)

        # DataFrame to store filled data for the current track_id
        track_filled_df = pd.DataFrame()

        for idx in range(len(track_df) - 1):
            current_row = track_df.loc[idx]
            next_row = track_df.loc[idx + 1]

            # Calculate frame difference between current frame and next frame
            frame_diff = next_row['frame_number'] - current_row['frame_number']

            # Add the current row to the filled data
            track_filled_df = track_filled_df.append(current_row, ignore_index=True)

            # If frame_diff > 1, missing frames exist and need interpolation
            if frame_diff > 1:
                for gap in range(1, frame_diff):
                    # Calculate interpolation ratio
                    ratio = gap / frame_diff

                    # Generate new frame number
                    new_frame_number = current_row['frame_number'] + gap
                    new_image_name = '{:06d}.png'.format(new_frame_number)

                    # Linearly interpolate x1, y1, x2, y2, score
                    interpolated_row = {
                        'image_name': new_image_name,
                        'frame_gap': 1,  # Gap for filled frame set to 1
                        'track_id': track_id,
                        'class': current_row['class'],
                        'x1': current_row['x1'] + ratio * (next_row['x1'] - current_row['x1']),
                        'y1': current_row['y1'] + ratio * (next_row['y1'] - current_row['y1']),
                        'x2': current_row['x2'] + ratio * (next_row['x2'] - current_row['x2']),
                        'y2': current_row['y2'] + ratio * (next_row['y2'] - current_row['y2']),
                        'score': current_row['score'] + ratio * (next_row['score'] - current_row['score']),
                        'frame_number': new_frame_number
                    }

                    # Add interpolated row to the filled data
                    track_filled_df = track_filled_df.append(interpolated_row, ignore_index=True)

        # Add the last row (the final frame of each track)
        track_filled_df = track_filled_df.append(track_df.iloc[-1], ignore_index=True)

        # Append the filled data of the current track_id to the total DataFrame
        filled_df = filled_df.append(track_filled_df, ignore_index=True)

    # Sort the filled data by track_id and frame_number
    filled_df = filled_df.sort_values(['track_id', 'frame_number']).reset_index(drop=True)

    # -------------- Validation: ensure x2 >= x1 and y2 >= y1 --------------
    invalid_mask = (filled_df['x2'] < filled_df['x1']) | (filled_df['y2'] < filled_df['y1'])
    invalid_rows = filled_df[invalid_mask]
    if not invalid_rows.empty:
        print("[Warning] The following rows have x2 < x1 or y2 < y1:")
        print(invalid_rows[['image_name', 'track_id', 'x1', 'y1', 'x2', 'y2', 'score']])

    # Drop the auxiliary frame_number column
    filled_df = filled_df.drop(columns=['frame_number'])

    return filled_df


# Example usage:
# Read the Excel file
input_excel = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_video_number/video_0016/detections.xlsx'  # Replace with your Excel filename
df = pd.read_excel(input_excel)

track_id_list = [1, 2]  # Specify the track_id list to process

# Fill missing frames
filled_df = fill_missing_frames(df, track_id_list=track_id_list)

# If you want to process all track_ids, simply call:
# filled_df = fill_missing_frames(df)

# Get the directory of the input file
input_dir = os.path.dirname(input_excel)

# Set the output file path in the same directory as the input file
output_excel = os.path.join(input_dir, 'detections_过渡.xlsx')  # Output filename ("过渡" kept intentionally)

# Save the filled data to a new Excel file
filled_df.to_excel(output_excel, index=False)

print(f'Filled data has been saved to {output_excel}')
