import pandas as pd
import numpy as np
import os  # New import


def fill_missing_frames(df, track_id_list=None):
    """Fill missing frames for specified track IDs by linear interpolation."""

    # Extract frame numbers, assuming image_name format is '000370.png'
    df['frame_number'] = df['image_name'].str.extract('(\d+)').astype(int)

    # If track_id_list is not specified, process all track_ids
    if track_id_list is None:
        track_id_list = df['track_id'].unique()

    # DataFrame to hold all filled data
    filled_df = pd.DataFrame()

    for track_id in track_id_list:
        # Filter data of the current track_id and sort by frame_number
        track_df = df[df['track_id'] == track_id].sort_values('frame_number').reset_index(drop=True)

        # DataFrame to store filled data for the current track_id
        track_filled_df = pd.DataFrame()

        for idx in range(len(track_df) - 1):
            current_row = track_df.loc[idx]
            next_row = track_df.loc[idx + 1]

            # Compute difference between current and next frame
            frame_diff = next_row['frame_number'] - current_row['frame_number']

            # Append the current row to the filled data
            track_filled_df = track_filled_df.append(current_row, ignore_index=True)

            # If frame_diff > 1, there are missing frames that need interpolation
            if frame_diff > 1:
                for gap in range(1, frame_diff):
                    # Interpolation ratio
                    ratio = gap / frame_diff

                    # Generate new frame number and name
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

                    # Append interpolated row
                    track_filled_df = track_filled_df.append(interpolated_row, ignore_index=True)

        # Append the last row of this track
        track_filled_df = track_filled_df.append(track_df.iloc[-1], ignore_index=True)

        # Concatenate to overall filled_df
        filled_df = filled_df.append(track_filled_df, ignore_index=True)

    # Sort by track_id and frame_number
    filled_df = filled_df.sort_values(['track_id', 'frame_number']).reset_index(drop=True)

    # =============== Validation: ensure x2 >= x1 & y2 >= y1 ===============
    invalid_mask = (filled_df['x2'] < filled_df['x1']) | (filled_df['y2'] < filled_df['y1'])
    invalid_rows = filled_df[invalid_mask]
    if not invalid_rows.empty:
        print("\n[Warning] The following rows have invalid BBoxes (x2 < x1 or y2 < y1):")
        print(invalid_rows[['image_name', 'track_id', 'x1', 'y1', 'x2', 'y2', 'score']])
    # =========================================================

    # =============== Deduplication: handle duplicate rows with the same (track_id, image_name) ===============
    # Method A: simply drop duplicates and keep the first row
    filled_df = filled_df.drop_duplicates(subset=['track_id', 'image_name'], keep='first')

    # Method B: keep the row with the highest score (uncomment if needed)
    # filled_df = (
    #     filled_df.sort_values('score', ascending=False)
    #              .groupby(['track_id', 'image_name'], as_index=False)
    #              .head(1)
    # )
    # filled_df = filled_df.sort_values(['track_id', 'image_name']).reset_index(drop=True)
    # =========================================================

    # Remove the auxiliary frame_number column
    filled_df = filled_df.drop(columns=['frame_number'])

    return filled_df


# =============== Example usage ===============
if __name__ == "__main__":
    # Read the Excel file
    input_excel = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_video_number/video_0016/detections_过渡.xlsx'  # Replace with your Excel filename
    df = pd.read_excel(input_excel)

    # -------- Key step: set ALL track_id values to 1 BEFORE processing --------
    df['track_id'] = 1

    filled_df = fill_missing_frames(df)

    # Directory of the input file
    input_dir = os.path.dirname(input_excel)

    # Output file path in the same directory
    output_excel = os.path.join(input_dir, 'detections_最终.xlsx')  # Output filename ('最终' kept intentionally)

    # Save the filled data to a new Excel file
    filled_df.to_excel(output_excel, index=False)

    print(f'\nFilled data has been saved to {output_excel}')
