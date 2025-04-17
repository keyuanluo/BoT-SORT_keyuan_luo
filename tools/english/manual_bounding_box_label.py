import cv2
import os
import csv
import pandas as pd

# Set the path to the image folder to annotate
image_folder = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/sunny 2/01_14_2023_12_05_13_scene8_stop_cross_close'  # Modify to your image folder path

# Make output_csv reside in the same folder as the input
output_csv = os.path.join(image_folder, 'points_coordinates.csv')  # Output CSV file path

# Path to the reshaped CSV file
reshaped_csv = os.path.join(image_folder, 'reshaped_points_coordinates.csv')

# Create an empty list to store annotation results
annotations = []

# Define mouse click callback function
def mouse_callback(event, x, y, flags, param):
    global annotations, image_name, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Clicked at position ({x}, {y}) in image {image_name}')
        annotations.append({'image_name': image_name, 'x': x, 'y': y})
        # Draw a smaller red circle on the image to indicate the click position
        cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow(window_name, img_display)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right‑click to delete the annotation point closest to the current position
        current_annotations = [ann for ann in annotations if ann['image_name'] == image_name]
        if current_annotations:
            # Calculate distance from all annotation points to the click position
            distances = [(idx, (ann['x'] - x) ** 2 + (ann['y'] - y) ** 2) for idx, ann in enumerate(current_annotations)]
            # Find the nearest annotation point
            min_idx, min_dist = min(distances, key=lambda t: t[1])
            # Remove this annotation point from annotations
            removed_annotation = current_annotations[min_idx]
            annotations.remove(removed_annotation)
            print(f'Removed annotation at ({removed_annotation["x"]}, {removed_annotation["y"]}) in image {image_name}')
            # Redraw the image
            img_display = img.copy()
            cv2.putText(img_display, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Redraw the remaining annotation points
            for ann in annotations:
                if ann['image_name'] == image_name:
                    cv2.circle(img_display, (ann['x'], ann['y']), 3, (0, 0, 255), -1)
            cv2.imshow(window_name, img_display)
        else:
            print('No annotation points to delete in the current image.')

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
image_files.sort()  # Sort filenames

# Create window and set to full‑screen
window_name = 'Image'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_callback)

is_fullscreen = True  # Full‑screen status flag

current_image_index = 0  # Index of the current image
should_exit = False  # Flag to control whether the program exits

while 0 <= current_image_index < len(image_files):
    image_file = image_files[current_image_index]
    image_path = os.path.join(image_folder, image_file)
    image_name = image_file
    img = cv2.imread(image_path)
    if img is None:
        print(f'Failed to read image: {image_path}')
        current_image_index += 1
        continue

    # Display image name
    img_display = img.copy()
    cv2.putText(img_display, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Redraw already annotated points
    for annotation in annotations:
        if annotation['image_name'] == image_name:
            cv2.circle(img_display, (annotation['x'], annotation['y']), 3, (0, 0, 255), -1)

    print(f'Annotating image: {image_name}')
    print('Left-click on the image to record point coordinates.')
    print('Right-click to delete the nearest annotation point.')
    print('Press "d" or right arrow to move to the next image, "a" or left arrow to go back, "s" to save annotations, "m" to toggle fullscreen/windowed mode, "Esc" to exit.')

    while True:
        # Display image
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('d') or key == 83:  # Press 'd' or right arrow to advance to next image
            if current_image_index < len(image_files) - 1:
                current_image_index += 1
            else:
                print("Already at the last image; cannot advance.")
            break
        elif key == ord('a') or key == 81:  # Press 'a' or left arrow to go back to previous image
            if current_image_index > 0:
                current_image_index -= 1
            else:
                print("Already at the first image; cannot go back further.")
            break
        elif key == ord('m'):  # Press 'm' to toggle full‑screen/windowed mode
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                is_fullscreen = False
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                is_fullscreen = True
        elif key == 27:  # Press 'Esc' to exit the program
            should_exit = True
            break  # Exit inner loop
        elif key == ord('r'):  # Press 'r' to undo the last annotation point
            # Undo the last annotation point of the current image
            current_annotations = [ann for ann in annotations if ann['image_name'] == image_name]
            if current_annotations:
                last_annotation = current_annotations[-1]
                annotations.remove(last_annotation)
                print(f'Undid annotation at ({last_annotation["x"]}, {last_annotation["y"]}) in image {image_name}')
                # Redraw the image
                img_display = img.copy()
                cv2.putText(img_display, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                for ann in annotations:
                    if ann['image_name'] == image_name:
                        cv2.circle(img_display, (ann['x'], ann['y']), 3, (0, 0, 255), -1)
                cv2.imshow(window_name, img_display)
            else:
                print('No annotation points to undo in the current image.')
        elif key == ord('s'):  # Press 's' to save annotation results
            # Save annotation results to CSV file
            with open(output_csv, 'w', newline='') as csvfile:
                fieldnames = ['image_name', 'x', 'y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for annotation in annotations:
                    writer.writerow(annotation)
            print(f'Annotation results have been saved to {output_csv}')

            # Perform data reshaping
            df = pd.read_csv(output_csv)
            df['point_index'] = df.groupby('image_name').cumcount() + 1

            # Reshape data and reorder columns as x1, y1, x2, y2, ...
            df_wide = df.pivot(index='image_name', columns='point_index', values=['x', 'y'])
            df_wide = df_wide.swaplevel(axis=1).sort_index(axis=1, level=0)  # Reorder columns to x1, y1, x2, y2
            df_wide.columns = [f'{coord}{idx}' for idx, coord in df_wide.columns]
            df_wide.fillna('', inplace=True)
            df_wide.reset_index(inplace=True)

            # Save to CSV file
            df_wide.to_csv(reshaped_csv, index=False)
            print(f'Data reshaped and saved to {reshaped_csv}')

    if should_exit:
        break  # Exit the outer loop

# Close all windows
cv2.destroyAllWindows()

# Save annotation results before program ends
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['image_name', 'x', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for annotation in annotations:
        writer.writerow(annotation)
print(f'Annotation completed; results saved to {output_csv}')
