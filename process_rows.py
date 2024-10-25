import cv2
import numpy as np
import os
from PIL import Image

'''
TODO :
  - Some resulting sprites are more like particles from a main sprite. Find a way to filter out these particles and include them in the main sprite.
  - When finding rows that are less than 32 pixels in height, this means that the subsequent rows are part of the same animation.
    Find a way to group these rows together.
    - Algo : Loop through the countours by x position instead of size.
'''

def find_next_standard_size(size, standards=[32, 64, 128, 256, 512]):
    """Find the next standard size greater than or equal to the given size."""
    for standard in standards:
        if size <= standard:
            return standard
    return standards[-1]  # Default to the largest standard if exceeded

def pad_sprite(sprite, target_width, target_height):
    """Pads the sprite to match the target width and height."""
    h, w, _ = sprite.shape
    # Calculate padding for centering the sprite
    top_padding = (target_height - h) // 2
    bottom_padding = target_height - h - top_padding
    left_padding = (target_width - w) // 2
    right_padding = target_width - w - left_padding

    # Pad the sprite to the target dimensions
    padded_sprite = cv2.copyMakeBorder(
        sprite, top_padding, bottom_padding, left_padding, right_padding,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)  # Transparent padding for RGBA
    )
    return padded_sprite

def merge_intersecting_boxes(contours, x_margin=1, y_margin=1):
    """Merges intersecting bounding boxes within the same row, with an acceptable margin in X and Y directions."""
    merged_boxes = []

    # Sort contours by their x position
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Initialize the first bounding box
    x, y, w, h = cv2.boundingRect(contours[0])
    current_box = [x - x_margin, y - y_margin, x + w + x_margin, y + h + y_margin]  # Expanded current box with margins

    for contour in contours[1:]:
        x, y, w, h = cv2.boundingRect(contour)
        box = [x - x_margin, y - y_margin, x + w + x_margin, y + h + y_margin]  # Expanded box with margins

        # Check if the current box intersects with the new box (with margins)
        if not (box[0] > current_box[2] or box[2] < current_box[0] or
                box[1] > current_box[3] or box[3] < current_box[1]):
            # Merge the boxes by expanding the current_box
            current_box[0] = min(current_box[0], box[0])
            current_box[1] = min(current_box[1], box[1])
            current_box[2] = max(current_box[2], box[2])
            current_box[3] = max(current_box[3], box[3])
        else:
            # Finalize the current box without margins and start a new one
            merged_boxes.append([current_box[0] + x_margin, current_box[1] + y_margin, 
                                 current_box[2] - x_margin, current_box[3] - y_margin])
            current_box = box

    # Append the last current box without margins
    merged_boxes.append([current_box[0] + x_margin, current_box[1] + y_margin, 
                         current_box[2] - x_margin, current_box[3] - y_margin])
    return merged_boxes


def process_rows(image_path, output_image_path='final_spritesheet.png', target_standards=[32, 64, 128, 256, 512], separation=10, ignore_height=32):
    """Processes the rows and generates a single image with standard width and height padding for all sprites."""
    # Load the image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if image has an alpha channel (transparency)
    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]  # Get the alpha channel
        pixel_sums = np.sum(alpha_channel, axis=1)  # Sum the alpha values across each row
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        pixel_sums = np.sum(thresh, axis=1)  # Sum the binary values across each row

    # Identify the lines by checking where the pixel sum is zero (indicating a blank line)
    lines = []
    in_line = False
    for i, pixel_sum in enumerate(pixel_sums):
        if pixel_sum > 0 and not in_line:
            line_start = i
            in_line = True
        elif pixel_sum == 0 and in_line:
            line_end = i + 1  # Include the buffer pixel
            in_line = False
            lines.append((line_start, line_end))

    # Find the largest width and height in all rows
    largest_width, largest_height = 0, 0
    row_sprites = []

    for line_num, (start_y, end_y) in enumerate(lines):
        line_height = end_y - start_y

        # Ignore rows less than 32 pixels in height
        if line_height < ignore_height:
            continue

        line_image = image[start_y:end_y, :]
        gray_line = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if line_image.shape[2] == 3 else cv2.cvtColor(line_image, cv2.COLOR_BGRA2GRAY)
        _, line_thresh = cv2.threshold(gray_line, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(line_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Merge intersecting bounding boxes in the row
        merged_boxes = merge_intersecting_boxes(contours)

        row_sprites.append([])

        for box in merged_boxes:
            x_min, y_min, x_max, y_max = box
            w, h = x_max - x_min, y_max - y_min

            # Ignore small merged boxes
            if w < 15 or h < 15:
                continue

            sprite = line_image[y_min:y_max, x_min:x_max]
            row_sprites[-1].append(sprite)

            # Update largest width and height found
            largest_width = max(largest_width, w)
            largest_height = max(largest_height, h)

    # Determine the standard width and height based on the largest found dimensions
    standard_width = find_next_standard_size(largest_width, target_standards)
    standard_height = find_next_standard_size(largest_height, target_standards)

    # Prepare the final image with equidistant padding
    final_sprites = []

    for row in row_sprites:
        if not row:
            continue

        # Create a new row with padded sprites
        padded_row = []
        for sprite in row:
            padded_sprite = pad_sprite(sprite, standard_width, standard_height)
            padded_row.append(padded_sprite)

        # Concatenate the sprites horizontally with separation
        row_image = np.concatenate(
            [np.pad(sprite, ((0, 0), (0, separation), (0, 0)), 'constant') for sprite in padded_row[:-1]] + [padded_row[-1]],
            axis=1
        )
        final_sprites.append(row_image)

    # Determine the maximum width of all rows
    max_row_width = max(row.shape[1] for row in final_sprites)

    # Pad each row to the maximum width
    padded_rows = [
        np.pad(row, ((0, 0), (0, max_row_width - row.shape[1]), (0, 0)), 'constant')
        for row in final_sprites
    ]

    # Concatenate all rows vertically
    final_image = np.concatenate(padded_rows, axis=0)

    # Save the final image
    final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGRA2RGBA))
    final_image_pil.save(output_image_path)

    print(f"Final image saved as '{output_image_path}'.")



# Example usage
image_path = 'mario_all.png'
process_rows(image_path, output_image_path='final_spritesheet.png', ignore_height=5)
