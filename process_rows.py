import cv2
import numpy as np
import os
from PIL import Image

'''
TODO :
  - Some resulting sprites are more like particles from a main sprite. Find a way to filter out these particles and include them in the main sprite.
  - When finding rows that are less than 32 pixels in height, this means that the subsequent rows are part of the same animation.
    Find a way to group these rows together.
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

def process_rows(image_path, output_image_path='final_spritesheet.png', target_standards=[32, 64, 128, 256, 512], separation=10):
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
        if line_height < 32:
            continue

        line_image = image[start_y:end_y, :]
        gray_line = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY) if line_image.shape[2] == 3 else cv2.cvtColor(line_image, cv2.COLOR_BGRA2GRAY)
        _, line_thresh = cv2.threshold(gray_line, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(line_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        row_sprites.append([])

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            sprite = line_image[y:y+h, x:x+w]
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
image_path = 'adventure_time.png'
process_rows(image_path, output_image_path='final_spritesheet.png')
