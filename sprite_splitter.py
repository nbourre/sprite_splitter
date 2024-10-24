import cv2
import numpy as np
from PIL import Image
import os
import json

def find_sprites(image_path, output_folder='data'):
    # Load the image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert image to grayscale and apply threshold to find the contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours of each separate sprite
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a folder for the separated images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # JSON to store all sprites' positions
    sprite_positions = {}

    for i, contour in enumerate(contours):
        # Get bounding box for each contour (minimum enclosing rectangle)
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the sprite using numpy slicing
        sprite = image[y:y+h, x:x+w]

        # Create the PIL image and save with a suffix based on coordinates and size
        sprite_image = Image.fromarray(cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA))
        file_name = f'sprite_{x}_{y}_{w}x{h}.png'
        sprite_image.save(os.path.join(output_folder, file_name))

        # Store each sprite's position in the JSON dictionary
        sprite_positions[file_name] = {
            'x': x,
            'y': y,
            'width': w,
            'height': h
        }

    # Save all positions to a JSON file
    with open(os.path.join(output_folder, 'sprites.json'), 'w') as json_file:
        json.dump(sprite_positions, json_file, indent=4)

    print(f"Sprites extracted and saved to '{output_folder}'.")

def pad_to_closest_height(image, target_heights=[32, 48, 64, 128]):
    """ Pads the image vertically to match the closest target height """
    current_height = image.shape[0]
    closest_height = min(target_heights, key=lambda h: abs(h - current_height))
    
    if closest_height > current_height:
        padding_needed = closest_height - current_height
        # Add padding evenly at the top and bottom
        top_padding = padding_needed // 2
        bottom_padding = padding_needed - top_padding
        image = cv2.copyMakeBorder(image, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    
    return image, closest_height

def separate_lines(image_path, output_folder='data', target_heights=[32, 48, 64, 128]):
    # Load the image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if image has an alpha channel (transparency)
    if image.shape[2] == 4:  # Assuming the image has 4 channels (RGBA)
        alpha_channel = image[:, :, 3]  # Get the alpha channel
        pixel_sums = np.sum(alpha_channel, axis=1)  # Sum the alpha values across each row
    else:
        # If the image does not have an alpha channel, use grayscale to detect blank lines
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

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract each line as a separate image
    for line_num, (start_y, end_y) in enumerate(lines):
        line_image = image[start_y:end_y, :]

        # Add padding to reach the closest target height
        padded_line_image, final_height = pad_to_closest_height(line_image, target_heights)

        # Save the extracted line as an image file with its height in the name
        line_file_name = f'sprite_line_{line_num}_{final_height}px.png'
        line_image_pil = Image.fromarray(cv2.cvtColor(padded_line_image, cv2.COLOR_BGRA2RGBA))
        line_image_pil.save(os.path.join(output_folder, line_file_name))

    print(f"Lines extracted and saved to '{output_folder}'.")
if __name__ == '__main__':
    # Path to your spritesheet image
    spritesheet_path = 'adventure_time.png'
    separate_lines(spritesheet_path)
