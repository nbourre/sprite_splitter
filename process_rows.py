import cv2
import numpy as np
import os
import json
import argparse
from PIL import Image

'''
TODO :
  - Some resulting sprites are more like particles from a main sprite. Find a way to filter out these particles and include them in the main sprite.
  - When finding rows that are less than 32 pixels in height, this means that the subsequent rows are part of the same animation.
    Find a way to group these rows together.
    - Algo : Loop through the countours by x position instead of size.
'''

CONFIG_FILE = 'config.json'

def load_config():
    """Load configuration from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'last_input_file': '',
        'last_magic_color': None,
        'last_color_tolerance': 50,
        'last_ignore_height': 35,
        'last_separation': 0
    }

def save_config(config):
    """Save configuration to JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def parse_color(color_str):
    """Parse color from string (hex or RGB format).
    
    Args:
        color_str: Color as hex (#FF00FF) or RGB (255,0,255 or 255 0 255)
    
    Returns:
        Tuple of (R, G, B) or None if invalid
    """
    color_str = color_str.strip()
    
    # Try hex format
    if color_str.startswith('#'):
        try:
            color_str = color_str[1:]
            if len(color_str) == 6:
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
                return (r, g, b)
        except:
            pass
    
    # Try RGB format (comma or space separated)
    try:
        parts = color_str.replace(',', ' ').split()
        if len(parts) == 3:
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (r, g, b)
    except:
        pass
    
    return None

def format_color(color_tuple):
    """Format color tuple as string for display."""
    if color_tuple is None:
        return "None"
    return f"RGB({color_tuple[0]}, {color_tuple[1]}, {color_tuple[2]}) / #{color_tuple[0]:02X}{color_tuple[1]:02X}{color_tuple[2]:02X}"

def get_user_input(prompt, default=None, validator=None):
    """Get user input with optional default value."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        user_input = input(prompt).strip()
        
        if not user_input and default is not None:
            return default
        
        if not user_input:
            print("Please provide a value.")
            continue
        
        if validator:
            result = validator(user_input)
            if result is None:
                print("Invalid input. Please try again.")
                continue
            return result
        
        return user_input

def interactive_menu():
    """Interactive menu for user input."""
    print("\n" + "="*60)
    print("  SPRITE SPLITTER - Interactive Mode")
    print("="*60 + "\n")
    
    config = load_config()
    
    # Get input filename
    print("--- Input File ---")
    default_input = config.get('last_input_file', '')
    input_file = get_user_input(
        "Enter input filename (relative or absolute path)",
        default=default_input if default_input else None
    )
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        return
    
    # Ask about magic color
    print("\n--- Magic Color Removal ---")
    last_magic = config.get('last_magic_color')
    default_response = 'y' if last_magic else 'n'
    
    has_magic = get_user_input(
        "Does the spritesheet have a background color to remove? (y/n)",
        default=default_response
    ).lower() in ['y', 'yes']
    
    magic_color = None
    color_tolerance = 50
    
    if has_magic:
        print(f"\nLast used magic color: {format_color(last_magic)}")
        print("Enter magic color in RGB format (e.g., '255,0,255' or '255 0 255')")
        print("or HEX format (e.g., '#FF00FF')")
        
        magic_color = get_user_input(
            "Magic color",
            default=format_color(last_magic) if last_magic else None,
            validator=lambda x: parse_color(x) if x != format_color(last_magic) else last_magic
        )
        
        if isinstance(magic_color, str):
            magic_color = last_magic
        
        color_tolerance = int(get_user_input(
            "Color tolerance (0-255, how close colors need to match)",
            default=str(config.get('last_color_tolerance', 50)),
            validator=lambda x: int(x) if x.isdigit() and 0 <= int(x) <= 255 else None
        ))
    
    # Get processing parameters
    print("\n--- Processing Parameters ---")
    
    print("Ignore Height: Rows shorter than this will be skipped (useful to filter out small artifacts)")
    ignore_height = int(get_user_input(
        "Ignore height (pixels)",
        default=str(config.get('last_ignore_height', 35)),
        validator=lambda x: int(x) if x.isdigit() and int(x) >= 0 else None
    ))
    
    print("\nSeparation: Space between sprites in the output (0 for no spacing)")
    separation = int(get_user_input(
        "Separation (pixels)",
        default=str(config.get('last_separation', 0)),
        validator=lambda x: int(x) if x.isdigit() and int(x) >= 0 else None
    ))
    
    # Get output filename
    print("\n--- Output File ---")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    default_output = f"output/{base_name}_output.png"
    
    output_file = get_user_input(
        "Output filename",
        default=default_output
    )
    
    # Ensure .png extension
    if not output_file.lower().endswith('.png'):
        output_file += '.png'
    
    # Save config
    config['last_input_file'] = input_file
    config['last_magic_color'] = magic_color
    config['last_color_tolerance'] = color_tolerance
    config['last_ignore_height'] = ignore_height
    config['last_separation'] = separation
    save_config(config)
    
    # Process the image
    print("\n" + "="*60)
    print("Processing...")
    print("="*60 + "\n")
    
    working_file = input_file
    
    if magic_color:
        working_file = remove_magic_color_from_image(
            input_file,
            magic_color,
            color_tolerance=color_tolerance
        )
    
    process_rows(
        working_file,
        output_image_path=output_file,
        separation=separation,
        ignore_height=ignore_height
    )
    
    print("\n" + "="*60)
    print("  Processing Complete!")
    print("="*60 + "\n")

def find_next_standard_size(size, standards=[32, 64, 128, 256, 512]):
    """Find the next standard size greater than or equal to the given size."""
    for standard in standards:
        if size <= standard:
            return standard
    return standards[-1]  # Default to the largest standard if exceeded

def pad_sprite(sprite, target_width, target_height):
    """Pads the sprite to match the target width and height."""
    h, w, _ = sprite.shape
    
    # If sprite is already larger than or equal to target, return as is or resize
    if h >= target_height and w >= target_width:
        return sprite
    
    # Ensure target dimensions are at least as large as sprite
    actual_target_width = max(target_width, w)
    actual_target_height = max(target_height, h)
    
    # Calculate padding for centering the sprite
    top_padding = (actual_target_height - h) // 2
    bottom_padding = actual_target_height - h - top_padding
    left_padding = (actual_target_width - w) // 2
    right_padding = actual_target_width - w - left_padding

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

    # Ensure the last line is included if still in line
    if in_line:
        lines.append((line_start, len(pixel_sums)))

    # Find the largest width and height in all rows
    largest_width, largest_height = 0, 0
    row_sprites = []

    for line_num, (start_y, end_y) in enumerate(lines):
        line_height = end_y - start_y

        # Ignore rows less than ignore_height
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
    
    # Create subfolder if it doesn't exist
    output_dir = os.path.dirname(output_image_path)
    os.makedirs(output_dir, exist_ok=True)
    final_image_pil.save(output_image_path)

    print(f"Final image saved as '{output_image_path}'.")

def remove_magic_color_from_image(image_path, magic_color, color_tolerance=10, output_suffix='_no_magic'):
    """Removes a specific color (magic color) from the image and saves it as a new PNG file.
    
    Args:
        image_path: Path to input image
        magic_color: Tuple of (R, G, B) representing the color to remove (e.g., (255, 0, 255) for magenta)
        color_tolerance: Tolerance for color matching (0-255)
        output_suffix: Suffix to add to the output filename
    
    Returns:
        Path to the new image file with magic color removed
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Ensure image has alpha channel
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
    # magic_color is in rgb format, convert to bgr for OpenCV
    magic_color_bgr = (magic_color[2], magic_color[1], magic_color[0])
    
    # First, create a mask for exact color match
    exact_mask = np.all(image[:, :, :3] == magic_color_bgr, axis=2).astype(np.uint8) * 255
    
    # Then, create a mask with tolerance for near-matches
    if color_tolerance > 0:
        lower_bound = np.array([max(0, magic_color_bgr[0] - color_tolerance),
                               max(0, magic_color_bgr[1] - color_tolerance),
                               max(0, magic_color_bgr[2] - color_tolerance)])
        upper_bound = np.array([min(255, magic_color_bgr[0] + color_tolerance),
                               min(255, magic_color_bgr[1] + color_tolerance),
                               min(255, magic_color_bgr[2] + color_tolerance)])
        
        tolerance_mask = cv2.inRange(image[:, :, :3], lower_bound, upper_bound)
        
        # Combine exact and tolerance masks
        mask = cv2.bitwise_or(exact_mask, tolerance_mask)
    else:
        mask = exact_mask
        
    # Set alpha channel to 0 (transparent) where mask matches
    image[:, :, 3] = np.where(mask == 255, 0, image[:, :, 3])
    
    # Generate output path
    base_name = os.path.splitext(image_path)[0]
    extension = '.png'
    output_path = f"{base_name}{output_suffix}{extension}"
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Magic color removed. Saved as '{output_path}'.")
    
    return output_path


def main():
    """Main entry point with CLI argument support."""
    parser = argparse.ArgumentParser(
        description='Split spritesheets into individual sprites with optional magic color removal.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""":
Examples:
  Interactive mode:
    python process_rows.py
  
  Automated mode:
    python process_rows.py -f input.png -o output/result.png
    python process_rows.py -f input.jpg -m "#FF00FF" -t 50 -o output.png
    python process_rows.py --file sprite.png --no-magic-color --ignore-height 40 --separation 5
        """
    )
    
    parser.add_argument('-f', '--file', help='Input spritesheet file path')
    parser.add_argument('-o', '--output', help='Output file path (default: output/<filename>_output.png)')
    parser.add_argument('-m', '--magic-color', help='Magic color to remove (RGB: "255,0,255" or HEX: "#FF00FF")')
    parser.add_argument('-t', '--tolerance', type=int, default=50, help='Color tolerance for magic color (0-255, default: 50)')
    parser.add_argument('--no-magic-color', action='store_true', help='Disable magic color removal')
    parser.add_argument('--ignore-height', type=int, default=35, help='Minimum row height to process (default: 35)')
    parser.add_argument('-s', '--separation', type=int, default=0, help='Pixel separation between sprites (default: 0)')
    
    args = parser.parse_args()
    
    # If no file specified, run interactive mode
    if not args.file:
        interactive_menu()
        return
    
    # Validate input file
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found!")
        return
    
    # Determine output path
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_file = f"output/{base_name}_output.png"
    
    # Ensure .png extension
    if not output_file.lower().endswith('.png'):
        output_file += '.png'
    
    # Handle magic color
    magic_color = None
    if not args.no_magic_color and args.magic_color:
        magic_color = parse_color(args.magic_color)
        if magic_color is None:
            print(f"Error: Invalid magic color format '{args.magic_color}'")
            print("Use RGB format (255,0,255) or HEX format (#FF00FF)")
            return
    
    # Process the image
    working_file = args.file
    
    if magic_color:
        print(f"Removing magic color: {format_color(magic_color)}")
        working_file = remove_magic_color_from_image(
            args.file,
            magic_color,
            color_tolerance=args.tolerance
        )
    
    print(f"Processing spritesheet...")
    process_rows(
        working_file,
        output_image_path=output_file,
        separation=args.separation,
        ignore_height=args.ignore_height
    )
    
    print(f"\nProcessing complete!")


if __name__ == '__main__':
    main()
