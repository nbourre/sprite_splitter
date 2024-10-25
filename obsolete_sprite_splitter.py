import cv2

def find_largest_sprite(image_path):
    """Finds the rectangle of the largest sprite in terms of area."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h

def find_widest_frame(image_path):
    """Finds the rectangle of the frame with the largest width."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    widest_contour = max(contours, key=lambda c: cv2.boundingRect(c)[2])
    x, y, w, h = cv2.boundingRect(widest_contour)

    return x, y, w, h

def find_highest_frame(image_path):
    """Finds the rectangle of the frame with the largest height."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    highest_contour = max(contours, key=lambda c: cv2.boundingRect(c)[3])
    x, y, w, h = cv2.boundingRect(highest_contour)

    return x, y, w, h

def show_rectangle(image_path, rect, scale=1.0):
    """
    Displays the image with a rectangle drawn on it using OpenCV's imshow.
    :param image_path: Path to the image file.
    :param rect: Tuple (x, y, w, h) representing the rectangle.
    :param scale: Float value to scale the displayed image.
    """
    # Load the image with OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Draw the rectangle on the image
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Use green color for the rectangle

    # Resize the image based on the scale value
    if scale != 1.0:
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Display the image using OpenCV
    cv2.imshow("Largest or widest frame", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = 'adventure_time.png'

# Find the widest frame
widest_frame = find_widest_frame(image_path)
if widest_frame:
    print(f"Widest frame found at: x={widest_frame[0]}, y={widest_frame[1]}, width={widest_frame[2]}, height={widest_frame[3]}")
    show_rectangle(image_path, widest_frame, scale=0.5)

# Find the highest frame
highest_frame = find_highest_frame(image_path)
if highest_frame:
    print(f"Highest frame found at: x={highest_frame[0]}, y={highest_frame[1]}, width={highest_frame[2]}, height={highest_frame[3]}")
    show_rectangle(image_path, highest_frame, scale=0.5)
