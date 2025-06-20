import numpy as np
import cv2
from PIL import Image
import time

# Load and display the initial image (image0)
image_0 = Image.open(r'D:\app\pycharm\learn\key\image0.bmp')  # Use absolute path
image_0_cv = np.array(image_0)  # Convert to OpenCV format
image_0_cv_resized = cv2.resize(image_0_cv, (1920, 1080))  # Resize the image for display

# Show the initial image
cv2.imshow('Displayed Image', image_0_cv_resized)

# Initialize webcam input
cap = cv2.VideoCapture(1)  # Use camera ID 1 (adjust based on your setup)
ret, first_frame = cap.read()

# Check if the camera is working
if not ret:
    print("Failed to read camera input")
    exit()

# Record the start time for saving frame after 10 seconds
start_time = time.time()
image_saved = False  # Flag to indicate if the image has been saved

# Convert the first frame to grayscale
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Define the crop region (region of interest)
useful_row_start = 100
useful_row_end = 350
useful_col_start = 260
useful_col_end = 460

# Crop the grayscale image to the region of interest
first_gray_cropped = first_gray[useful_row_start:useful_row_end, useful_col_start:useful_col_end]

# Check if cropped region is valid
if first_gray_cropped.size == 0:
    print("Cropped region is empty, cannot proceed")
    exit()

# Get size of cropped region
cropped_rows, cropped_cols = first_gray_cropped.shape

# Divide the cropped region into a 3x3 grid
row_step = cropped_rows // 3
col_step = cropped_cols // 3

# Function to extract 3x3 zones from cropped image
def get_zones(image_cropped, row_step, col_step, cropped_rows, cropped_cols):
    return [
        image_cropped[0:row_step, 0:col_step],
        image_cropped[0:row_step, col_step:2 * col_step],
        image_cropped[0:row_step, 2 * col_step:cropped_cols],
        image_cropped[row_step:2 * row_step, 0:col_step],
        image_cropped[row_step:2 * row_step, col_step:2 * col_step],
        image_cropped[row_step:2 * row_step, 2 * col_step:cropped_cols],
        image_cropped[2 * row_step:cropped_rows, 0:col_step],
        image_cropped[2 * row_step:cropped_rows, col_step:2 * col_step],
        image_cropped[2 * row_step:cropped_rows, 2 * col_step:cropped_cols]
    ]

# Track currently displayed image
current_image = r'D:\app\pycharm\learn\key\image0.bmp'
displayed_image = image_0_cv_resized  # Start with image0
last_change_time = time.time()  # Last image change timestamp
change_detected = False  # Change detection flag

# Function to compute peak intensity in Fourier transform
def compute_fourier_peak_intensity(image, r_high=20):
    f_transform = np.fft.fft2(image)
    f_transform_shift = np.fft.fftshift(f_transform)

    # Create a mask to keep only high-frequency components
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if dist > r_high:
                mask[i, j] = 1

    # Apply mask and calculate magnitude spectrum
    f_transform_shift_masked = f_transform_shift * mask
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shift_masked) + 1)

    return np.max(magnitude_spectrum)

# Compute initial peak intensities of 3x3 zones
initial_zones = get_zones(first_gray_cropped, row_step, col_step, cropped_rows, cropped_cols)
initial_peak_intensities = [compute_fourier_peak_intensity(zone) for zone in initial_zones]

# Start real-time monitoring
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read video frame")
        break

    # Convert to grayscale and crop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_cropped = gray[useful_row_start:useful_row_end, useful_col_start:useful_col_end]

    if gray_cropped.size == 0:
        print("Cropped camera frame is empty, stopping")
        break

    # Display the cropped camera region
    cv2.imshow('Camera Feed', gray_cropped)

    # Compute current peak intensity of each zone
    zones = get_zones(gray_cropped, row_step, col_step, cropped_rows, cropped_cols)
    detected_change = False

    for zone_index, zone in enumerate(zones):
        current_peak_intensity = compute_fourier_peak_intensity(zone)
        initial_peak_intensity = initial_peak_intensities[zone_index]

        # If the difference exceeds threshold, change image
        if abs(current_peak_intensity - initial_peak_intensity) > 50:
            image_path = rf'D:\app\pycharm\learn\key\image{zone_index+1}.bmp'
            if current_image != image_path:
                img = Image.open(image_path)
                img_cv = np.array(img)
                img_cv_resized = cv2.resize(img_cv, (1920, 1080))
                displayed_image = img_cv_resized
                current_image = image_path
                last_change_time = time.time()
                detected_change = True
                change_detected = True
            break

    # Revert to image0 if no change detected for 2 seconds
    if not detected_change and change_detected and time.time() - last_change_time > 2:
        displayed_image = image_0_cv_resized
        current_image = r'D:\app\pycharm\learn\key\image0.bmp'
        change_detected = False

    # Always display current image
    cv2.imshow('Displayed Image', displayed_image)

    # Save cropped frame after 10 seconds (once)
    if not image_saved and time.time() - start_time > 10:
        save_path = r'D:\app\pycharm\learn\key\saved_cropped_image.png'
        cv2.imwrite(save_path, gray_cropped)
        print(f"Cropped region saved to: {save_path}")
        image_saved = True

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
