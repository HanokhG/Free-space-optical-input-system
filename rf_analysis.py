import numpy as np
import cv2
import matplotlib.pyplot as plt

# Compute Fourier transform and extract ratio of first-order to DC component
def compute_rf_ratio(image, plot_first_order=False, image_index=1):
    # Ensure image is grayscale
    if len(image.shape) != 2:
        print("Error: Image is not grayscale.")
        return None, None, None, None, None

    # Compute Fourier transform and shift the zero-frequency component to the center
    f_transform = np.fft.fft2(image)
    f_transform_shift = np.fft.fftshift(f_transform)

    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shift)

    # Get DC component (center of frequency domain)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    dc_value = magnitude_spectrum[crow, ccol]

    # Search for the first-order peak along 45° direction (starting from radius 20)
    min_radius = 20
    search_radius = 50
    max_value = 0
    first_order_pos = (0, 0)

    for r in range(min_radius, search_radius):
        x_pos = int(ccol + r / np.sqrt(2))  # Horizontal offset along 45°
        y_pos = int(crow - r / np.sqrt(2))  # Vertical offset along 45°

        # Check bounds
        if x_pos < 0 or x_pos >= cols or y_pos < 0 or y_pos >= rows:
            continue

        # Find local maximum
        if magnitude_spectrum[y_pos, x_pos] > max_value:
            max_value = magnitude_spectrum[y_pos, x_pos]
            first_order_pos = (y_pos, x_pos)

    # Get value of first-order component
    first_order_value = magnitude_spectrum[first_order_pos]

    # Compute RF ratio = first-order / DC component
    rf_ratio = first_order_value / dc_value

    # Plot first-order position if needed
    if plot_first_order:
        plt.figure(figsize=(8, 6))
        plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
        plt.scatter([first_order_pos[1]], [first_order_pos[0]], color='red', marker='o', s=50, label="First Order")
        plt.title(f"First Order Position on Frequency Spectrum (Image {image_index})")
        plt.legend()
        plt.show()

    return dc_value, first_order_value, rf_ratio, magnitude_spectrum, first_order_pos


# Function to crop center 200×200 region from the image
def crop_center(image, crop_size=200):
    rows, cols = image.shape
    start_row = (rows - crop_size) // 2
    start_col = (cols - crop_size) // 2
    return image[start_row:start_row + crop_size, start_col:start_col + crop_size]


# Define distance range (-2.5 to 4.5 cm, total 15 images)
distances = np.linspace(-2.5, 4.5, 15)
rf_ratios = []

# Read each image and compute RF ratio
for i in range(1, 16):  # From 1 to 15
    image_path = rf'D:\app\pycharm\learn\key\saved_image_{i}.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image was successfully loaded
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Crop center 200×200 region
    cropped_image = crop_center(image)

    # Check if cropped image is valid
    if len(cropped_image.shape) != 2:
        print(f"Error: Cropped image {i} is not valid grayscale.")
        continue

    # Compute RF ratio and plot first-order location
    dc_value, first_order_value, rf_ratio, magnitude_spectrum, first_order_pos = compute_rf_ratio(
        cropped_image,
        plot_first_order=True,
        image_index=i
    )

    # If computation failed, skip
    if dc_value is None:
        continue

    print(f"Image {i}: DC = {dc_value}, First Order = {first_order_value}, RF = {rf_ratio}")
    rf_ratios.append(rf_ratio)

# Plot distance vs. RF ratio
plt.figure(figsize=(8, 6))
plt.plot(distances, rf_ratios, marker='o', linestyle='-', color='b')
plt.title('Ratio of the First Order to the Fundamental Component (RF)')
plt.xlabel('Distance (cm)')
plt.ylabel('RF')
plt.grid(True)
plt.show()
