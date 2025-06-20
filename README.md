# 📡 Free-space Optical Input System – Real-time Monitoring and Fourier Analysis

This project demonstrates a free-space optical input system that combines **real-time monitoring** with **Fourier-domain analysis**. It includes two main Python scripts: `keyboard.py` performs real-time monitoring and image switching using camera input, while `rf_analysis.py` analyzes image frequency characteristics at various distances to estimate the system's depth of field.

`keyboard.py` captures live video from the camera, crops a region of interest (ROI), and divides it into a 3×3 grid. It computes the 2D Fourier Transform of each zone and monitors high-frequency components. When significant changes in spatial frequency are detected in any zone, the system switches to a corresponding image (e.g., `image1.bmp`, `image2.bmp`, etc.). If no change is detected for 2 seconds, it reverts to the default `image0.bmp`. After 10 seconds, it automatically saves the cropped ROI image for future analysis.

`rf_analysis.py` is used to analyze a sequence of 15 images (`saved_image_1.png` to `saved_image_15.png`) captured at distances ranging from −2.5 cm to 4.5 cm. It crops the center 200×200 region from each image, performs Fourier Transform, extracts the DC component (fundamental) and first-order diffraction peak (along 45° direction), and computes the RF ratio (First-order / DC). The resulting RF-vs-Distance plot helps quantify the system’s **depth of field (DOF)** — higher RF ratios indicate better focus and stronger periodic structure. This analysis also helps determine the **effective detection range** for `keyboard.py`, showing when spatial frequency changes are most detectable.

### File Structure:
- `keyboard.py`: real-time image switch system
- `rf_analysis.py`: offline Fourier analysis tool
- `images/image0.bmp ~ image9.bmp`: switching targets for real-time display
- `saved_images/saved_image_1.png ~ saved_image_15.png`: static images at various distances
- `saved_cropped_image.png`: cropped image saved during live run
- `requirements.txt`: Python package dependencies

### Dependencies:
Install required libraries via:
```bash
pip install -r requirements.txt
# 📡 Free-space Optical Input System – Real-time Monitoring and Fourier Analysis

This project demonstrates a free-space optical input system that combines **real-time monitoring** with **Fourier-domain analysis**. It includes two main Python scripts: `keyboard.py` performs real-time monitoring and image switching using camera input, while `rf_analysis.py` analyzes image frequency characteristics at various distances to estimate the system's depth of field.

`keyboard.py` captures live video from the camera, crops a region of interest (ROI), and divides it into a 3×3 grid. It computes the 2D Fourier Transform of each zone and monitors high-frequency components. When significant changes in spatial frequency are detected in any zone, the system switches to a corresponding image (e.g., `image1.bmp`, `image2.bmp`, etc.). If no change is detected for 2 seconds, it reverts to the default `image0.bmp`. After 10 seconds, it automatically saves the cropped ROI image for future analysis.

`rf_analysis.py` is used to analyze a sequence of 15 images (`saved_image_1.png` to `saved_image_15.png`) captured at distances ranging from −2.5 cm to 4.5 cm. It crops the center 200×200 region from each image, performs Fourier Transform, extracts the DC component (fundamental) and first-order diffraction peak (along 45° direction), and computes the RF ratio (First-order / DC). The resulting RF-vs-Distance plot helps quantify the system’s **depth of field (DOF)** — higher RF ratios indicate better focus and stronger periodic structure. This analysis also helps determine the **effective detection range** for `keyboard.py`, showing when spatial frequency changes are most detectable.

### File Structure:
- `keyboard.py`: real-time image switch system
- `rf_analysis.py`: offline Fourier analysis tool
- `images/image0.bmp ~ image9.bmp`: switching targets for real-time display
- `saved_images/saved_image_1.png ~ saved_image_15.png`: static images at various distances
- `saved_cropped_image.png`: cropped image saved during live run
- `requirements.txt`: Python package dependencies

### Dependencies:
Install required libraries via:
```bash
pip install -r requirements.txt
