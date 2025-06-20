# 🎥 Image-Triggered Display Based on Fourier Frequency Analysis

A Python tool that uses a webcam and 2D Fourier transform to monitor visual changes in subregions of a live image. When changes are detected, it dynamically displays corresponding images.

## 🔍 Features

- Real-time image capture via webcam  
- Region-based 2D Fourier spectrum analysis  
- Auto-switching displayed image based on frequency-domain difference  
- Image saved after a fixed time interval

## 🛠 Requirements

- Python 3.7+
- OpenCV (`cv2`)
- Pillow
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
