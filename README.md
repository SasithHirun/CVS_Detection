# Eye Strain Detection

This project detects eye strain in real-time by analyzing the Eye Aspect Ratio (EAR) using a webcam. It utilizes **dlib**'s face detection and landmark detection, along with **OpenCV** for image processing. The system calculates the EAR to determine if the user is experiencing eye strain (or tiredness) based on their eyelid movement. If a user’s EAR falls below a certain threshold for multiple consecutive frames, it will trigger an alert.

## Features

- Real-time face detection using dlib’s frontal face detector.
- Eye Aspect Ratio (EAR) calculation to detect eye strain.
- Webcam access to capture frames for analysis.
- Detection of continuous eye strain based on EAR thresholds.
- Simple visualization of the result via OpenCV with real-time updates.

## Installation

To run this project, you need Python 3.13 or above. Ensure that the required dependencies are installed.

### Prerequisites

- Python 3.13+
- pip (Python package installer)

### Dependencies

1. Install the required packages:

   ```bash
   pip install opencv-python dlib scipy matplotlib flask numpy
