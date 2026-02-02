# eyetracking-reader
![Static Badge](https://img.shields.io/badge/status-done-brightgreen?style=for-the-badge)
![Static Badge](https://img.shields.io/badge/type-class_project-blue?style=for-the-badge)

An eye-tracking project for gaze-based reading assistance using webcam video and facial landmarks. Developed for the 2024 Computer Science Project class at Seoul Science High School (서울과학고등학교). The project combines dlib face/eye landmarks, OpenCV pupil detection, and a TensorFlow gaze model to map gaze to screen position, and explores calibration steps and bionic-reading-style text display to support reading focus and analytics.

## How to Start

### Environment
- Python 3.10+
- Conda (recommended; use `environment.yaml`)
- Webcam
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/davidKMJ/eyetracking-reader.git
cd eyetracking-reader

# Create and activate conda environment from environment.yaml
conda env create -f environment.yaml
conda activate LK2

# Or install dependencies manually (OpenCV, dlib, TensorFlow, etc.)
# pip install opencv-contrib-python dlib tensorflow matplotlib numpy

# Run with Jupyter Notebook (recommended — avoids display issues on the Data screen)
jupyter notebook code_notebook.ipynb

# Or run the Python script (Data screen may have display issues)
python code_python.py
```

## Key Features
1. **Calibration** – Landmarks (68-point face), pupil detection threshold, and eye-data calibration for accurate gaze mapping
2. **Gaze-based text reading** – Display text and track which character/word the user is looking at
3. **Bionic reading** – Optional bold-first-half-of-word style to aid reading focus
4. **Test mode** – Try reading with gaze tracking and visual feedback (e.g., circles where you look)
5. **Data / analytics** – View reading time per sentence and simple time-based graphs
6. **Model training** – Train the eye-tracking model on your calibration data for personalized gaze estimation

## Technical Stack
- **OpenCV (cv2)** – Video capture, image processing, UI (buttons, text, windows)
- **dlib** – Face detection and 68-point facial landmark prediction (`shape_68.dat`)
- **TensorFlow / Keras** – Eye-tracking model for mapping eye features to screen position
- **NumPy** – Arrays and numerical operations
- **Matplotlib** – Plotting reading-time graphs
- **pickle** – Saving/loading calibration data and landmarks

## Project Structure
```
eyetracking-reader/
├── code_notebook.ipynb       # Main entry (recommended) — Jupyter version
├── code_python.py            # Standalone Python version
├── environment.yaml          # Conda environment (Python 3.10, OpenCV, dlib, TensorFlow, etc.)
├── shape_68.dat              # dlib 68-point face landmark predictor
├── default_model.keras        # Pre-trained eye-tracking model
├── default_landmarks.pkl     # Default landmark reference
├── default_calibration_data.pkl  # Default eye calibration data
├── text.txt                  # Sample text for reading
├── UI-*.png                  # UI assets (main menu, calibration, read text, etc.)
└── README.md
```

## Usage Flow
1. **Main menu** — Choose **Calibration**, **Read Text**, or **Exit**.
2. **Calibration** — Run **Landmarks**, **Threshold**, and **Eye datas** as needed, then optionally **Train** to update the gaze model.
3. **Read Text** — Open **Test** to read with gaze tracking (and optional bionic style), or **Data** to see reading-time analytics.

Press **q** to quit from most screens.
