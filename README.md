# Face Tracking

This project is a Python-based program for real-time face detection and tracking using OpenCV. It leverages a pre-trained SSD face detection model and a CSRT tracker to detect and keep the face centered in the video feed. This project can be used for applications such as face tracking and maintaining focus on a moving person in the frame.

## Features

- **Real-time Face Detection**: Detects faces in real-time using a pre-trained SSD model.
- **Face Tracking**: Tracks the detected face using OpenCV’s CSRT tracker for high accuracy.
- **Face Centering**: Keeps the face centered in the frame by adjusting the frame position dynamically.

## Requirements

This project requires Python 3.8 or above and the libraries listed in `requirements.txt`. Install dependencies using:

```bash
pip install -r requirements.txt
