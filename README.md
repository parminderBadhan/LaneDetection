# Lane Detection

Simple lane and vehicle detection using Python, OpenCV, and YOLOv4-tiny.

## Features

- Detects lane boundaries from road video frames
- Detects vehicles using YOLOv4-tiny
- Draws visual overlays on output video
- Lightweight setup with minimal dependencies

## Requirements

- Python 3.8+
- OpenCV
- NumPy

## Quick Start

```bash
pip install -r requirements.txt
python lane_detection.py
```

## Input and Output

- Input video: input_video2.mp4
- Output video: output_video.avi

You can change file names directly in lane_detection.py if needed.

## Project Files

- lane_detection.py - main script
- requirements.txt - Python dependencies
- yolov4-tiny.cfg and yolov4-tiny.weights - YOLO model files
- coco.names - class labels
