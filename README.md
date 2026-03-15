# Real-Time Multi-Camera Crowd Behavior Analysis
## Overview

  This project implements a real-time intelligent surveillance system that analyzes crowd behavior using multiple camera feeds. The system detects crowd density, motion patterns, and abnormal group behavior using computer vision and deep learning techniques.

The system processes multiple video streams simultaneously and provides visual alerts, heatmaps, and behavior classification through a monitoring dashboard.

Key Features

Multi-camera video monitoring

Real-time human detection

Crowd density estimation

Motion analysis using optical flow

Crowd behavior classification

Heatmap visualization

Alert generation for abnormal behavior

Web dashboard using FastAPI

System Architecture
Video Input (Multiple Cameras)
        ↓
Frame Preprocessing
        ↓
Human Detection (YOLO)
        ↓
Crowd Density Estimation
        ↓
Optical Flow Motion Analysis
        ↓
Crowd Behavior Detection
        ↓
Heatmap Visualization
        ↓
Alert Generation
        ↓
Web Dashboard (FastAPI)
Technologies Used
Category	Technology
Programming Language	Python
Computer Vision	OpenCV
Deep Learning	YOLO
Motion Analysis	Optical Flow
Backend	FastAPI
Frontend	HTML + CSS
Visualization	Heatmaps
Deployment	Docker (optional)
Project Workflow

Capture video from multiple cameras

Extract frames and preprocess them

Detect humans using object detection

Calculate crowd density

Analyze motion patterns

Detect abnormal crowd behavior

Generate alerts and visual heatmaps

Display results on the monitoring dashboard

Running the Project
Step 1

Clone the repository

git clone https://github.com/yourusername/crowd-behavior-analysis.git
Step 2

Install dependencies

pip install -r requirements.txt
Step 3

Run the server

uvicorn main:app --reload
Step 4

Open dashboard

http://127.0.0.1:8000
Example Output

Dashboard displays:

Live camera feeds

Crowd density

Detected behavior

Alert notifications

Heatmap visualization

Applications

This system can be used in:

Airports

Railway stations

Stadiums

Shopping malls

Public events

Smart city surveillance

Future Improvements

CNN-LSTM spatio-temporal modeling

GPU acceleration

Real-time RTSP camera integration

Alert notification system

Cloud deployment

Author

Data Science Project by Abubakkar Sithik
