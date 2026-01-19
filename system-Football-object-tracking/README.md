# ⚽ Football Object Tracking System

A **Computer Vision system for football (soccer) object detection and tracking**.  
This project detects and tracks **players, referees, and the ball** from football match videos using deep learning and multi-object tracking techniques.

---

## 📌 Features

- ✅ Detect players, referees, and the ball
- ✅ Track objects across video frames with unique IDs
- ✅ Works on full match or highlight videos
- ✅ Real-time or offline processing
- ✅ Outputs annotated videos and tracking data
- ✅ Modular and easy to extend

---

## 🧠 System Overview

The system works in three main stages:

1. **Object Detection**  
   A YOLO-based deep learning model detects football objects in each frame.

2. **Multi-Object Tracking**  
   A tracking algorithm (e.g. ByteTrack / DeepSORT) assigns consistent IDs to detected objects across frames.

3. **Visualization & Output**  
   Bounding boxes, object IDs, and labels are drawn on the video and saved to disk.

---

## 🎬 Demo (Running System)

**Insert your running video or GIF below:**  
👉 Example: *(replace `DEMO_VIDEO_URL_OR_GIF` with your real link)*

![Demo Running Video]([])


https://github.com/user-attachments/assets/627168f6-dd50-488a-817c-afd43d9dd227



## 📁 Project Structure

