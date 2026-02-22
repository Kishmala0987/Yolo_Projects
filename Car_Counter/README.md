# 🚗 Car Counter using YOLO + SORT

A real-time vehicle detection and counting system built using **YOLO object detection** and **SORT tracking algorithm**.
The system detects cars in a video stream, assigns unique IDs, and counts them when they cross a defined region or line.

---

<img width="1340" height="889" alt="image" src="https://github.com/user-attachments/assets/152c2d9d-3fe0-4aa8-bc34-f393db259001" />

## 📌 Overview

This project combines:

* **YOLO** → detects vehicles in each frame
* **SORT** → tracks detected objects across frames
* **Region logic** → counts vehicles only once when they enter a specified area

It ensures **accurate counting**, even when vehicles move fast or overlap.

---

## 🖥 Demo Output

* Bounding boxes around cars
* Unique tracking IDs
* Counter displayed on screen

## 🧠 How It Works

**Pipeline:**

1. Read frame from video
2. Detect vehicles using YOLO
3. Extract bounding boxes `(x1,y1,x2,y2,conf)`
4. Pass detections to tracker
5. Tracker assigns persistent IDs
6. Calculate object center
7. Check if center enters region
8. Count only if ID not counted before

---

## 🚀 Future Improvements

* Direction-based counting (in/out)
* Multi-lane counting
* Speed estimation
* Traffic analytics dashboard
* Cloud deployment

---

## 📜 License

This project is open-source and free to use for educational purposes.

---
