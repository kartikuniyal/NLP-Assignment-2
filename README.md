# Human Pose Skeleton Video Generator

## 1. Project Overview

The Human Pose Skeleton Video Generator is a Python-based application that converts ordinary videos into skeleton-based videos using Human Pose Estimation.

It detects human body joints and connects them to form a skeleton. The generated video displays white joints and body connections on a completely black background.

The project uses YOLO Pose Estimation for detecting human keypoints and Streamlit for providing an interactive web interface.

## 2. Technologies Used

* **Python:** Main programming language.
* **Streamlit:** Creates the web application interface.
* **OpenCV:** Reads, processes and writes video frames.
* **NumPy:** Handles image arrays and numerical operations.
* **Ultralytics YOLO:** Performs human pose detection.
* **ImageIO-FFmpeg:** Encodes processed videos into MP4 format.

## 3. Features

* Upload videos in multiple formats.
* Detect multiple people in a single video.
* Identify 17 human body keypoints.
* Connect joints to generate a human skeleton.
* Display skeletons using white lines and circles.
* Generate videos with a black background.
* Show real-time processing progress.
* Preview the generated video.
* Download the final MP4 video.
* Automatically use a backup YOLO model if the primary model fails to load.

## 4. Working Process

1. The user uploads a video through the Streamlit interface.
2. OpenCV reads the uploaded video frame by frame.
3. YOLO Pose Estimation detects people and their body keypoints.
4. Keypoints with sufficient confidence are selected.
5. OpenCV connects the detected joints using predefined skeleton connections.
6. A black image is created for each frame.
7. White lines and circles are drawn to represent the human skeleton.
8. Processed frames are combined into an MP4 video.
9. The generated video is displayed and made available for download.

## 5. Project Structure

```text
Human-Pose-Skeleton/
│
├── app.py
├── requirements.txt
├── README.md
└── pose_result.mp4
```

The YOLO model weights are downloaded automatically when the application runs for the first time.

## 6. Installation Requirements

* Python 3.10 or newer
* Internet connection for downloading model weights
* Required Python libraries

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## 7. Requirements File

```txt
streamlit
opencv-python
numpy
ultralytics>=8.4.0
imageio-ffmpeg
```

## 8. How to Run

**Step 1:** Open the project directory in VS Code.

**Step 2:** Install the required dependencies.

```bash
pip install -r requirements.txt
```

**Step 3:** Start the Streamlit application.

```bash
streamlit run app.py
```

**Step 4:** Open the local URL displayed in the terminal.

**Step 5:** Upload a video and click on Generate Pose Video.

**Step 6:** Preview and download the generated skeleton video.

## 9. Pose Detection Model

The project uses YOLO Pose Estimation models:

* YOLO26n-pose: Primary model.
* YOLO11n-pose: Backup model.

The model identifies 17 body keypoints based on the COCO keypoint format.

These keypoints include the nose, eyes, ears, shoulders, elbows, wrists, hips, knees and ankles.

## 10. Output

The final output is an MP4 video containing detected human skeletons.

* Background: Black
* Skeleton joints: White circles
* Body connections: White lines
* Output format: MP4

## 11. Applications

* Human movement analysis
* Sports performance analysis
* Fitness and exercise monitoring
* Human activity recognition
* Computer vision applications
* Gesture and posture analysis

## 12. Conclusion

This project demonstrates the use of computer vision and deep learning for human pose estimation. By combining YOLO, OpenCV and Streamlit, it provides a simple way to convert normal videos into human skeleton videos.
