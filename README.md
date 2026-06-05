# 🧠 Emotion Detector

A real-time **Emotion Detection System** built using **Java**, **OpenCV**, and a pre-trained **Mini-XCEPTION Deep Learning Model**. The application captures live webcam input, detects faces, and predicts the user's emotional state with instant visual feedback.

## ✨ Features

- 📷 Real-time webcam capture
- 😀 Facial emotion recognition
- 👤 Face detection using Haar Cascade Classifier
- 🧠 Deep Learning powered emotion prediction
- 📝 Emotion logging for analysis
- ⚡ Fast and lightweight Java implementation

---

## 🛠️ Technologies Used

- Java 17
- OpenCV 4.12
- Maven
- Mini-XCEPTION Model
- Haar Cascade Face Detection

---

## 📂 Project Structure

```text
Emotion-Detector/
│
├── src/
│   └── Java source files
│
├── models/
│   └── fer2013_mini_XCEPTION.102-0.66.hdf5
│
├── bin/
│   └── Compiled class files
│
├── target/
│   └── Maven build output
│
├── Referenced Libraries/
│   └── opencv-4120.jar
│
├── emotion_log.txt
├── haarcascade_frontalface_default.xml
├── pom.xml
└── README.md
```

---

## How It Works

1. Captures live video from the webcam.
2. Detects faces using OpenCV's Haar Cascade classifier.
3. Extracts the facial region from each frame.
4. Processes the image using the Mini-XCEPTION model.
5. Predicts the user's emotion.
6. Displays and logs the detected emotion.

---

##Supported Emotions

- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

---

## Getting Started

### Prerequisites

- Java 17+
- Maven
- OpenCV 4.12

### Installation

```bash
git clone https://github.com/your-username/Emotion-Detector.git
cd Emotion-Detector
```

Build the project:

```bash
mvn clean install
```

Run the application and allow webcam access.

---

## Demo

Real-time emotion recognition from webcam feed using OpenCV and Deep Learning.

> Detect. Analyze. Understand Emotions.

---

## Future Improvements

- Higher accuracy emotion models
- Emotion statistics dashboard
- Multi-face emotion tracking
- Export emotion history to CSV
- GUI enhancements

---

## Author

Developed by **Jefin Judson** as a Computer Science project exploring Computer Vision, Machine Learning, and Java application development.

 If you found this project useful, consider giving it a star!
