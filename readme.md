# Flexy: AI-Powered Rehabilitation & Mobility Assistant

> 🏆 **Winner: AllHealth Track — HackHarvard 2025**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-18.0-61DAFB.svg)
![Flask](https://img.shields.io/badge/Flask-2.0-000000.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-green.svg)
![IRIS](https://img.shields.io/badge/InterSystems-IRIS_Vector-orange.svg)

**Flexy** is an award-winning rehabilitation platform designed to make physical therapy engaging, accessible, and accurate. By leveraging computer vision, gamification, and Generative AI, Flexy helps users with arthritis and sports injuries perform exercises correctly and consistently.

---

## 📸 Demo & Features

### 1. Real-Time Posture Correction
Using **MediaPipe** and **OpenCV**, Flexy tracks 33 distinct body landmarks in real-time. It compares user movements against a custom dataset of correct forms using cosine similarity algorithms to ensure safe exercise execution.

![Posture Detection Demo](https://github.com/user-attachments/assets/00cf41a7-3317-46a3-bf24-a1322c22c701)

### 2. Gamified "Air Drawing" Rehabilitation
To combat the monotony of physical therapy, we implemented a gamified module where users control on-screen elements by moving specific joints (shoulders, nose, wrists). Users can "draw" shapes like infinity loops or waves, promoting range of motion in a fun, interactive way.

![Gamification Demo](https://github.com/user-attachments/assets/f27ada11-704b-4c63-8573-c9a10eeb9af6)

### 3. AI Health Companion & Dashboard
A **RAG (Retrieval-Augmented Generation)** chatbot, powered by **Llama 3** and **InterSystems IRIS Vector Search**, provides personalized recovery advice based on the user's history. The React dashboard visualizes accuracy trends and daily streaks.

![Dashboard Demo](https://github.com/user-attachments/assets/5d8d61f1-6f48-40d0-8c83-04e9203ea033)

---

## 🛠️ Tech Stack

### Frontend
* **React.js:** Dynamic user interface and dashboard.
* **Material UI (MUI):** Responsive design components.
* **Clerk:** User management and standard authentication.

### Backend & API
* **Flask (Python):** RESTful API handling video streams and model inference.
* **InterSystems IRIS:** Vector database for storing exercise memory embeddings.
* **Node.js:** Handling blockchain interactions for data integrity.

### AI & Computer Vision
* **MediaPipe:** Real-time pose and hand landmark detection.
* **OpenCV:** Image processing and frame manipulation.
* **LangChain & Llama:** LLM orchestration for the conversational assistant.
* **Deepgram:** Voice-to-text transcription for hands-free commands.
* **Face Recognition:** Biometric login system.

### Security
* **Simulated Blockchain:** A Node.js module that stores facial encoding hashes to ensure user biometric data integrity and privacy.

---

## ⚙️ System Architecture

1.  **Input:** User provides video feed via webcam and voice commands.
2.  **Processing:**
    * Video frames are processed by **MediaPipe** to extract skeletal coordinates.
    * Coordinates are compared using **NumPy** vector math against a "Gold Standard" JSON dataset of exercises.
    * Facial data is hashed and verified against the **Blockchain** ledger.
3.  **Intelligence:**
    * User queries are embedded and searched against the **IRIS Vector Database**.
    * Context is fed into **Llama** to generate medical/fitness advice.
4.  **Output:** Real-time visual feedback overlays, rep counting, and audio guidance.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* Node.js & npm
* Webcam

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/flexy.git](https://github.com/yourusername/flexy.git)
    cd flexy
    ```

2.  **Backend Setup**
    Navigate to the root directory and install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have your `.env` file set up with `BASETEN_API_KEY`, `DEEPGRAM_API_KEY`, and IRIS connection details.*

3.  **Frontend Setup**
    Navigate to the client folder:
    ```bash
    cd client
    npm install
    npm start
    ```

4.  **Run the Backend**
    ```bash
    python api/app.py
    ```

---

## 🧠 Challenges & Solutions

* **Latency vs. Accuracy:** Processing live video for 33 keypoints while running similarity calculations caused initial lag. We optimized this by vectorizing the math operations using NumPy and reducing the frame processing resolution without sacrificing detection accuracy.
* **Gamification Geometry:** Mapping 3D body movements to 2D drawing planes required complex computational geometry. We implemented a dynamic thresholding system that adapts to the user's distance from the camera.
* **Data Privacy:** To secure biometric data, we implemented a prototype blockchain storage system (in `backend/blockchain.js`) that ensures facial encodings are decentralized and tamper-proof.

---

## 🔮 What's Next?

* **VR Integration:** Porting the "Air Drawing" feature to Oculus/Vision Pro for immersive rehabilitation.
* **Wearable Sync:** Integrating with Apple Health/Fitbit to correlate heart rate data with exercise intensity.
* **Telehealth Portal:** allowing physical therapists to remotely view patient accuracy stats and adjust exercise difficulty.