# 👁️ AssistAI v3

AI-powered assistive system for visually impaired users using **voice + vision + AI models**.

---

## 🚀 Features

* 🎤 Voice-controlled system (no wake word)
* 👁️ Scene description (BLIP model)
* 📖 Text reading (Gemini Vision OCR)
* 💰 Indian currency recognition
* 📦 Object detection (YOLO)
* 🧍 Face recognition + emotion analysis (DeepFace)
* 🎨 Color detection
* 📏 Distance estimation
* ⚠️ Danger detection (knife, fire, etc.)
* 🌍 Location & Weather info
* 🕒 Time, Date & Daily briefing

---

## 🏗️ Tech Stack

### Frontend

* HTML, CSS, JavaScript
* Web Speech API (Speech Recognition + TTS)

### Backend

* Python, Flask

### AI / ML

* BLIP (Image Captioning)
* YOLO (Object Detection)
* DeepFace (Face Recognition)
* Sentence Transformers (Intent Detection)

### APIs

* Gemini Vision API (OCR & Currency)
* Open-Meteo (Weather)
* OpenStreetMap (Location)

---

## 📁 Project Structure

```
project/
│
├── app.py                # Main Flask backend
├── templates/
│   └── index.html        # Frontend UI
├── static/               # CSS, JS (if separated)
├── faces/                # Stored face images
├── logs/                 # System logs
├── model_coco/           # YOLO weights/config
├── .env                  # API keys
└── requirements.txt      # Dependencies
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```
git clone <repo-url>
cd assistai
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Setup environment variables

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### 4️⃣ Add YOLO model files

Place inside `model_coco/`:

* yolov3.weights
* yolov3.cfg
* coco.names

### 5️⃣ Run the project

```
python app.py
```

Open in browser:

```
http://localhost:5000
```

---

## 🎤 Usage

1. Allow camera, microphone, and location permissions
2. Speak commands directly:

   * "what is this"
   * "read this"
   * "how much"
   * "who am I"
   * "what objects"
   * "where am I"
3. Say **"stop"** to deactivate

---

## 🔄 System Flow

Voice Input → Intent Detection → AI Processing → Response → Speech Output

---

## ⚠️ Limitations

* Requires internet for Gemini API
* Performance depends on lighting conditions
* YOLO v3 may be slower on low-end systems

---

## 📌 Future Improvements

* Upgrade to YOLOv8
* Offline OCR support
* Navigation assistance
* Mobile app version

---

## 👩‍💻 Author

Developed as an AI Assistive System Project.

---

## ⭐ Conclusion

AssistAI v3 enhances independence and safety for visually impaired users by combining computer vision, speech processing, and AI into a real-time assistive solution.
