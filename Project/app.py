"""
ASSISTIVE AI v3
Fixes:
- Wake word stays active until "stop/sleep/goodbye"
- Currency detection via Gemini Vision API
- Removed: scan QR, check expiry, read medicine, how many people
- Windows Unicode logging fix
- sentence_transformers graceful fallback
"""

import os, sys, cv2, json, time, logging, threading, requests, numpy as np, base64
from datetime import datetime
from collections import deque, Counter
from flask import Flask, request, jsonify, render_template, send_from_directory

# ── WINDOWS UNICODE FIX ──────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except:
        pass

# ── CONFIG ───────────────────────────────────────────────────
FACE_DB        = "faces"
LOG_DIR        = "logs"
CONF_THRESH    = 0.55
CURR_THRESH    = 0.60
DANGER_THRESH  = 0.50
FRAME_COUNT    = 5
CACHE_SECS     = 2
REPEAT_LIMIT   = 3
DANGER_CLASSES = {"knife","scissors","fire","stairs","step"}

# Gemini API key
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
os.makedirs(FACE_DB, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── LOGGING ──────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, datetime.now().strftime("assistive_%Y%m%d.log"))

class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = msg.replace("\u2713","OK").replace("\u2717","FAIL").replace("\u2192",">")
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
fh  = logging.FileHandler(log_file, encoding="utf-8")
fh.setFormatter(fmt)
ch  = SafeStreamHandler(sys.stdout)
ch.setFormatter(fmt)
logging.basicConfig(level=logging.INFO, handlers=[fh, ch])
for lib in ["urllib3","requests","PIL","transformers","httpx","sentence_transformers","h11"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
log = logging.getLogger("AssistAI")

app = Flask(__name__)

# ── GLOBAL STATE ─────────────────────────────────────────────
pending_name     = False
last_result      = ""
repeat_count     = 0
conversation_ctx = []
model_lock       = threading.Lock()
_cached_img      = None
_cached_img_time = 0
currency_buffer  = deque(maxlen=FRAME_COUNT)
object_buffer    = deque(maxlen=FRAME_COUNT)
_weather_cache   = None
_weather_time    = 0

# ── MODEL HOLDERS ────────────────────────────────────────────
_blip_processor    = None
_blip_model        = None
_intent_model      = None
_intent_labels     = None
_intent_embeddings = None
_coco_net          = None
_coco_names        = None
_use_semantic      = False

# ── MEDICINE PRONUNCIATION ───────────────────────────────────
MEDICINE_PRON = {
    "paracetamol":"para-seet-a-mol","ibuprofen":"eye-byoo-pro-fen",
    "amoxicillin":"a-mox-i-sil-in","metformin":"met-for-min",
    "atorvastatin":"a-tor-va-sta-tin","amlodipine":"am-lo-di-peen",
    "omeprazole":"oh-mep-ra-zole","azithromycin":"a-zi-thro-my-sin",
    "cetirizine":"se-ti-ri-zeen","pantoprazole":"pan-to-pra-zole",
    "ciprofloxacin":"sip-ro-flox-a-sin","dolo":"do-lo",
    "crocin":"kro-sin","combiflam":"kom-bi-flam","allegra":"a-le-gra",
}

# ── INTENT CORPUS ────────────────────────────────────────────
INTENT_CORPUS = {
    "scene":    ["what is this","describe what you see","what do you see",
                 "what am i holding","describe the scene","look at this",
                 "tell me what this is","what can you see","what is in front of me"],
    "ocr":      ["read this","read the text","what does it say","read it out",
                 "what is written here","read out loud","tell me what is written"],
    "currency": ["how much is this","what note is this","identify this money",
                 "what currency","how much money","what rupee note",
                 "what denomination","how much rupees","identify the note"],
    "objects":  ["what objects are here","what things are around me","detect objects",
                 "what is in this room","list the objects","what items are here"],
    "face":     ["who am i","recognize my face","identify me","who is this person",
                 "tell me who i am","do you know me","what is my emotion","how do i look"],
    "register": ["register my face","save my face","remember me","add my face",
                 "store my face","register face","save face to memory"],
    "color":    ["what color is this","identify the color","what colour",
                 "tell me the color","what shade is this","what color am i wearing"],
    "location": ["where am i","what is my location","tell me the address",
                 "where is this place","current location","where are we"],
    "weather":  ["what is the weather","how is the weather today","will it rain",
                 "temperature outside","weather today","whats the weather"],
    "time":     ["what time is it","tell me the time","current time","time please"],
    "date":     ["what day is today","what is the date","which day is it","today is date"],
    "briefing": ["good morning","morning briefing","start my day",
                 "daily briefing","good afternoon","good evening"],
    "help":     ["help me","what can you do","list commands","available commands"],
    "repeat":   ["say that again","repeat","again","one more time","what did you say"],
    "more":     ["tell me more","give more details","more information","elaborate"],
    "faster":   ["speak faster","talk faster","speed up","too slow","faster please"],
    "slower":   ["speak slower","slow down","too fast","slower please"],
    "light":    ["where is the light coming from","light direction",
                 "where is the light source","which direction is the light"],
    "distance": ["how far","distance to object","how close","how near","how far is it"],
    "sleep":    ["stop","sleep","goodbye","go to sleep","stop listening",
                 "deactivate","turn off","bye"],
    "danger":   ["is it safe","any danger","check for danger","any obstacles","scan for hazards"],
}

# ── KEYWORD FALLBACK ─────────────────────────────────────────
def keyword_intent(cmd):
    c = cmd.lower()
    if any(w in c for w in ["stop","sleep","goodbye","go to sleep","bye","deactivate"]): return "sleep"
    if "time" in c: return "time"
    if any(w in c for w in ["date","day","today"]): return "date"
    if "weather" in c: return "weather"
    if any(w in c for w in ["where am","location","address"]): return "location"
    if any(w in c for w in ["read","text","written","says"]): return "ocr"
    if any(w in c for w in ["who am","face","recognize","emotion","look"]): return "face"
    if "register" in c: return "register"
    if any(w in c for w in ["color","colour"]): return "color"
    if any(w in c for w in ["money","rupee","how much","note","currency","denomination"]): return "currency"
    if any(w in c for w in ["object","thing","item","surround"]): return "objects"
    if any(w in c for w in ["again","repeat"]): return "repeat"
    if "help" in c: return "help"
    if any(w in c for w in ["morning","briefing","afternoon","evening"]): return "briefing"
    if any(w in c for w in ["more","detail","elaborate"]): return "more"
    if any(w in c for w in ["faster","speed up"]): return "faster"
    if any(w in c for w in ["slower","slow down"]): return "slower"
    if any(w in c for w in ["light","lamp","bulb"]): return "light"
    if any(w in c for w in ["how far","distance","close","near"]): return "distance"
    if any(w in c for w in ["danger","safe","obstacle","hazard"]): return "danger"
    if any(w in c for w in ["what","describe","see","show","look"]): return "scene"
    return "unknown"

def get_intent(cmd):
    global _use_semantic
    if _use_semantic and _intent_model is not None:
        try:
            from sentence_transformers import util
            q = _intent_model.encode(cmd, convert_to_tensor=True)
            scores = util.cos_sim(q, _intent_embeddings)[0]
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            intent = _intent_labels[best_idx]
            log.info(f"Semantic intent: '{intent}' score={best_score:.2f}")
            if best_score >= 0.32:
                return intent
        except Exception as e:
            log.warning(f"Semantic intent error: {e}")
    return keyword_intent(cmd)

# ── MODEL LOADERS ────────────────────────────────────────────
def load_blip():
    global _blip_processor, _blip_model
    if _blip_processor is not None: return
    try:
        log.info("Loading BLIP...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        log.info("BLIP loaded OK")
    except Exception as e:
        log.error(f"BLIP load failed: {e}")

def load_intent_model():
    global _intent_model, _intent_labels, _intent_embeddings, _use_semantic
    if _intent_model is not None: return
    try:
        log.info("Loading sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        _intent_model = SentenceTransformer("all-MiniLM-L6-v2")
        phrases, labels = [], []
        for intent, corpus in INTENT_CORPUS.items():
            for p in corpus:
                phrases.append(p)
                labels.append(intent)
        _intent_embeddings = _intent_model.encode(phrases, convert_to_tensor=True)
        _intent_labels = labels
        _use_semantic = True
        log.info("Intent model loaded OK")
    except ModuleNotFoundError:
        log.warning("sentence_transformers not installed. Using keyword fallback.")
        _use_semantic = False
    except Exception as e:
        log.warning(f"Intent model failed: {e}. Using keyword fallback.")
        _use_semantic = False

def load_coco_net():
    global _coco_net, _coco_names
    w = "model_coco/yolov3.weights"
    c = "model_coco/yolov3.cfg"
    n = "model_coco/coco.names"
    if _coco_net is not None: return
    if os.path.exists(w) and os.path.exists(c):
        try:
            log.info("Loading COCO YOLO...")
            _coco_net = cv2.dnn.readNet(w, c)
            _coco_names = [l.strip() for l in open(n, encoding="utf-8")]
            log.info("COCO YOLO loaded OK")
        except Exception as e:
            log.error(f"COCO YOLO failed: {e}")
    else:
        log.warning("COCO model files not found.")

def warmup_all():
    log.info("=" * 50)
    log.info("  WARMING UP ALL MODELS...")
    log.info("=" * 50)
    for fn in [load_blip, load_intent_model, load_coco_net]:
        threading.Thread(target=fn, daemon=True).start()

# ── IMAGE HELPERS ─────────────────────────────────────────────
def check_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = np.mean(gray)
    s = cv2.Laplacian(gray, cv2.CV_64F).var()
    if b < 40:  return False, "Image is too dark. Please move to a brighter area."
    if b > 235: return False, "Image is too bright. Please reduce the glare."
    if s < 50:  return False, "Image is too blurry. Please hold the camera steady."
    return True, "OK"

def decode_image(img_bytes):
    global _cached_img, _cached_img_time
    now = time.time()
    if _cached_img is not None and (now - _cached_img_time) < CACHE_SECS:
        return _cached_img
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        _cached_img = img
        _cached_img_time = now
    return img

def img_to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")

def add_memory(cmd, response, intent):
    global conversation_ctx
    conversation_ctx.append({
        "cmd": cmd, "response": response,
        "intent": intent, "time": time.time()
    })
    if len(conversation_ctx) > 15:
        conversation_ctx.pop(0)

def read_text_gemini(img):
    try:
        log.info("Running Gemini OCR...")

        if not GEMINI_API_KEY:
            return "API key missing."

        b64 = img_to_base64(img)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": b64
                            }
                        },
                        {
                            "text": (
                                "Read all visible text from this image clearly. "
                                "Return only the extracted text in proper sentences. "
                                "If multiple languages exist, read everything."
                            )
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent",
            params={"key": GEMINI_API_KEY},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            log.error(f"Gemini OCR error: {response.status_code} {response.text}")
            return "Text reading failed."

        data = response.json()
        result = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        if result:
            return f"The text says: {result}"

        return "No readable text found."

    except Exception as e:
        log.error(f"Gemini OCR error: {e}")
        return "Text reading failed."
# ── SCENE ────────────────────────────────────────────────────
def caption_scene(img, detailed=False):
    global _blip_processor, _blip_model

    if _blip_processor is None or _blip_model is None:
        log.info("BLIP not ready, loading now...")
        load_blip()

    if _blip_processor is None or _blip_model is None:
        return "Scene model is still loading. Please try again."

    try:
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)

        inputs = _blip_processor(images=pil, return_tensors="pt")
        out = _blip_model.generate(**inputs, max_new_tokens=60 if detailed else 40)

        result = _blip_processor.decode(out[0], skip_special_tokens=True)
        return f"I see {result}"

    except Exception as e:
        log.error(f"BLIP error: {e}")
        return "Unable to understand the scene. Please try again."
def get_scene_type(img):
    if _coco_net is None: return ""
    try:
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, False)
        _coco_net.setInput(blob)
        layers = _coco_net.getLayerNames()
        outs = _coco_net.forward([layers[i-1] for i in _coco_net.getUnconnectedOutLayers()])
        detected = set()
        for o in outs:
            for d in o:
                s = d[5:]; cid = np.argmax(s)
                if float(s[cid]) > CONF_THRESH: detected.add(_coco_names[cid])
        scene_map = {
            "a kitchen": {"cup","bowl","bottle","fork","knife","spoon","microwave","oven","refrigerator","sink"},
            "an office":  {"laptop","keyboard","mouse","monitor","book","chair"},
            "outdoors":   {"car","bicycle","motorcycle","bus","truck","traffic light"},
            "a bedroom":  {"bed","pillow","alarm clock"},
        }
        best = max(scene_map, key=lambda k: len(detected & scene_map[k]))
        return f" You appear to be in {best}." if len(detected & scene_map[best]) >= 2 else ""
    except: return ""

# ── OCR (EasyOCR — Surya v0.17 API incompatible) ────────────
# EasyOCR reader cached so it loads only once
_easyocr_reader = None

# ── CURRENCY via GEMINI VISION ────────────────────────────────
def detect_currency_gemini(img):
    try:
        log.info("Running Gemini currency detection...")

        if not GEMINI_API_KEY:
            return "API key missing."

        b64 = img_to_base64(img)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": b64
                            }
                        },
                        {
                            "text": (
                                "Identify Indian currency from this image. "
                                "Reply ONLY in this format: "
                                "'This is a X rupee note.' or "
                                "'This is a X rupee coin.' "
                                "No explanation."
                            )
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent",
            params={"key": GEMINI_API_KEY},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code != 200:
            log.error(f"Gemini API error: {response.status_code} {response.text}")
            return "Currency detection failed."

        data = response.json()
        result = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        return result

    except Exception as e:
        log.error(f"Gemini error: {e}")
        return "Currency detection failed."
# ── COLOR ─────────────────────────────────────────────────────
def detect_color(img):
    try:
        h, w = img.shape[:2]
        cx, cy = w//2, h//2
        region = img[max(0,cy-60):cy+60, max(0,cx-60):cx+60]
        if region.size == 0: region = img
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hv = float(np.mean(hsv[:,:,0]))
        sv = float(np.mean(hsv[:,:,1]))
        vv = float(np.mean(hsv[:,:,2]))
        shade = "very dark" if vv < 50 else "dark" if vv < 100 else "light" if (vv > 200 and sv < 30) else ""
        if sv < 30:
            color = "black" if vv < 50 else "white" if vv > 200 else "gray"
        elif hv < 10 or hv > 170: color = "red"
        elif hv < 25:  color = "orange"
        elif hv < 35:  color = "yellow"
        elif hv < 85:  color = "green"
        elif hv < 130: color = "blue"
        elif hv < 150: color = "purple"
        else:          color = "pink"
        c = f"{shade} {color}".strip()
        return f"The dominant color is {c}."
    except Exception as e:
        log.error(f"Color error: {e}")
        return "Could not detect the color."

# ── LIGHT ─────────────────────────────────────────────────────
def detect_light(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        regions = {
            "above": float(np.mean(gray[:h//2,:])),
            "below": float(np.mean(gray[h//2:,:])),
            "left":  float(np.mean(gray[:,:w//2])),
            "right": float(np.mean(gray[:,w//2:])),
        }
        brightest = max(regions, key=regions.get)
        diff = regions[brightest] - min(regions.values())
        if diff < 15: return "The lighting appears even from all directions."
        return f"The light source appears to be coming from {brightest}."
    except Exception as e:
        log.error(f"Light error: {e}")
        return "Could not determine light direction."

# ── DISTANCE ──────────────────────────────────────────────────
def estimate_distance(img):
    if _coco_net is None:
        return "Distance estimation requires the object detection model."
    try:
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, False)
        _coco_net.setInput(blob)
        layers = _coco_net.getLayerNames()
        outs = _coco_net.forward([layers[i-1] for i in _coco_net.getUnconnectedOutLayers()])
        best_obj, best_area = None, 0
        for o in outs:
            for d in o:
                s = d[5:]; cid = np.argmax(s); conf = float(s[cid])
                if conf > CONF_THRESH:
                    area = float(d[2]) * float(d[3])
                    if area > best_area:
                        best_area = area
                        best_obj = _coco_names[cid]
        if not best_obj: return "No object detected to estimate distance."
        if best_area > 0.5:    dist = "very close, within arm reach"
        elif best_area > 0.2:  dist = "close, about 1 to 2 metres away"
        elif best_area > 0.05: dist = "moderate distance, about 2 to 5 metres away"
        else:                  dist = "far away, more than 5 metres"
        return f"The nearest object is a {best_obj}. It appears to be {dist}."
    except Exception as e:
        log.error(f"Distance error: {e}")
        return "Could not estimate distance."

# ── DANGER ────────────────────────────────────────────────────
def check_danger(img):
    if _coco_net is None: return None
    try:
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, False)
        _coco_net.setInput(blob)
        layers = _coco_net.getLayerNames()
        outs = _coco_net.forward([layers[i-1] for i in _coco_net.getUnconnectedOutLayers()])
        dangers = []
        for o in outs:
            for d in o:
                s = d[5:]; cid = np.argmax(s); conf = float(s[cid])
                if conf > DANGER_THRESH and _coco_names[cid].lower() in DANGER_CLASSES:
                    dangers.append(_coco_names[cid].lower())
        if not dangers: return None
        items = list(set(dangers))
        return (f"WARNING. {items[0]} detected nearby. Please be careful."
                if len(items) == 1
                else f"WARNING. Dangerous objects: {', '.join(items)}. Please be careful.")
    except Exception as e:
        log.error(f"Danger error: {e}")
        return None

# ── OBJECTS ───────────────────────────────────────────────────
def detect_objects(img):
    if _coco_net is None: return caption_scene(img)
    try:
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, False)
        _coco_net.setInput(blob)
        layers = _coco_net.getLayerNames()
        outs = _coco_net.forward([layers[i-1] for i in _coco_net.getUnconnectedOutLayers()])
        frame_det = set()
        for o in outs:
            for d in o:
                s = d[5:]; cid = np.argmax(s); conf = float(s[cid])
                if conf > CONF_THRESH and _coco_names[cid] != "person":
                    frame_det.add(_coco_names[cid])
        object_buffer.append(frame_det)
        all_objs = Counter(obj for frame in object_buffer for obj in frame)
        thresh = max(1, len(object_buffer)//2)
        stable = [obj for obj, cnt in all_objs.items() if cnt >= thresh]
        if not stable: return "I do not see any clear objects right now."
        if len(stable) == 1: return f"I see a {stable[0]}."
        if len(stable) == 2: return f"I see a {stable[0]} and a {stable[1]}."
        top = stable[:5]
        return f"I see {', '.join(top[:-1])} and {top[-1]}."
    except Exception as e:
        log.error(f"Objects error: {e}")
        return "Object detection failed. Please try again."

# ── FACE ──────────────────────────────────────────────────────
def register_face(name, img):
    try:
        os.makedirs(FACE_DB, exist_ok=True)
        cv2.imwrite(f"{FACE_DB}/{name}.jpg", img)
        return f"{name} has been registered successfully."
    except Exception as e:
        log.error(f"Register error: {e}")
        return "Could not save the face. Please try again."

def recognize_face(img):
    try:
        from deepface import DeepFace
        analysis = DeepFace.analyze(img, actions=["emotion","age","gender"],
                                    enforce_detection=False, silent=True)
        emotion = analysis[0]["dominant_emotion"]
        age     = int(analysis[0]["age"])
        gender  = analysis[0]["dominant_gender"].lower()
        if not os.path.exists(FACE_DB) or not os.listdir(FACE_DB):
            return f"I see a {gender} who is approximately {age} years old and looks {emotion}."
        results = DeepFace.find(img, db_path=FACE_DB, enforce_detection=False, silent=True)
        if len(results) > 0 and len(results[0]) > 0:
            name = os.path.basename(results[0].iloc[0]["identity"]).split(".")[0]
            name = name.replace("_"," ").title()
            return f"Hello {name}! You look {emotion} today. You appear to be around {age} years old."
        return f"I see someone new. You look {emotion} and appear to be around {age} years old."
    except ImportError:
        return "Face recognition not available. Please install deepface."
    except Exception as e:
        log.error(f"Face error: {e}")
        return "Face is not clear. Please look directly at the camera."

# ── TIME / DATE / BRIEFING ────────────────────────────────────
def get_time():    return f"The current time is {datetime.now().strftime('%I:%M %p')}."
def get_date():    return f"Today is {datetime.now().strftime('%A, %d %B %Y')}."
def get_briefing(lat=None, lon=None):
    h = datetime.now().hour
    g = "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"
    b = f"{g}. {get_date()} {get_time()} "
    if lat and lon:
        b += get_weather(float(lat), float(lon))
    return b.strip()

# ── WEATHER ───────────────────────────────────────────────────
def get_weather(lat, lon):
    global _weather_cache, _weather_time
    now = time.time()
    if _weather_cache and (now - _weather_time) < 600:
        return _weather_cache
    try:
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "current": ["temperature_2m","weathercode","windspeed_10m"],
            "timezone": "auto"
        }, timeout=5).json()
        cw    = resp.get("current", {})
        temp  = cw.get("temperature_2m","?")
        wind  = cw.get("windspeed_10m","?")
        wcode = int(cw.get("weathercode", 0))
        desc  = ("clear sky" if wcode==0 else "mainly clear" if wcode<4 else
                 "cloudy" if wcode<50 else "drizzling" if wcode<60 else
                 "raining" if wcode<70 else "snowing" if wcode<80 else
                 "rain showers" if wcode<90 else "thunderstorm")
        result = (f"Current weather: {desc}. "
                  f"Temperature is {temp} degrees Celsius. "
                  f"Wind speed is {wind} kilometres per hour.")
        _weather_cache, _weather_time = result, now
        return result
    except Exception as e:
        log.error(f"Weather error: {e}")
        return "Weather information is not available right now."

# ── LOCATION ──────────────────────────────────────────────────
def get_address(lat, lon):
    try:
        data = requests.get("https://nominatim.openstreetmap.org/reverse", params={
            "lat":lat,"lon":lon,"format":"json","zoom":16
        }, headers={"User-Agent":"AssistiveAI/3.0","Accept-Language":"en"}, timeout=6).json()
        addr  = data.get("address",{})
        parts = [addr[k] for k in
                 ["road","neighbourhood","suburb","city","town","state","country"]
                 if addr.get(k)]
        if parts: return f"You are currently at {', '.join(parts)}."
        display = data.get("display_name","")
        return f"You are at {display}." if display else "Address not found."
    except Exception as e:
        log.error(f"Location error: {e}")
        return "Unable to fetch your location."

# ── HELP ──────────────────────────────────────────────────────
def get_help():
    return (
        "Here is what I can do. "
        "Say what is this to describe the scene. "
        "Say read this to read text. "
        "Say how much to identify Indian currency. "
        "Say what objects to list objects. "
        "Say who am I to recognize your face. "
        "Say register face to save your face. "
        "Say what color to detect color. "
        "Say how far to estimate distance. "
        "Say where is the light to find light direction. "
        "Say where am I for your location. "
        "Say weather for current weather. "
        "Say what time or good morning for time and date. "
        "Say again to repeat. "
        "Say speak faster or slower to adjust speed. "
        "Say stop or goodbye to deactivate. "
        "Say help to hear this again."
    )

# ═══════════════════════════════════════════════════════
#   ROUTES
# ═══════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/startup_info")
def startup_info():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    weather_text = get_weather(float(lat), float(lon)) if lat and lon else ""
    briefing = get_briefing(lat, lon)
    return jsonify({
        "briefing": briefing,
        "weather":  weather_text,
        "time":     get_time(),
        "date":     get_date(),
        "models": {
            "blip":   _blip_model is not None,
            "intent": _use_semantic,
            "coco":   _coco_net is not None,
        }
    })

@app.route("/process", methods=["POST"])
def process():
    global pending_name, last_result, repeat_count

    cmd = request.form.get("command","").lower().strip()
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    log.info(f"CMD: '{cmd}'")

    if not cmd:
        return jsonify({"msg":"I did not hear that clearly. Please speak again.","intent":"error"})

    intent = get_intent(cmd)
    log.info(f"INTENT: {intent}")

    # ── SLEEP / DEACTIVATE ──
    if intent == "sleep":
        last_result = ""
        return jsonify({"msg":"Goodbye. Say Hey Assistant when you need me again.","intent":"sleep"})

    # ── REPEAT ──
    if intent == "repeat":
        if not last_result:
            return jsonify({"msg":"Nothing to repeat yet.","intent":"repeat"})
        repeat_count += 1
        if repeat_count >= REPEAT_LIMIT:
            repeat_count = 0
            return jsonify({"msg":"I have repeated that several times. Try a new command or say help.","intent":"repeat_limit"})
        add_memory(cmd, last_result, "repeat")
        return jsonify({"msg":last_result,"intent":"repeat"})
    else:
        repeat_count = 0

    # ── SPEECH RATE ──
    if intent == "faster":
        add_memory(cmd, "Speaking faster.", "faster")
        return jsonify({"msg":"Speaking faster now.","intent":"faster","rate_change":"faster"})
    if intent == "slower":
        add_memory(cmd, "Speaking slower.", "slower")
        return jsonify({"msg":"Speaking slower now.","intent":"slower","rate_change":"slower"})

    # ── NO-IMAGE INTENTS ──
    if intent == "help":
        msg = get_help()
        last_result = msg; add_memory(cmd, msg, intent)
        return jsonify({"msg":msg,"intent":"help"})

    if intent == "time":
        msg = get_time()
        last_result = msg; add_memory(cmd, msg, intent)
        return jsonify({"msg":msg,"intent":"time"})

    if intent == "date":
        msg = get_date()
        last_result = msg; add_memory(cmd, msg, intent)
        return jsonify({"msg":msg,"intent":"date"})

    if intent == "briefing":
        msg = get_briefing(lat, lon)
        last_result = msg; add_memory(cmd, msg, intent)
        return jsonify({"msg":msg,"intent":"briefing"})

    if intent == "weather":
        msg = get_weather(float(lat), float(lon)) if lat and lon else "Location not available for weather."
        last_result = msg; add_memory(cmd, msg, intent)
        return jsonify({"msg":msg,"intent":"weather"})

    if intent == "location":
        msg = get_address(lat, lon) if lat and lon else "Location access not granted."
        last_result = msg; add_memory(cmd, msg, intent)
        return jsonify({"msg":msg,"intent":"location"})

    # ── GET IMAGE ──
    if "image" not in request.files:
        return jsonify({"msg":"No image received. Please check the camera.","intent":"error"})

    with model_lock:
        img_bytes = request.files["image"].read()
        img = decode_image(img_bytes)
        if img is None:
            return jsonify({"msg":"Could not read image. Please try again.","intent":"error"})

        # ── PENDING FACE NAME ──
        if pending_name:
            name = cmd.strip()
            for prefix in ["my name is","i am","call me","name is"]:
                if name.startswith(prefix):
                    name = name[len(prefix):].strip()
            name = name.replace(" ","_").title() or "Unknown"
            msg = register_face(name, img)
            pending_name = False
            last_result = msg; add_memory(cmd, msg, "register_complete")
            return jsonify({"msg":msg,"intent":"register_complete"})

        # ── DANGER AUTO-CHECK ──
        danger = check_danger(img)
        if danger:
            last_result = danger; add_memory(cmd, danger, "danger")
            return jsonify({"msg":danger,"intent":"danger"})

        # ── IMAGE QUALITY ──
        if intent not in ["color","light"]:
            ok, reason = check_quality(img)
            if not ok:
                return jsonify({"msg":reason,"intent":"quality_error"})

        # ── MORE ──
        if intent == "more":
            msg = caption_scene(img, detailed=True)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"more"})

        if intent == "register":
            pending_name = True
            return jsonify({"msg":"Please say your name now.","intent":"register_prompt"})

        if intent == "face":
            msg = recognize_face(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"face"})

        if intent == "ocr":
            msg = read_text_gemini(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"ocr"})

        if intent == "scene":
            msg = caption_scene(img) + get_scene_type(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"scene"})

        if intent == "currency":
            msg = detect_currency_gemini(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"currency"})

        if intent == "objects":
            msg = detect_objects(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"objects"})

        if intent == "color":
            msg = detect_color(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"color"})

        if intent == "light":
            msg = detect_light(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"light"})

        if intent == "distance":
            msg = estimate_distance(img)
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"distance"})

        if intent == "danger":
            msg = check_danger(img) or "No obvious dangers detected in the scene."
            last_result = msg; add_memory(cmd, msg, intent)
            return jsonify({"msg":msg,"intent":"danger"})

        msg = "I did not understand that. Say help to hear all commands."
        return jsonify({"msg":msg,"intent":"unknown"})


@app.route("/health")
def health():
    return jsonify({
        "status":"running","time":datetime.now().isoformat(),
        "semantic_intent":_use_semantic,
        "models":{"blip":_blip_model is not None,"coco":_coco_net is not None},
        "faces":len(os.listdir(FACE_DB)) if os.path.exists(FACE_DB) else 0,
        "memory":len(conversation_ctx)
    })

@app.route("/faces")
def list_faces():
    if not os.path.exists(FACE_DB): return jsonify({"faces":[]})
    faces = [f.replace(".jpg","").replace("_"," ").title()
             for f in os.listdir(FACE_DB) if f.endswith(".jpg")]
    return jsonify({"faces":faces})

@app.route("/faces/<n>", methods=["DELETE"])
def delete_face(name):
    path = f"{FACE_DB}/{name}.jpg"
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"msg":f"{name} removed."})
    return jsonify({"msg":"Face not found."}), 404

@app.route("/memory")
def memory():
    return jsonify({"memory":conversation_ctx})

@app.route("/memory", methods=["DELETE"])
def clear_memory():
    global conversation_ctx; conversation_ctx = []
    return jsonify({"msg":"Memory cleared."})

# ── STARTUP ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ASSISTIVE AI v3 - STARTING")
    print("=" * 60)
    print(f"  Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log  : {log_file}")
    print(f"  URL  : http://localhost:5000")
    print("=" * 60)
    warmup_all()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)