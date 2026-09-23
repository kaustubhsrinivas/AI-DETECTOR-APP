"""
Multi-Modal Detection System
============================
- Object Detection (YOLOv8)
- Voice Emotion Detection (librosa + sklearn)
- Face Expression Detection (DeepFace / fer)

Run: python main.py
"""

import cv2
import numpy as np
import threading
import queue
import time
import sys
import os

# ── suppress TF/CUDA noise ──────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # use CPU; remove if you have GPU

# ─────────────────────────────────────────────────────────────────────────────
# OBJECT DETECTION  (YOLOv8 via ultralytics)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not found – object detection disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# FACE EXPRESSION DETECTION  (fer library)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("[WARN] fer not found – face expression detection disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# VOICE EMOTION DETECTION  (sounddevice + librosa + scikit-learn)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    import librosa
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("[WARN] sounddevice/librosa/sklearn not found – voice detection disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# VOICE EMOTION MODEL  (lightweight demo classifier)
# ─────────────────────────────────────────────────────────────────────────────
EMOTIONS_VOICE = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

def build_demo_voice_model():
    """
    Builds a *demo* RandomForest model seeded with synthetic feature vectors.
    Replace with a real trained model (e.g. trained on RAVDESS) for production.
    """
    np.random.seed(42)
    n = 300
    X, y = [], []
    for idx, emo in enumerate(EMOTIONS_VOICE):
        base = np.random.randn(n // len(EMOTIONS_VOICE), 40) * 0.5
        base[:, 0] += idx * 2          # separate classes by mean energy shift
        X.append(base)
        y += [emo] * (n // len(EMOTIONS_VOICE))
    X = np.vstack(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler

def extract_voice_features(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract 40-dim MFCC feature vector from raw audio."""
    if len(audio) == 0:
        return np.zeros(40)
    mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO CAPTURE THREAD
# ─────────────────────────────────────────────────────────────────────────────
class AudioCapture(threading.Thread):
    SAMPLE_RATE = 22050
    CHUNK       = 22050   # 1-second window

    def __init__(self, result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.q = result_queue
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                audio = sd.rec(self.CHUNK, samplerate=self.SAMPLE_RATE,
                               channels=1, dtype="float32")
                sd.wait()
                self.q.put(audio.flatten())
            except Exception as e:
                time.sleep(0.5)

    def stop(self):
        self._stop.set()

# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "object":     (0,   200, 255),
    "face":       (0,   255, 120),
    "voice":      (255, 180,   0),
    "panel_bg":   (15,   15,  15),
    "text":       (240, 240, 240),
}

def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, alpha=0.55):
    overlay = img.copy()
    x1, y1 = pt1; x2, y2 = pt2
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color,   thickness,   cv2.LINE_AA)

def draw_emotion_bar(img, emotion: str, confidence: float, y_pos: int,
                     bar_color, x_start=10, bar_width=180):
    label = f"{emotion}: {confidence:.0%}"
    draw_label(img, label, (x_start, y_pos - 4), COLORS["text"], 0.45)
    cv2.rectangle(img, (x_start, y_pos), (x_start + bar_width, y_pos + 8),
                  (60, 60, 60), -1)
    filled = int(bar_width * min(confidence, 1.0))
    cv2.rectangle(img, (x_start, y_pos), (x_start + filled, y_pos + 8),
                  bar_color, -1)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class MultiDetectionApp:
    WINDOW = "Multi-Modal Detection System  |  Q to quit"

    def __init__(self):
        # ── camera ──
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            sys.exit("[ERROR] Cannot open webcam.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ── object detection ──
        self.yolo = None
        if YOLO_AVAILABLE:
            print("[INFO] Loading YOLOv8n …")
            self.yolo = YOLO("yolov8n.pt")   # auto-downloads on first run

        # ── face expression ──
        self.fer_detector = FER(mtcnn=True) if FER_AVAILABLE else None

        # ── voice emotion ──
        self.voice_q     = queue.Queue(maxsize=3)
        self.voice_emotion = "–"
        self.voice_conf    = {}
        self.voice_thread  = None
        self.clf = self.scaler = None
        if VOICE_AVAILABLE:
            print("[INFO] Building demo voice model …")
            self.clf, self.scaler = build_demo_voice_model()
            self.voice_thread = AudioCapture(self.voice_q)
            self.voice_thread.start()
            threading.Thread(target=self._voice_worker, daemon=True).start()

        # ── state ──
        self.face_emotions: list  = []
        self.detected_objects: list = []
        self.fps = 0.0
        self._last_t = time.time()

    # ── voice inference loop (background thread) ──────────────────────────
    def _voice_worker(self):
        while True:
            try:
                audio = self.voice_q.get(timeout=2)
                feat  = extract_voice_features(audio).reshape(1, -1)
                feat  = self.scaler.transform(feat)
                proba = self.clf.predict_proba(feat)[0]
                idx   = np.argmax(proba)
                self.voice_emotion = self.clf.classes_[idx]
                self.voice_conf    = dict(zip(self.clf.classes_, proba))
            except queue.Empty:
                pass

    # ── process one frame ─────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Object detection
        self.detected_objects = []
        if self.yolo is not None:
            results = self.yolo(frame, verbose=False, conf=0.40)[0]
            for box in results.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                label = results.names[int(box.cls[0])]
                self.detected_objects.append((x1,y1,x2,y2, label, conf))

        # Face expression
        self.face_emotions = []
        if self.fer_detector is not None:
            try:
                result = self.fer_detector.detect_emotions(frame)
                for face in result:
                    (fx,fy,fw,fh) = face["box"]
                    emotions      = face["emotions"]
                    top_emo       = max(emotions, key=emotions.get)
                    top_conf      = emotions[top_emo]
                    self.face_emotions.append((fx,fy,fw,fh, top_emo, top_conf, emotions))
            except Exception:
                pass

        return frame

    # ── draw all annotations ─────────────────────────────────────────────
    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # ── object bounding boxes ──
        for (x1,y1,x2,y2, label, conf) in self.detected_objects:
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLORS["object"], 2)
            tag = f"{label} {conf:.0%}"
            tw, th = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), COLORS["object"], -1)
            cv2.putText(frame, tag, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        # ── face expression boxes ──
        for (fx,fy,fw,fh, top_emo, top_conf, all_emos) in self.face_emotions:
            cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), COLORS["face"], 2)
            tag = f"{top_emo} {top_conf:.0%}"
            tw, th = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.rectangle(frame, (fx, fy-th-10), (fx+tw+6, fy), COLORS["face"], -1)
            cv2.putText(frame, tag, (fx+3, fy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

        # ──────────────────────────────────────────────────────────────────
        # LEFT PANEL  – Voice Emotion
        # ──────────────────────────────────────────────────────────────────
        PW, PH = 210, 230
        draw_rounded_rect(frame, (5, 5), (5+PW, 5+PH), COLORS["panel_bg"],
                          radius=12, alpha=0.65)
        draw_label(frame, "VOICE EMOTION", (15, 28), COLORS["voice"], 0.55, 1)
        cv2.line(frame, (15, 34), (5+PW-10, 34), (80,80,80), 1)
        draw_label(frame, self.voice_emotion.upper(), (15, 56),
                   COLORS["voice"], 0.7, 2)

        y = 75
        for emo in EMOTIONS_VOICE:
            conf = self.voice_conf.get(emo, 0.0)
            draw_emotion_bar(frame, emo, conf, y, COLORS["voice"])
            y += 22

        # ──────────────────────────────────────────────────────────────────
        # RIGHT PANEL  – Face Expressions summary
        # ──────────────────────────────────────────────────────────────────
        if self.face_emotions:
            fx_, fy_, fw_, fh_, top_emo, top_conf, all_emos = self.face_emotions[0]
            PW2 = 210
            rx = w - PW2 - 10
            draw_rounded_rect(frame, (rx, 5), (rx+PW2, 5+PH),
                               COLORS["panel_bg"], radius=12, alpha=0.65)
            draw_label(frame, "FACE EXPRESSION", (rx+10, 28), COLORS["face"], 0.55, 1)
            cv2.line(frame, (rx+10, 34), (rx+PW2-10, 34), (80,80,80), 1)
            draw_label(frame, top_emo.upper(), (rx+10, 56), COLORS["face"], 0.7, 2)
            y = 75
            for emo, conf in sorted(all_emos.items(), key=lambda x: -x[1]):
                draw_emotion_bar(frame, emo, conf, y, COLORS["face"],
                                 x_start=rx+10, bar_width=170)
                y += 22

        # ──────────────────────────────────────────────────────────────────
        # BOTTOM BAR  – Objects + FPS
        # ──────────────────────────────────────────────────────────────────
        bar_y = h - 38
        draw_rounded_rect(frame, (5, bar_y), (w-5, h-5),
                          COLORS["panel_bg"], radius=8, alpha=0.65)

        obj_text = "Objects: " + (
            ", ".join(f"{lbl}({c:.0%})" for _,_,_,_,lbl,c in self.detected_objects[:6])
            if self.detected_objects else "none"
        )
        draw_label(frame, obj_text, (15, h-14), COLORS["object"], 0.5, 1)
        draw_label(frame, f"FPS {self.fps:.1f}", (w-90, h-14),
                   COLORS["text"], 0.5, 1)

        # ── top hint ──
        draw_label(frame, "Press  Q  to quit", (w//2-65, 22),
                   COLORS["text"], 0.5, 1)

        return frame

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1280, 720)
        print("\n[INFO] Running – press Q in the window to quit.\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Frame grab failed.")
                break

            t0 = time.time()
            frame = self.process_frame(frame)
            frame = self.draw_annotations(frame)

            self.fps = 0.9 * self.fps + 0.1 / max(time.time() - t0, 1e-6)

            cv2.imshow(self.WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

        self.cleanup()

    def cleanup(self):
        print("[INFO] Shutting down …")
        if self.voice_thread:
            self.voice_thread.stop()
        self.cap.release()
        cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MultiDetectionApp()
    app.run()
