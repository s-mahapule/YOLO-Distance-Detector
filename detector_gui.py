import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread, Event
import time, cv2 as cv, numpy as np, pyttsx3

# ==== CONFIG ====
CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
CLASSES = "classes.txt"

EACH_WIDTH = [16,12,12,12,12,12,12,12,12,12,15,16,10,18,5,8,15,20,14,20,22,18,20,20,10,20,12,3,12,12,12,15,10,8,5,5,10,10,10,3,4,4,1,1,1,3,2,3,4,3,4,1,2,5,5,8,16,20,6,25,20,10,10,10,2,2,9,3,12,12,10,10,12,8,11,12,4,16,12,4]
EACH_WIDTH_in_rf = [367,288,288,288,288,288,288,288,288,288,360,367,240,432,120,192,360,480,336,480,528,432,480,480,240,480,288,72,288,288,288,360,240,192,120,120,240,240,240,72,96,96,24,24,24,72,48,72,96,72,96,24,48,120,120,192,384,480,144,600,480,240,240,240,48,48,216,72,288,288,240,240,288,192,264,288,96,367,288,96]

STREET_IDS = [0, 1, 2, 3, 5, 7, 9, 11]
ANIMAL_IDS = [14, 15, 16, 20]

# ==== APP ====
class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Distance Detector (Smooth UI)")
        self.running = False
        self.stop_event = Event()
        self.thread = None
        self.voice_on = tk.BooleanVar(value=True)

        # UI
        ctrl = ttk.Frame(root); ctrl.pack(padx=5, pady=5)

        ttk.Label(ctrl, text="Mode:").grid(row=0, column=0)
        self.mode_var = tk.StringVar(value="Street")
        ttk.Combobox(ctrl, textvariable=self.mode_var, values=["Street","Animals","All"], width=10, state="readonly").grid(row=0, column=1)

        ttk.Label(ctrl, text="Conf:").grid(row=0, column=2)
        self.conf_scale = ttk.Scale(ctrl, from_=0.2, to=0.6, value=0.35, orient="horizontal", length=120)
        self.conf_scale.grid(row=0, column=3)

        ttk.Label(ctrl, text="Res:").grid(row=0, column=4)
        self.res_var = tk.IntVar(value=416)
        ttk.Combobox(ctrl, textvariable=self.res_var, values=[416,608], width=6, state="readonly").grid(row=0, column=5)

        ttk.Checkbutton(ctrl, text="Voice", variable=self.voice_on).grid(row=0, column=6)

        ttk.Button(ctrl, text="Start", command=self.start).grid(row=0, column=7, padx=3)
        ttk.Button(ctrl, text="Stop", command=self.stop).grid(row=0, column=8, padx=3)

        self.status = ttk.Label(root, text="Ready", relief="sunken", anchor="w")
        self.status.pack(fill="x")

        # vars
        self.engine = pyttsx3.init()
        self.last_spoken = {}
        self.speech_cooldown = 2.0
        self.cap = None
        self.model = None
        self.class_names = []
        self.focals = []

    def log(self, msg):
        self.status.config(text=msg)
        self.root.update_idletasks()

    def load_model(self):
        try:
            with open(CLASSES) as f:
                self.class_names = [c.strip() for c in f.readlines()]
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load {CLASSES}: {e}"); return False
        try:
            net = cv.dnn.readNet(WEIGHTS, CFG)
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            self.model = cv.dnn_DetectionModel(net)
            res = int(self.res_var.get())
            self.model.setInputParams(size=(res, res), scale=1/255, swapRB=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load YOLO: {e}"); return False
        self.focals = [(EACH_WIDTH_in_rf[i]*45)/EACH_WIDTH[i] if EACH_WIDTH[i] else None for i in range(len(EACH_WIDTH))]
        return True

    def _pos(self, rect, w, h):
        x,y,bw,bh = rect
        cx, cy = (2*x+bw)//2, (2*y+bh)//2
        horiz = "left" if cx <= w/3 else "center" if cx <= (w/3*2) else "right"
        vert  = "top" if cy <= h/3 else "mid" if cy <= (h/3*2) else "bottom"
        return f"{vert} {horiz}"

    def _speak(self, cid, text):
        now = time.time()
        if now - self.last_spoken.get(cid,0) < self.speech_cooldown:
            return
        try:
            self.engine.say(text); self.engine.runAndWait()
        except:
            pass
        self.last_spoken[cid] = now

    def start(self):
        if self.running: return
        if not self.load_model(): return
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error","Cannot open camera"); return
        self.stop_event.clear()
        self.running = True
        self.thread = Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.log("Running")

    def stop(self):
        self.stop_event.set()
        self.running = False
        time.sleep(0.2)
        if self.cap: self.cap.release()
        cv.destroyAllWindows()
        self.log("Stopped")

    def loop(self):
        conf_th = float(self.conf_scale.get())
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret: continue
            h, w = frame.shape[:2]
            classes, scores, boxes = self.model.detect(frame, conf_th, 0.3)
            seen = set()
            for cid, score, box in zip(np.array(classes).flatten(), np.array(scores).flatten(), boxes):
                cid = int(cid)
                mode = self.mode_var.get()
                if mode == "Street" and cid not in STREET_IDS: continue
                if mode == "Animals" and cid not in ANIMAL_IDS: continue
                name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
                x,y,bw,bh = box
                cv.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
                cv.putText(frame, f"{name} {score:.2f}", (x, y-8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                dist = None
                if cid < len(self.focals) and self.focals[cid] and bw>0:
                    dist = (EACH_WIDTH[cid] * self.focals[cid]) / bw
                    cv.putText(frame, f"{dist:.1f} in", (x, y+bh+18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                if self.voice_on.get() and cid not in seen:
                    pos = self._pos((x,y,bw,bh), w, h)
                    msg = f"{name} at {pos}" + (f", {int(dist)} inches" if dist else "")
                    self._speak(cid, msg)
                    seen.add(cid)
            cv.imshow("Detection", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    root = tk.Tk()
    DetectorApp(root)
    root.mainloop()
