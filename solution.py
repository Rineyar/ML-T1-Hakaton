import cv2
import numpy as np
import os
import threading
import time
import subprocess
import json
import keyboard
from rembg import remove
import psutil
HAVE_PSUTIL = True
from flask import Flask, Response
from PIL import Image, ImageDraw, ImageFont

    #КОНФИГИ
FOLDER = "backgrounds"
SHORT = 320
MASK_INTERVAL_MS = 120
JPEG_Q = 60
GPU_POLL_MS = 500
EMPLOYEE_JSON = "employee.json"

    #СПИСОК КАРТИНОК
files = [f for f in os.listdir(FOLDER)
         if f.lower().endswith((".jpg", ".png", ".jpeg"))]
if not files:
    raise RuntimeError("В папке ./backgrounds нет ни одного фона!")

bg_index = 0
bg = cv2.imread(os.path.join(FOLDER, files[bg_index]))

    #ИЗОБРАЖЕНИЕ С КАМЕРЫ
cap = cv2.VideoCapture(0)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

    #ФЛЯГА И ПЕРЕМЕННЫЕ-ФЛАГИ
app = Flask(__name__)

mask_cache = None
alpha_cache = None
mask_ready = False
mask_lock = threading.Lock()

latest_frame_for_mask = None
latest_frame_lock = threading.Lock()

frame_id = 0

last_gpu_query_time = 0.0
last_gpu_util = None
last_gpu_mem = None
gpu_lock = threading.Lock()

    #РАБОТА С JSON
EMPLOYEE = {}
EMPLOYEE_FIELDS_FOR_MEDIUM = [
    "full_name",
    "position",
    "company",
    "department",
    "office_location",
]

def load_employee_json(path=EMPLOYEE_JSON):
    global EMPLOYEE
    if not os.path.exists(path):
        EMPLOYEE = {}
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            EMPLOYEE = json.load(f)
            return
    except Exception:
        pass
    try:
        with open(path, "r", encoding="cp1251") as f:
            EMPLOYEE = json.load(f)
            return
    except Exception:
        EMPLOYEE = {}
        return

load_employee_json()

def employee_watcher(interval_s=5):
    last_mtime = None
    while True:
        try:
            if os.path.exists(EMPLOYEE_JSON):
                mtime = os.path.getmtime(EMPLOYEE_JSON)
                if last_mtime is None or mtime != last_mtime:
                    load_employee_json()
                    last_mtime = mtime
        except Exception:
            pass
        time.sleep(interval_s)

threading.Thread(target=employee_watcher, daemon=True).start()

    #РАБОТА С КЛАВИАТУРОЙ   
def kb_listener():
    global bg, bg_index
    if keyboard is None:
        return
    while True:
        try:
            keyboard.wait("space")
        except Exception:
            print("Требуется запуск от админа")
            return
        bg_index = (bg_index + 1) % len(files)
        bg = cv2.imread(os.path.join(FOLDER, files[bg_index]))
        print("BG -> ", files[bg_index])

if keyboard is not None:
    threading.Thread(target=kb_listener, daemon=True).start()

    #ВЫВОД ЗАГРУЖЕННОСТИ
def query_nvidia_smi_once():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if proc.returncode != 0:
            return None, None
        out = proc.stdout.strip().splitlines()
        if not out:
            return None, None
        first = out[0].strip()
        parts = [p.strip() for p in first.split(',')]
        if len(parts) >= 2:
            try:
                util = int(round(float(parts[0])))
            except:
                util = None
            try:
                mem = int(round(float(parts[1])))
            except:
                mem = None
            return util, mem
        else:
            return None, None
    except Exception:
        return None, None

    #РАБОТА С МАСКОЙ
def mask_worker():
    global mask_cache, alpha_cache, mask_ready, latest_frame_for_mask

    while True:
        time.sleep(MASK_INTERVAL_MS / 1000.0)
        with latest_frame_lock:
            frame_copy = None if latest_frame_for_mask is None else latest_frame_for_mask.copy()
        if frame_copy is None:
            continue

        h, w = frame_copy.shape[:2]
        rh = SHORT
        rw = int(w * rh / h)
        try:
            small = cv2.resize(frame_copy, (rw, rh), interpolation=cv2.INTER_LINEAR)
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("prepare small frame error:", e)
            continue
        try:
            out_rgba_small = remove(small_rgb)
        except Exception as e:
            print("rembg.remove error:", e)
            continue
        try:
            out_rgba_full = cv2.resize(out_rgba_small, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print("resize error:", e)
            continue

        fg_rgb = out_rgba_full[:, :, :3]
        alpha = out_rgba_full[:, :, 3]

        with mask_lock:
            mask_cache = fg_rgb.astype(np.uint8)
            alpha_cache = alpha.astype(np.uint8)
            mask_ready = True

threading.Thread(target=mask_worker, daemon=True).start()

    #JSON ВЫВОД
def put_text_unicode(img, text, pos, font_size=20, color=(255, 255, 255), outline_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, font_size)

    x, y = pos
    draw.text((x - 1, y - 1), text, font=font, fill=outline_color)
    draw.text((x + 1, y - 1), text, font=font, fill=outline_color)
    draw.text((x - 1, y + 1), text, font=font, fill=outline_color)
    draw.text((x + 1, y + 1), text, font=font, fill=outline_color)
    draw.text(pos, text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    #ПОЛУЧЕНИЕ КАДРА
def gen():
    global bg, mask_cache, alpha_cache, frame_id, latest_frame_for_mask, mask_ready
    global last_gpu_query_time, last_gpu_util, last_gpu_mem

    fps = 0.0
    t_prev = time.time()

    field_labels = {
        "full_name": "ФИО",
        "position": "Должность",
        "company": "Компания",
        "department": "Отдел",
        "office_location": "Офис"
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        with latest_frame_lock:
            latest_frame_for_mask = frame.copy()

        bg_resized = cv2.resize(bg, (w, h))

        with mask_lock:
            local_mask = None if mask_cache is None else mask_cache.copy()
            local_alpha = None if alpha_cache is None else alpha_cache.copy()
            local_ready = mask_ready

        now = time.time()
        dt = now - t_prev if t_prev is not None else 0.0
        t_prev = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = fps * 0.9 + inst_fps * 0.1 if fps > 0 else inst_fps
        frame_ms = dt * 1000.0

        cpu_text = "CPU n/a"
        if HAVE_PSUTIL:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_text = f"CPU {cpu_percent:.0f}%"
            except Exception:
                pass

        now_ms = int(round(now * 1000))
        do_query = False
        with gpu_lock:
            if (now_ms - int(round(last_gpu_query_time * 1000))) >= GPU_POLL_MS or last_gpu_util is None:
                do_query = True
        if do_query:
            util, mem = query_nvidia_smi_once()
            with gpu_lock:
                last_gpu_query_time = time.time()
                last_gpu_util = util
                last_gpu_mem = mem

        with gpu_lock:
            gpu_util_val = last_gpu_util
            gpu_mem_val = last_gpu_mem

        gpu_text = f"GPU {gpu_util_val}%" if gpu_util_val is not None else "GPU n/a"
        gpu_mem_text = f"GPU MEM {gpu_mem_val}MB" if gpu_mem_val is not None else "GPU MEM n/a"

        hud_lines = [f"{fps:5.1f} FPS", f"{frame_ms:5.1f} ms", cpu_text, gpu_text, gpu_mem_text]
        x, y0, line_h = 10, 40, 30

        emp_root = EMPLOYEE.get("employee", {}) if isinstance(EMPLOYEE, dict) else {}
        emp_lines = []
        for key in EMPLOYEE_FIELDS_FOR_MEDIUM:
            val = emp_root.get(key)
            label = field_labels.get(key, key)
            emp_lines.append(f"{label}: {val}" if val else f"{label}: -")

        if not local_ready or local_mask is None or local_alpha is None:
            img = frame
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            a = (local_alpha.astype(np.float32) / 255.0)[..., None]
            comp_rgb = (frame_rgb * a + bg_rgb * (1.0 - a)).astype(np.uint8)
            img = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)

        for i, line in enumerate(hud_lines):
            y = y0 + i * line_h
            img = put_text_unicode(img, line, (x, y), font_size=24)

        for j, eline in enumerate(emp_lines):
            y_emp = y0 + len(hud_lines) * line_h + j * 22
            img = put_text_unicode(img, eline, (x, y_emp), font_size=18)

        _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

@app.route('/')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    #ЗАПУСК
if __name__ == '__main__':
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    app.run(host='127.0.0.1', port=5000, threaded=True)