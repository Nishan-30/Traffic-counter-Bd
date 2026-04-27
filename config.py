# ================================================================
#  config.py  --  VELOXIS v1.0
#  Author: Nishan, SUST CEE, 2026
#  Product: NextCity Tessera
#  ASCII only -- no special characters
# ================================================================

# --- Model ---
# YOLO26 model trained on BD vehicle dataset
# Place bd_vehicles_yolo26.pt in same folder as app_windows.py
YOLO_MODEL  = "bd_vehicles_yolo11.pt"

# Fallback: uncomment if YOLO26 model not yet trained
# Fallback (old YOLOv8 model):
# YOLO_MODEL = "bd_vehicles_best.pt"

# YOLO26 confidence guide:
#   Daylight clear road   : 0.40 - 0.50
#   Night / low light     : 0.25 - 0.35
#   Crowded intersection  : 0.30 - 0.40
CONFIDENCE  = 0.35

# --- YOLO26 specific ---
# NMS-free by default (One-to-One head) -- faster deployment
# Set False only if you need maximum accuracy (enables NMS head)
YOLO26_NMS_FREE = True

# Tracker: botsort.yaml recommended for YOLO26 (better Re-ID)
# bytetrack.yaml also works
TRACKER_CONFIG = "botsort.yaml"

# --- Performance ---
FRAME_SKIP    = 1      # YOLO26 fast enough for 1 -- set 2 for slow CPU
RESIZE_BEFORE = True
RESIZE_WIDTH  = 640

# --- CPU Performance Mode (for laptops without dedicated GPU) ---
# HP Envy x360, Surface, or any Intel/AMD integrated graphics:
# Set CPU_PERFORMANCE_MODE = True for smooth real-time detection
# This enables: auto frame-skip, smaller inference resolution, half-precision skip
CPU_PERFORMANCE_MODE = True   # True = optimized for CPU/iGPU laptops
CPU_RESIZE_WIDTH     = 416    # smaller than 640 — faster on CPU (416 recommended)
CPU_FRAME_SKIP       = 2      # process every 2nd frame — halves CPU load
# Speed display still works — frame_skip is compensated in speed math

# --- Night enhancement ---
ENHANCE_NIGHT   = True
NIGHT_THRESHOLD = 60

# --- Vehicle classes (COCO fallback only) ---
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    6: "train",
}

# --- Tracking ---
MAX_AGE       = 20
MIN_HITS      = 2
IOU_THRESHOLD = 0.3

# --- Speed estimation ---
# Use Calibrate Speed page in VELOXIS for accurate homography.
# Typical px/m values:
#   Camera 3-4m high : 30-60  px/m
#   Camera 6-8m high : 15-30  px/m
#   Dashcam          : 80-150 px/m
# Set 0 to disable speed display.
PIXELS_PER_METER = 55
VIDEO_FPS        = 25

# --- Display ---
SHOW_SPEED    = True
SHOW_IDS      = True
SHOW_WINDOW   = True
WINDOW_WIDTH  = 960
DATA_FOLDER   = "data"

# --- Human / pedestrian detection ---
DETECT_HUMANS = True

# --- Counting line ---
COUNTING_LINE_POSITION = 0.55

# --- Dual counting lines (bidirectional roads) ---
USE_DUAL_LINES = False
LINE_POS_A     = 0.38
LINE_POS_B     = 0.70

# --- Zones ---
ENABLE_ZONES = False
ZONES = {
    "North": (0.0, 0.0, 1.0, 0.45),
    "South": (0.0, 0.55, 1.0, 1.0),
}
