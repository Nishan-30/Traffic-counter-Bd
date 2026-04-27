# ================================================================
#  detector.py  --  VELOXIS v1.0
#  Author : Nishan, SUST CEE, 2026
#  Product: NextCity Tessera
#
#  Detection engine upgraded to YOLO26:
#    - YOLO26 NMS-free end-to-end inference (faster deployment)
#    - BoTSORT tracker (better Re-ID than ByteTrack)
#    - ByteTrack automatic fallback
#    - Homography speed calibration (perspective-correct)
#    - DeepSORT fallback if ultralytics tracker unavailable
# ================================================================

import cv2, datetime, csv, os, math
import numpy as np
from ultralytics import YOLO
import config

# ── CPU / GPU auto-detection ──────────────────────────────────
import torch
_HAS_GPU = torch.cuda.is_available()
_CPU_MODE = getattr(config, "CPU_PERFORMANCE_MODE", False) or not _HAS_GPU

if _CPU_MODE:
    # Apply CPU-optimised settings automatically
    # Overrides FRAME_SKIP and RESIZE_WIDTH from config if CPU mode active
    _EFFECTIVE_FRAME_SKIP  = getattr(config, "CPU_FRAME_SKIP",   2)
    _EFFECTIVE_RESIZE_W    = getattr(config, "CPU_RESIZE_WIDTH", 416)
    print(f"[INFO] CPU Performance Mode ON — resize={_EFFECTIVE_RESIZE_W}, skip={_EFFECTIVE_FRAME_SKIP}")
    print(f"[INFO] GPU detected: {_HAS_GPU}. For best speed use a dedicated GPU.")
else:
    _EFFECTIVE_FRAME_SKIP  = getattr(config, "FRAME_SKIP",    1)
    _EFFECTIVE_RESIZE_W    = getattr(config, "RESIZE_WIDTH", 640)
    gpu_name = torch.cuda.get_device_name(0) if _HAS_GPU else "CPU"
    print(f"[INFO] Running on: {gpu_name}")

# ── Tracker availability check ────────────────────────────────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort
    _HAS_DEEPSORT = True
except ImportError:
    _HAS_DEEPSORT = False

# ── Zone loader ───────────────────────────────────────────────
try:
    from lane_tool import load_polygon_zones, point_in_polygon
    _POLYGON_ZONES = load_polygon_zones()
except Exception:
    _POLYGON_ZONES = None
    def point_in_polygon(px, py, pts, w, h): return False

# ── Colours per vehicle type ──────────────────────────────────
COLOURS = {
    "car":             ( 57, 197, 187),
    "motorcycle":      (255,  80,  30),
    "rickshaw":        (255, 180,  20),
    "rickshaw/CNG":    (255, 180,  20),
    "cng":             (251, 146,  60),
    "CNG/auto":        (251, 146,  60),
    "bus":             ( 80, 130, 240),
    "truck":           (160,  80, 220),
    "bicycle":         ( 80, 220,  80),
    "easybike":        (100, 220, 200),
    "battery_rickshaw":(255, 220, 100),
    "human_hauler":    (200, 100, 255),
    "leguna":          (100, 200, 255),
    "nosimon":         (255, 150, 100),
    "microbus":        ( 80, 180, 180),
    "pickup":          (180, 130,  80),
    "tempo":           (220, 180,  60),
    "train":           ( 80, 160, 220),
}
DEFAULT_COLOUR  = (180, 180, 180)
DIR_FORWARD     = "FWD"
DIR_BACKWARD    = "BWD"
NEAR_MISS_PX    = 25
NEAR_MISS_SPEED = 8
BRAKE_DROP_KMH  = 18


def _corrected_vtype(raw_cls, box_w, box_h, frame_w, frame_h, class_names=None):
    """
    Resolve detection class index to vehicle type string.
    Custom model class names take priority over COCO fallback.
    """
    if class_names and raw_cls in class_names:
        name = str(class_names[raw_cls])
        # Heuristic: small area 'truck' detections are likely CNGs
        if name == "truck":
            if (box_w * box_h) / max(frame_w * frame_h, 1) < 0.04:
                return "cng"
        return name
    # COCO fallback
    vtype = config.VEHICLE_CLASSES.get(raw_cls, "vehicle")
    if vtype == "motorcycle":
        return "rickshaw" if box_w / max(box_h, 1) >= 1.4 else "motorcycle"
    if vtype == "truck":
        if (box_w * box_h) / max(frame_w * frame_h, 1) < 0.04:
            return "cng"
    return vtype


class VehicleDetector:

    def __init__(self, session_label="session"):
        model_name = config.YOLO_MODEL
        print(f"[INFO] Loading model: {model_name}")
        # Check if model file exists before loading
        if not os.path.exists(model_name):
            # Try fallback models in order
            fallbacks = ["bd_vehicles_best.pt","yolo11s.pt","yolov8s.pt","yolov8n.pt"]
            found = None
            for fb in fallbacks:
                if os.path.exists(fb):
                    found = fb
                    break
            if found:
                print(f"[WARN] {model_name} not found. Using fallback: {found}")
                model_name = found
            else:
                print(f"[WARN] {model_name} not found. Downloading yolo11s.pt from ultralytics...")
                model_name = "yolo11s.pt"
        self.model = YOLO(model_name)
        print(f"[INFO] Model loaded: {model_name}")
        self._class_names = (
            self.model.names
            if hasattr(self.model, "names") and self.model.names
            else config.VEHICLE_CLASSES
        )

        # ── YOLO26 tracker setup ──────────────────────────────
        # BoTSORT tracker — tuned for BD mixed traffic (reduces ID switches)
        # Write a custom botsort config with tighter thresholds
        self._tracker_cfg = self._setup_tracker()
        print(f"[INFO] Tracker: BoTSORT (custom config — low ID-switch mode)")

        # DeepSORT fallback (only if ultralytics tracker completely fails)
        self._deepsort = None
        if _HAS_DEEPSORT:
            self._deepsort = _DeepSort(
                max_age=20, n_init=3,
                nms_max_overlap=0.6, max_cosine_distance=0.25)

        # ── Homography speed calibration ──────────────────────
        self.H_matrix = None
        self._load_homography()

        # ── Counters ──────────────────────────────────────────
        self.counted_ids   = set()
        self.total_counts  = {}
        self.dir_counts    = {DIR_FORWARD: {}, DIR_BACKWARD: {}}
        self.zone_counts   = {}
        self.speed_history = {}
        self.prev_speeds   = {}
        self.prev_cy       = {}
        self._counted_fwd  = set()
        self.frame_no      = 0
        self.session_label = session_label

        # ── Re-ID cache (prevents double-counting re-entrants) ─
        self._reid_cache   = {}
        self._reid_mapped  = {}
        self._reid_max_age = 90  # frames (~3s at 30fps)

        # ── Queue detection ───────────────────────────────────
        self.queue_length        = 0
        self.queue_history       = []
        self.queue_threshold_kmh = 3.0

        # ── Peak hour (15-min intervals) ──────────────────────
        self._interval_start = datetime.datetime.now()
        self._interval_count = 0
        self.peak_intervals  = []
        self.peak_rate       = 0
        self.current_rate    = 0

        # ── Miovision-style advanced metrics ──────────────────
        # Peak Hour Factor (PHF) = total volume / (4 x peak 15-min volume)
        # Standard intersection capacity analysis metric
        self.phf             = 0.0   # 0.0–1.0, ideal ~0.85–0.95

        # Turning movement counts: approach x turn direction
        # Keys: "N_L","N_T","N_R","S_L","S_T","S_R","E_L","E_T","E_R","W_L","W_T","W_R"
        # Populated only when 4-approach zones are drawn (Lane Drawing page)
        self.turning_counts  = {}

        # Approach volume (vehicles per approach arm)
        # Used for v/c ratio and LOS calculation
        self.approach_counts = {}   # {"North": n, "South": n, ...}

        # Level of Service (LOS) — HCM 6th edition thresholds
        # Based on avg control delay per vehicle (seconds)
        # A:<10  B:10-20  C:20-35  D:35-55  E:55-80  F:>80
        self.los_letter      = "—"
        self.avg_delay_sec   = 0.0

        # Headway tracking (time between consecutive vehicles crossing line)
        self._last_cross_time = {}   # vtype → last crossing datetime
        self.headway_history  = []   # list of (vtype, headway_sec)
        self.avg_headway_sec  = 0.0

        # Saturation flow tracking (vehicles per green hour equivalent)
        # Approximated from observed headway
        self.saturation_flow = 0    # veh/hr

        # Speed percentiles (for traffic engineering reports)
        self.all_speeds      = []   # all observed speed samples
        self.speed_85th      = 0.0  # 85th percentile speed (design speed reference)
        self.speed_mean      = 0.0

        # ── Live stats ────────────────────────────────────────
        self.live_vehicles   = 0
        self.occupancy_pct   = 0.0
        self.person_count    = 0
        self.safety_events   = 0
        self.density_history = []
        self.near_miss_log   = []

        # ── Line settings (set by app) ────────────────────────
        self.ai_line_start = None
        self.ai_line_end   = None
        self.manual_line_a = None

        # ── CSV log ───────────────────────────────────────────
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(config.DATA_FOLDER, f"log_{ts}.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "track_id", "vehicle_type",
                "zone", "direction", "speed_kmh", "session",
                # Extended columns — session-level metrics updated per crossing
                "queue_length", "occupancy_pct", "current_rate_vph",
                "avg_headway_sec", "saturation_flow", "phf",
                "speed_85th_kmh", "speed_mean_kmh"])
        print(f"[INFO] Log -> {self.csv_path}")

    # ── Tracker setup ─────────────────────────────────────────
    def _setup_tracker(self):
        """
        Write a tuned BoTSORT config to reduce ID switches in
        BD mixed-traffic intersections (dense, slow-moving, occluded).

        Key tuning:
          track_high_thresh: lower = track more detections (fewer lost IDs)
          new_track_thresh:  higher = don't start tracks on weak detections
          track_buffer:      longer = keep lost tracks longer before deleting
          match_thresh:      lower = stricter match (fewer wrong Re-ID)
          with_reid:         True = use appearance features for Re-ID
        """
        import yaml as _yaml
        cfg = {
            "tracker_type":       "botsort",
            "track_high_thresh":  0.30,   # was 0.5 — track more confidently detected vehicles
            "track_low_thresh":   0.15,   # low conf detections still considered for matching
            "new_track_thresh":   0.35,   # min conf to START a new track
            "track_buffer":       45,     # frames to keep lost track (~1.5s at 30fps)
            "match_thresh":       0.70,   # IoU match threshold — stricter to avoid wrong match
            "fuse_score":         True,   # fuse detection score into match
            "with_reid":          True,   # use appearance Re-ID (key for ID consistency)
            "proximity_thresh":   0.5,
            "appearance_thresh":  0.25,   # stricter appearance match
            "cmc_method":         "sparseOptFlow",  # camera motion compensation
            "frame_rate":         30,
        }
        tracker_path = os.path.join(
            getattr(config, "DATA_FOLDER", "data"), "botsort_veloxis.yaml")
        os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
        with open(tracker_path, "w") as f:
            _yaml.dump(cfg, f)
        return tracker_path

    # ── Homography ────────────────────────────────────────────
    def _load_homography(self):
        hpath = os.path.join(getattr(config, "DATA_FOLDER", "data"), "homography.npy")
        if os.path.exists(hpath):
            try:
                self.H_matrix = np.load(hpath)
                print(f"[INFO] Homography loaded from {hpath}")
            except Exception:
                self.H_matrix = None

    def calibrate_homography(self, image_pts, world_pts):
        img = np.float32(image_pts)
        wld = np.float32(world_pts)
        H, mask = cv2.findHomography(img, wld, cv2.RANSAC, 5.0)
        if H is not None:
            self.H_matrix = H
            hpath = os.path.join(getattr(config, "DATA_FOLDER", "data"), "homography.npy")
            os.makedirs(os.path.dirname(hpath), exist_ok=True)
            np.save(hpath, H)
            print(f"[INFO] Homography calibrated. Inliers: {mask.sum()}/{len(image_pts)}")
            return True
        return False

    def pixel_to_world(self, px, py):
        if self.H_matrix is None:
            return None
        pt = np.float32([[px, py]]).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pt, self.H_matrix)
        return float(world[0][0][0]), float(world[0][0][1])

    # ── Main processing ───────────────────────────────────────
    def process_frame(self, frame):
        self.frame_no += 1
        h, w = frame.shape[:2]

        # Frame skip — uses CPU-optimised value automatically
        skip = _EFFECTIVE_FRAME_SKIP
        if skip > 1 and self.frame_no % skip != 0:
            if hasattr(self, "_last_ann"):
                return self._last_ann, self._last_sum
            return frame, self._empty_summary()

        # Night enhancement (CLAHE)
        if getattr(config, "ENHANCE_NIGHT", False):
            if frame.mean() < getattr(config, "NIGHT_THRESHOLD", 60):
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
                frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Resize for detection — uses CPU-optimised resolution automatically
        detect_frame = frame
        coord_scale  = 1.0
        if getattr(config, "RESIZE_BEFORE", True):
            rw = _EFFECTIVE_RESIZE_W
            if w > rw:
                s = rw / w
                detect_frame = cv2.resize(frame, (rw, int(h * s)))
                coord_scale  = 1.0 / s

        # ── YOLO26 tracking ───────────────────────────────────
        # model.track() runs YOLO26 + tracker in a single call.
        # YOLO26 NMS-free head: no post-processing needed.
        # persist=True keeps track IDs consistent across frames.
        tracks = []
        self.person_count = 0
        try:
            track_results = self.model.track(
                detect_frame,
                persist   = True,
                verbose   = False,
                conf      = config.CONFIDENCE,
                iou       = 0.3,
                tracker   = self._tracker_cfg,
            )[0]

            if track_results.boxes.id is not None:
                for box, tid, cls_id, conf_v in zip(
                    track_results.boxes.xyxy,
                    track_results.boxes.id,
                    track_results.boxes.cls,
                    track_results.boxes.conf,
                ):
                    cls_int = int(cls_id)
                    # Person (COCO class 0) — draw separately
                    if cls_int == 0:
                        if getattr(config, "DETECT_HUMANS", True):
                            x1, y1, x2, y2 = [int(v * coord_scale) for v in box.tolist()]
                            self.person_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 1)
                            cv2.putText(frame, "person", (x1, max(y1 - 4, 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 100, 100), 1)
                        continue
                    x1, y1, x2, y2 = [int(v * coord_scale) for v in box.tolist()]
                    tracks.append({
                        "tid":  int(tid),
                        "ltrb": (x1, y1, x2, y2),
                        "cls":  cls_int,
                        "conf": float(conf_v),
                    })

        except Exception as e:
            # Fallback: run standard inference then DeepSORT
            print(f"[WARN] YOLO26 tracker failed ({e}), falling back to DeepSORT")
            try:
                results = self.model(detect_frame, verbose=False, conf=config.CONFIDENCE)[0]
                detections = []
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        if getattr(config, "DETECT_HUMANS", True):
                            x1, y1, x2, y2 = [int(v * coord_scale) for v in box.xyxy[0].tolist()]
                            self.person_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 1)
                            cv2.putText(frame, "person", (x1, max(y1 - 4, 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 100, 100), 1)
                        continue
                    x1, y1, x2, y2 = [int(v * coord_scale) for v in box.xyxy[0].tolist()]
                    bw, bh = x2 - x1, y2 - y1
                    detections.append(([x1, y1, bw, bh], float(box.conf[0]), cls))
                detections = self._nms(detections)
                if self._deepsort:
                    ds_tracks = self._deepsort.update_tracks(detections, frame=frame)
                    for t in ds_tracks:
                        if not t.is_confirmed(): continue
                        l2, t2, r2, b2 = map(int, t.to_ltrb())
                        tracks.append({
                            "tid":  t.track_id,
                            "ltrb": (l2, t2, r2, b2),
                            "cls":  getattr(t, "det_class", 2),
                            "conf": 0.5,
                        })
            except Exception as e2:
                print(f"[ERROR] Fallback also failed: {e2}")

        lines = self._get_lines(w, h)

        # ── Per-track processing ──────────────────────────────
        for trk in tracks:
            tid     = trk["tid"]
            l, t, r, b = trk["ltrb"]
            l, t, r, b = max(l, 0), max(t, 0), min(r, w - 1), min(b, h - 1)
            cx, cy  = (l + r) // 2, (t + b) // 2
            bw, bh  = r - l, b - t
            cls_int = trk["cls"]

            vtype  = _corrected_vtype(cls_int, bw, bh, w, h, self._class_names)
            colour = COLOURS.get(vtype, DEFAULT_COLOUR)

            speed_kmh = self._estimate_speed(tid, cx, cy, self.frame_no)

            # Re-ID: map re-entering vehicles to original track
            effective_tid = self._reid_lookup(tid, cx, cy, w, h, vtype)

            # Sudden brake detection
            prev_sp = self.prev_speeds.get(effective_tid, 0)
            if speed_kmh and prev_sp > 15 and (prev_sp - speed_kmh) > BRAKE_DROP_KMH:
                self.safety_events += 1
                cv2.putText(frame, "! BRAKE", (cx - 28, t - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            if speed_kmh:
                self.prev_speeds[effective_tid] = speed_kmh

            # Direction (FWD = moving down frame, BWD = moving up)
            prev_cy_ = self.prev_cy.get(effective_tid)
            direction = DIR_FORWARD if (prev_cy_ is None or cy >= prev_cy_) else DIR_BACKWARD
            self.prev_cy[effective_tid] = cy

            # Queue tint: blue if slow and behind line
            lp1_main, lp2_main, _ = lines[0]
            line_y = (lp1_main[1] + lp2_main[1]) // 2
            if cy > line_y and (speed_kmh or 0) < self.queue_threshold_kmh:
                colour = (100, 100, 255)

            # Line crossing → count
            for lp1, lp2, line_label in lines:
                if self._crosses_line(f"{effective_tid}_{line_label}", cx, cy, lp1, lp2):
                    if effective_tid not in self.counted_ids:
                        self.counted_ids.add(effective_tid)
                        if direction == DIR_FORWARD:
                            self._counted_fwd.add(effective_tid)
                        self._reid_cache[effective_tid] = {
                            "cx_n": cx / max(w, 1), "cy_n": cy / max(h, 1),
                            "vtype": vtype, "frame": self.frame_no
                        }
                        zone = self._get_zone(cx, cy, w, h) if config.ENABLE_ZONES else "all"
                        self.total_counts[vtype] = self.total_counts.get(vtype, 0) + 1
                        self.dir_counts[direction][vtype] = \
                            self.dir_counts[direction].get(vtype, 0) + 1
                        self._interval_count += 1
                        if config.ENABLE_ZONES:
                            self.zone_counts.setdefault(zone, {})
                            self.zone_counts[zone][vtype] = \
                                self.zone_counts[zone].get(vtype, 0) + 1
                        self._log(effective_tid, vtype, zone, direction, speed_kmh)
                    break

            # Draw bounding box
            cv2.rectangle(frame, (l, t), (r, b), colour, 2)
            parts = [vtype.split("/")[0]]
            if config.SHOW_IDS:
                parts.append(f"#{tid}")
            if getattr(config, "SHOW_SPEED", True) and speed_kmh:
                parts.append(f"{speed_kmh:.0f}km/h")
            if effective_tid in self.counted_ids:
                parts.append("FWD" if effective_tid in self._counted_fwd else "BWD")
            lbl = " ".join(parts)
            lx = max(l, 0); ly_top = max(t - 20, 0)
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.rectangle(frame, (lx, ly_top), (lx + tw + 6, t), colour, -1)
            cv2.putText(frame, lbl, (lx + 3, max(t - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Near-miss detection (capped at 20 tracks) ─────────
        if len(tracks) <= 20:
            pos = [(trk2["tid"],
                    (trk2["ltrb"][0] + trk2["ltrb"][2]) // 2,
                    (trk2["ltrb"][1] + trk2["ltrb"][3]) // 2)
                   for trk2 in tracks]
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    ta, ax, ay = pos[i]; tb, bx, by = pos[j]
                    if math.hypot(ax - bx, ay - by) < NEAR_MISS_PX:
                        sp_a = self._estimate_speed(ta, ax, ay, self.frame_no) or 0
                        sp_b = self._estimate_speed(tb, bx, by, self.frame_no) or 0
                        if sp_a > NEAR_MISS_SPEED and sp_b > NEAR_MISS_SPEED:
                            pair = tuple(sorted([ta, tb]))
                            if not any(e[1] == pair and self.frame_no - e[0] < 90
                                       for e in self.near_miss_log):
                                self.near_miss_log.append((self.frame_no, pair))
                                self.safety_events += 1
                                cv2.line(frame, (ax, ay), (bx, by), (0, 0, 255), 1)
                                cv2.putText(frame, "NEAR-MISS",
                                            ((ax + bx) // 2, (ay + by) // 2 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            if len(self.near_miss_log) > 100:
                self.near_miss_log = self.near_miss_log[-50:]

        # ── Live stats ────────────────────────────────────────
        vehicle_tracks = tracks  # already excludes person class
        self.live_vehicles = len(vehicle_tracks)
        road_area = max(w * h * 0.6, 1)
        occ_px = sum(
            max(trk2["ltrb"][2] - trk2["ltrb"][0], 0) *
            max(trk2["ltrb"][3] - trk2["ltrb"][1], 0)
            for trk2 in vehicle_tracks)
        self.occupancy_pct = min(100.0, occ_px / road_area * 100)
        self.density_history.append((self.frame_no, self.live_vehicles))
        if len(self.density_history) > 300:
            self.density_history = self.density_history[-150:]

        # ── Queue length ──────────────────────────────────────
        lp1_q, lp2_q, _ = lines[0]
        line_y_q = (lp1_q[1] + lp2_q[1]) // 2
        self.queue_length = sum(
            1 for trk2 in vehicle_tracks
            if (trk2["ltrb"][1] + trk2["ltrb"][3]) // 2 > line_y_q
            and (self._estimate_speed(trk2["tid"],
                 (trk2["ltrb"][0] + trk2["ltrb"][2]) // 2,
                 (trk2["ltrb"][1] + trk2["ltrb"][3]) // 2,
                 self.frame_no) or 999) < self.queue_threshold_kmh
        )
        self.queue_history.append((self.frame_no, self.queue_length))
        if len(self.queue_history) > 600:
            self.queue_history = self.queue_history[-300:]

        if self.queue_length > 0:
            cv2.putText(frame, f"Q:{self.queue_length}",
                        (lp1_q[0] + 4, lp1_q[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1)

        # ── Peak hour (15-min intervals) + PHF ───────────────
        now = datetime.datetime.now()
        elapsed_min = (now - self._interval_start).total_seconds() / 60
        if elapsed_min >= 15:
            rate = round(self._interval_count / (elapsed_min / 60))
            self.peak_intervals.append((self._interval_start, self._interval_count, rate))
            if rate > self.peak_rate:
                self.peak_rate = rate
            self.current_rate = rate
            # Peak Hour Factor: PHF = hourly_vol / (4 × peak_15min_vol)
            # Uses last 4 intervals (1 hour window)
            if len(self.peak_intervals) >= 4:
                last4 = [p[2] for p in self.peak_intervals[-4:]]
                total_hr = sum(last4)
                peak_15  = max(last4)
                self.phf = round(total_hr / (4 * peak_15), 3) if peak_15 > 0 else 0.0
            self._interval_start = now
            self._interval_count = 0
        elif elapsed_min > 0:
            self.current_rate = round(self._interval_count / (elapsed_min / 60))

        # ── Prune Re-ID cache ─────────────────────────────────
        stale = [k for k, v in self._reid_cache.items()
                 if self.frame_no - v["frame"] > self._reid_max_age]
        for k in stale:
            del self._reid_cache[k]

        # ── Draw counting lines ───────────────────────────────
        fwd_c = sum(self.dir_counts.get(DIR_FORWARD, {}).values())
        bwd_c = sum(self.dir_counts.get(DIR_BACKWARD, {}).values())
        lcols = [(57, 197, 187), (255, 180, 40)]
        for i, (lp1, lp2, _) in enumerate(lines):
            col = lcols[i % 2]
            cv2.line(frame, lp1, lp2, col, 2)
            mx = (lp1[0] + lp2[0]) // 2; my = (lp1[1] + lp2[1]) // 2
            if len(lines) == 1:
                cv2.arrowedLine(frame, (mx - 22, my - 10), (mx - 22, my + 10),
                                (57, 197, 187), 2, tipLength=0.4)
                cv2.putText(frame, f"FWD:{fwd_c}", (mx - 48, my - 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (57, 197, 187), 1)
                cv2.arrowedLine(frame, (mx + 22, my + 10), (mx + 22, my - 10),
                                (255, 180, 40), 2, tipLength=0.4)
                cv2.putText(frame, f"BWD:{bwd_c}", (mx + 4, my - 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 40), 1)
            else:
                arr = ((mx, my - 12), (mx, my + 12)) if i == 0 else ((mx, my + 12), (mx, my - 12))
                cv2.arrowedLine(frame, arr[0], arr[1], col, 2, tipLength=0.4)
                cnt = fwd_c if i == 0 else bwd_c
                tag = f"{'FWD' if i==0 else 'BWD'}:{cnt}"
                cv2.rectangle(frame, (lp1[0] + 2, lp1[1] - 18),
                              (lp1[0] + len(tag) * 8 + 4, lp1[1] - 3), (10, 20, 30), -1)
                cv2.putText(frame, tag, (lp1[0] + 4, lp1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        if config.ENABLE_ZONES:
            self._draw_zones(frame, w, h)

        frame = self._draw_hud(frame)

        summary = {
            "total_unique":   len(self.counted_ids),
            "by_type":        self.total_counts,
            "by_zone":        self.zone_counts,
            "by_direction":   self.dir_counts,
            "frame":          self.frame_no,
            "live_vehicles":  self.live_vehicles,
            "occupancy_pct":  round(self.occupancy_pct, 1),
            "person_count":   self.person_count,
            "safety_events":  self.safety_events,
            "near_miss_log":  self.near_miss_log[-5:],
            "queue_length":   self.queue_length,
            "current_rate":   self.current_rate,
            "peak_rate":      self.peak_rate,
            # Miovision-style
            "phf":            self.phf,
            "avg_headway_sec":self.avg_headway_sec,
            "saturation_flow":self.saturation_flow,
            "speed_85th":     self.speed_85th,
            "speed_mean":     self.speed_mean,
            "approach_counts":self.approach_counts,
        }
        self._last_ann = frame
        self._last_sum = summary
        return frame, summary

    # ── Helper methods ────────────────────────────────────────

    def _empty_summary(self):
        return {
            "total_unique":  len(self.counted_ids),
            "by_type":       self.total_counts,
            "by_zone":       self.zone_counts,
            "by_direction":  self.dir_counts,
            "frame":         self.frame_no,
            "live_vehicles": self.live_vehicles,
            "occupancy_pct": round(self.occupancy_pct, 1),
            "person_count":  self.person_count,
            "safety_events": self.safety_events,
            "near_miss_log": [],
            "queue_length":  self.queue_length,
            "current_rate":  self.current_rate,
            "peak_rate":     self.peak_rate,
        }

    def _reid_lookup(self, tid, cx, cy, w, h, vtype):
        if tid in self._reid_mapped:
            return self._reid_mapped[tid]
        if tid in self.counted_ids:
            return tid
        cx_n = cx / max(w, 1); cy_n = cy / max(h, 1)
        POSITION_THRESH = 0.12
        for orig_tid, entry in self._reid_cache.items():
            if entry["vtype"] != vtype: continue
            dist = math.hypot(cx_n - entry["cx_n"], cy_n - entry["cy_n"])
            if dist < POSITION_THRESH:
                self._reid_mapped[tid] = orig_tid
                return orig_tid
        return tid

    def _get_lines(self, w, h):
        if getattr(config, "USE_DUAL_LINES", False):
            a = getattr(self, "manual_line_a", None) or getattr(config, "LINE_POS_A", 0.38)
            b = getattr(config, "LINE_POS_B", 0.70)
            return [((0, int(h * a)), (w, int(h * a)), "A"),
                    ((0, int(h * b)), (w, int(h * b)), "B")]
        if self.manual_line_a is not None:
            ly = int(h * self.manual_line_a)
        elif self.ai_line_start and self.ai_line_end:
            p1 = (int(self.ai_line_start[0] * w), int(self.ai_line_start[1] * h))
            p2 = (int(self.ai_line_end[0] * w),   int(self.ai_line_end[1] * h))
            return [(p1, p2, "A")]
        else:
            ly = int(h * config.COUNTING_LINE_POSITION)
        return [((0, ly), (w, ly), "A")]

    def _crosses_line(self, key, cx, cy, p1, p2):
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = max(math.sqrt(dx * dx + dy * dy), 1)
        def sd(px, py): return (dx * (py - p1[1]) - dy * (px - p1[0])) / length
        prev = getattr(self, f"_d_{key}", None)
        curr = sd(cx, cy)
        setattr(self, f"_d_{key}", curr)
        if prev is None: return False
        if prev * curr < 0: return True
        band = 10.0
        was_in = getattr(self, f"_b_{key}", False)
        now_in = abs(curr) < band
        setattr(self, f"_b_{key}", now_in)
        return was_in and not now_in

    def _estimate_speed(self, tid, cx, cy, frame_no):
        # Use effective frame skip so speed is correct even in CPU mode
        sk  = max(_EFFECTIVE_FRAME_SKIP, 1)
        fps = max(getattr(config, "VIDEO_FPS", 25), 1)
        hist = self.speed_history.setdefault(tid, [])

        if self.H_matrix is not None:
            world = self.pixel_to_world(cx, cy)
            if world is None: return None
            xm, ym = world
            hist.append((xm, ym, frame_no))
            if len(hist) < 8: return None
            speeds = []
            for i in range(1, min(len(hist), 8)):
                ox, oy, ofn = hist[-i - 1]; nx, ny, nfn = hist[-i]
                dt = max(nfn - ofn, 1) * sk / fps
                kmh = math.hypot(nx - ox, ny - oy) / dt * 3.6
                if 0.3 < kmh < 120: speeds.append(kmh)
        else:
            ppm = getattr(config, "PIXELS_PER_METER", 0)
            if ppm <= 0: return None
            hist.append((cx, cy, frame_no))
            if len(hist) < 8: return None
            speeds = []
            for i in range(1, min(len(hist), 8)):
                ox, oy, ofn = hist[-i - 1]; nx, ny, nfn = hist[-i]
                dt = max(nfn - ofn, 1) * sk / fps
                kmh = math.hypot(nx - ox, ny - oy) / ppm / dt * 3.6
                if 0.5 < kmh < 120: speeds.append(kmh)

        if len(hist) > 30:
            self.speed_history[tid] = hist[-15:]
        if not speeds: return None
        return round(sorted(speeds)[len(speeds) // 2], 1)

    def _get_zone(self, cx, cy, w, h):
        if _POLYGON_ZONES:
            for lane in _POLYGON_ZONES:
                if point_in_polygon(cx, cy, lane["points"], w, h):
                    return lane["name"]
        for name, (x1, y1, x2, y2) in config.ZONES.items():
            if x1 * w <= cx <= x2 * w and y1 * h <= cy <= y2 * h:
                return name
        return "all"

    @staticmethod
    def _nms(detections, iou_thresh=0.45):
        if len(detections) < 2: return detections
        boxes  = [[d[0][0], d[0][1], d[0][0] + d[0][2], d[0][1] + d[0][3]] for d in detections]
        scores = [d[1] for d in detections]
        idxs   = cv2.dnn.NMSBoxes(
            [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
            scores, score_threshold=0.0, nms_threshold=iou_thresh)
        if len(idxs) == 0: return detections
        flat = [int(i) for i in (idxs.flatten() if hasattr(idxs, "flatten") else idxs)]
        return [detections[i] for i in flat]

    def _draw_zones(self, frame, w, h):
        cols = [(255, 165, 30), (57, 197, 187), (220, 80, 80), (130, 80, 220), (80, 200, 80)]
        if _POLYGON_ZONES:
            for i, lane in enumerate(_POLYGON_ZONES):
                c   = cols[i % len(cols)]
                pts = np.array([[int(fx * w), int(fy * h)] for fx, fy in lane["points"]], np.int32)
                ov  = frame.copy(); cv2.fillPoly(ov, [pts], c)
                cv2.addWeighted(ov, 0.10, frame, 0.90, 0, frame)
                cv2.polylines(frame, [pts], True, c, 2)
                cnt = sum(self.zone_counts.get(lane["name"], {}).values())
                cx_ = int(pts[:, 0].mean()); cy_ = int(pts[:, 1].mean())
                cv2.putText(frame, f"{lane['name']}:{cnt}", (cx_ - 40, cy_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"{lane['name']}:{cnt}", (cx_ - 40, cy_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)
            return
        for i, (name, (x1, y1, x2, y2)) in enumerate(config.ZONES.items()):
            c = cols[i % len(cols)]
            cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), c, 1)
            cnt = sum(self.zone_counts.get(name, {}).values())
            cv2.putText(frame, f"{name}:{cnt}", (int(x1 * w) + 4, int(y1 * h) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2)

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        fwd = sum(self.dir_counts.get(DIR_FORWARD, {}).values())
        bwd = sum(self.dir_counts.get(DIR_BACKWARD, {}).values())
        lines_text = [(f"Total: {len(self.counted_ids)}", (57, 197, 187))]
        for vt, cnt in self.total_counts.items():
            lines_text.append((f"  {vt[:12]}: {cnt}", (200, 200, 200)))
        lines_text.append((f"FWD:{fwd}  BWD:{bwd}", (180, 220, 100)))
        lines_text.append((f"Live:{self.live_vehicles}  Occ:{self.occupancy_pct:.0f}%  Q:{self.queue_length}",
                           (150, 180, 255)))
        lines_text.append((f"Rate:{self.current_rate}v/hr  Peak:{self.peak_rate}  PHF:{self.phf:.2f}",
                           (255, 200, 80)))
        if self.avg_headway_sec > 0:
            lines_text.append((f"Hdwy:{self.avg_headway_sec:.1f}s  SatFlow:{self.saturation_flow}v/hr",
                               (180, 220, 180)))
        if self.speed_85th > 0:
            lines_text.append((f"V85:{self.speed_85th}km/h  Vmean:{self.speed_mean}km/h",
                               (200, 160, 255)))
        if self.person_count:
            lines_text.append((f"People:{self.person_count}", (255, 100, 100)))
        if self.safety_events:
            lines_text.append((f"Safety:{self.safety_events}", (80, 80, 255)))
        ph = 14 + len(lines_text) * 20 + 8
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (230, ph), (10, 14, 22), -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
        for i, (txt, col) in enumerate(lines_text):
            cv2.putText(frame, txt, (6, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
        cv2.putText(frame, "VELOXIS  |  NextCity Tessera",
                    (w - 220, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (70, 70, 80), 1)
        return frame

    def _log(self, tid, vtype, zone, direction, speed_kmh):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, tid, vtype, zone, direction,
                round(speed_kmh, 1) if speed_kmh else "",
                self.session_label,
                # Session-level metrics at time of crossing
                self.queue_length,
                round(self.occupancy_pct, 1),
                self.current_rate,
                self.avg_headway_sec if self.avg_headway_sec else "",
                self.saturation_flow if self.saturation_flow else "",
                self.phf if self.phf else "",
                self.speed_85th if self.speed_85th else "",
                self.speed_mean if self.speed_mean else "",
            ])
        # Headway calculation (time gap between consecutive crossings)
        now_dt = datetime.datetime.now()
        if vtype in self._last_cross_time:
            gap = (now_dt - self._last_cross_time[vtype]).total_seconds()
            if 0.5 < gap < 120:   # filter noise and long gaps
                self.headway_history.append((vtype, gap))
                if len(self.headway_history) > 200:
                    self.headway_history = self.headway_history[-100:]
                # Average headway and saturation flow
                recent = [g for _, g in self.headway_history[-50:]]
                if recent:
                    self.avg_headway_sec = round(sum(recent) / len(recent), 2)
                    # Saturation flow ≈ 3600 / avg_headway (veh/hr)
                    self.saturation_flow = int(3600 / self.avg_headway_sec)
        self._last_cross_time[vtype] = now_dt

        # Speed percentiles
        if speed_kmh and speed_kmh > 0:
            self.all_speeds.append(speed_kmh)
            if len(self.all_speeds) > 500:
                self.all_speeds = self.all_speeds[-250:]
            sorted_sp = sorted(self.all_speeds)
            n = len(sorted_sp)
            self.speed_85th = round(sorted_sp[int(n * 0.85)], 1) if n >= 5 else 0.0
            self.speed_mean = round(sum(sorted_sp) / n, 1)

    def save_session_summary(self):
        """Save a one-row session summary CSV alongside the main log."""
        try:
            fwd = sum(self.dir_counts.get(DIR_FORWARD, {}).values())
            bwd = sum(self.dir_counts.get(DIR_BACKWARD, {}).values())
            summary_path = self.csv_path.replace(".csv", "_summary.csv")
            with open(summary_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "session", "total_vehicles", "fwd", "bwd",
                    "peak_rate_vph", "phf",
                    "avg_headway_sec", "saturation_flow_vph",
                    "speed_85th_kmh", "speed_mean_kmh",
                    "safety_events", "near_miss_count",
                ] + [f"count_{vt}" for vt in self.total_counts])
                w.writerow([
                    self.session_label,
                    len(self.counted_ids), fwd, bwd,
                    self.peak_rate, self.phf,
                    self.avg_headway_sec, self.saturation_flow,
                    self.speed_85th, self.speed_mean,
                    self.safety_events, len(self.near_miss_log),
                ] + list(self.total_counts.values()))
            print(f"[INFO] Session summary -> {summary_path}")
            return summary_path
        except Exception as e:
            print(f"[WARN] Could not save session summary: {e}")
            return None
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            if df.empty: return None
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            dur = max((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600, 0.017)
            rows = []
            for vt in df["vehicle_type"].dropna().unique():
                sub = df[df["vehicle_type"] == vt]; cnt = len(sub)
                fwd = len(sub[sub["direction"].str.contains("FWD|Forward", na=False)])
                rows.append({"VehicleType": vt, "TotalCount": cnt,
                             "Volume_veh_per_hour": round(cnt / dur, 1),
                             "Forward_count": fwd, "Backward_count": cnt - fwd,
                             "Duration_hours": round(dur, 3),
                             "Session": self.session_label})
            if not output_path:
                output_path = self.csv_path.replace(".csv", "_vissim.csv")
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"[INFO] Vissim export -> {output_path}")
            return output_path
        except Exception as e:
            print(f"[WARN] Vissim export: {e}"); return None

    def print_summary(self):
        fwd = sum(self.dir_counts[DIR_FORWARD].values())
        bwd = sum(self.dir_counts[DIR_BACKWARD].values())
        print("\n" + "=" * 55)
        print(f"  VELOXIS  --  {self.session_label}")
        print(f"  Nishan, SUST CEE 2026  |  NextCity Tessera")
        print("=" * 55)
        print(f"  Total vehicles : {len(self.counted_ids)}")
        for vt, cnt in self.total_counts.items():
            print(f"  {vt:<22} {cnt:>5}  {'#' * min(cnt, 20)}")
        print(f"\n  FWD:{fwd}  BWD:{bwd}  Safety events:{self.safety_events}")
        print(f"\n  -- Intersection Capacity Metrics --")
        print(f"  Peak rate      : {self.peak_rate} veh/hr")
        print(f"  PHF            : {self.phf:.3f}  (ideal 0.85-0.95)")
        print(f"  Avg headway    : {self.avg_headway_sec:.1f} sec")
        print(f"  Saturation flow: {self.saturation_flow} veh/hr")
        print(f"  Speed V85      : {self.speed_85th} km/h")
        print(f"  Speed mean     : {self.speed_mean} km/h")
        print(f"\n  Log -> {self.csv_path}")
        print("=" * 55)
