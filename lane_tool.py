# ============================================================
#  lane_tool.py  —  Interactive Lane / Zone Drawing Tool
#
#  How to use:
#    python lane_tool.py videos/your_video.mp4
#
#  Controls (shown on screen too):
#    Left-click        → place a point (draw polygon for one lane)
#    Right-click       → finish current lane polygon
#    Middle-click      → delete last point
#    'n'               → next lane (give it a name)
#    'c'               → clear all lanes, start over
#    'z'               → undo last lane
#    's'               → save lanes & exit
#    'q'               → quit without saving
#    +/-               → go forward/back in the video to find a good frame
# ============================================================

import cv2
import json
import os
import sys
import numpy as np
import config

# ── Colour palette for lanes ─────────────────────────────────
LANE_COLOURS = [
    (57,  197, 187),   # teal
    (255, 165,  30),   # orange
    (220,  80,  80),   # red
    (130,  80, 220),   # purple
    (80,  200,  80),   # green
    (220, 180,  40),   # yellow
    (200,  80, 160),   # pink
    (80,  160, 220),   # sky blue
]

ZONES_SAVE_PATH = "data/lanes.json"
HELP_LINES = [
    "LEFT CLICK  — add point",
    "RIGHT CLICK — finish lane",
    "N           — new lane",
    "Z           — undo lane",
    "C           — clear all",
    "+/-         — seek video",
    "S           — SAVE & exit",
    "Q           — quit",
]


class LaneTool:
    def __init__(self, source):
        self.source = source
        self.cap    = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open: {source}")
            sys.exit(1)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx    = 0
        self.base_frame   = None   # clean frame (no drawings)
        self.display      = None   # frame with all overlays

        # Finished lanes: list of {"name": str, "points": [(x,y),...]}
        self.lanes   = []
        # Current lane being drawn
        self.current_points = []
        self.current_name   = ""

        self._jump_to(0)
        cv2.namedWindow("Lane Tool", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lane Tool", 1100, 650)
        cv2.setMouseCallback("Lane Tool", self._mouse)

    # ── Video navigation ──────────────────────────────────────
    def _jump_to(self, idx):
        idx = max(0, min(idx, self.total_frames - 1))
        self.frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.base_frame = frame.copy()
        self._redraw()

    # ── Mouse callback ────────────────────────────────────────
    def _mouse(self, event, x, y, flags, param):
        h, w = self.base_frame.shape[:2]
        # Clamp to frame
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) >= 3:
                self._finish_lane()
            else:
                print("[INFO] Need at least 3 points to finish a lane.")

        elif event == cv2.EVENT_MBUTTONDOWN:
            if self.current_points:
                self.current_points.pop()
                self._redraw()

    # ── Finish drawing current lane ───────────────────────────
    def _finish_lane(self):
        if not self.current_name:
            # Ask for name in terminal
            default = f"Lane {len(self.lanes) + 1}"
            name = input(f"\n  Lane name (press Enter for '{default}'): ").strip()
            self.current_name = name if name else default

        self.lanes.append({
            "name":   self.current_name,
            "points": self.current_points.copy(),
        })
        print(f"  ✓ Saved lane: '{self.current_name}'  ({len(self.current_points)} points)")
        self.current_points = []
        self.current_name   = ""
        self._redraw()

    # ── Redraw display frame ──────────────────────────────────
    def _redraw(self):
        frame = self.base_frame.copy()
        h, w  = frame.shape[:2]

        # Draw finished lanes
        for i, lane in enumerate(self.lanes):
            colour = LANE_COLOURS[i % len(LANE_COLOURS)]
            pts    = np.array(lane["points"], dtype=np.int32)
            # Filled translucent polygon
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], colour)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            # Outline
            cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)
            # Label at centroid
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(frame, lane["name"], (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, lane["name"], (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 1, cv2.LINE_AA)

        # Draw current (in-progress) lane
        if self.current_points:
            cur_col = LANE_COLOURS[len(self.lanes) % len(LANE_COLOURS)]
            for pt in self.current_points:
                cv2.circle(frame, pt, 5, cur_col, -1)
                cv2.circle(frame, pt, 6, (255, 255, 255), 1)
            if len(self.current_points) > 1:
                cv2.polylines(frame,
                              [np.array(self.current_points, dtype=np.int32)],
                              isClosed=False, color=cur_col, thickness=2)
            # "Close" preview line
            if len(self.current_points) >= 3:
                cv2.line(frame, self.current_points[-1], self.current_points[0],
                         cur_col, 1, cv2.LINE_AA)

        # Help panel (top-right)
        panel_x = w - 220
        cv2.rectangle(frame, (panel_x - 8, 8), (w - 8, 20 + len(HELP_LINES) * 22),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x - 8, 8), (w - 8, 20 + len(HELP_LINES) * 22),
                      (200, 200, 200), 1)
        for j, line in enumerate(HELP_LINES):
            cv2.putText(frame, line, (panel_x, 28 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # Status bar (bottom)
        status = (f"  Frame {self.frame_idx}/{self.total_frames}  |  "
                  f"Lanes saved: {len(self.lanes)}  |  "
                  f"Points in current: {len(self.current_points)}")
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, status, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        self.display = frame
        cv2.imshow("Lane Tool", frame)

    # ── Main loop ─────────────────────────────────────────────
    def run(self):
        print("\n" + "="*52)
        print("  LANE DRAWING TOOL")
        print("="*52)
        print("  1. Click points around one lane/road arm")
        print("  2. Right-click to finish it → enter a name")
        print("  3. Repeat for each lane/direction")
        print("  4. Press S to save and exit")
        print("="*52 + "\n")

        seek_speed = max(1, self.total_frames // 200)   # ~0.5% per keypress

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("[INFO] Quit without saving.")
                break

            elif key == ord('s'):
                if not self.lanes:
                    print("[WARNING] No lanes drawn yet. Draw at least one lane first.")
                    continue
                self._save()
                break

            elif key == ord('n'):
                # Force-finish current lane with a name prompt
                if len(self.current_points) >= 3:
                    self._finish_lane()
                else:
                    name = input("  Name for next lane: ").strip()
                    self.current_name = name if name else f"Lane {len(self.lanes)+1}"
                    print(f"  Drawing '{self.current_name}' — click points then right-click to finish.")

            elif key == ord('z'):
                if self.lanes:
                    removed = self.lanes.pop()
                    print(f"  Undid lane: '{removed['name']}'")
                    self._redraw()

            elif key == ord('c'):
                self.lanes          = []
                self.current_points = []
                self.current_name   = ""
                print("  Cleared all lanes.")
                self._redraw()

            elif key in (ord('+'), ord('='), 83):   # +, =, right arrow
                self._jump_to(self.frame_idx + seek_speed)

            elif key in (ord('-'), 81):             # -, left arrow
                self._jump_to(self.frame_idx - seek_speed)

            elif key == ord('f'):                   # fast-forward 10%
                self._jump_to(self.frame_idx + self.total_frames // 10)

            elif key == ord('b'):                   # back 10%
                self._jump_to(self.frame_idx - self.total_frames // 10)

        cv2.destroyAllWindows()
        self.cap.release()

    # ── Save lanes to JSON ────────────────────────────────────
    def _save(self):
        os.makedirs("data", exist_ok=True)
        h, w = self.base_frame.shape[:2]

        # Convert pixel coords → fractions (0–1) so it works at any resolution
        output = []
        for lane in self.lanes:
            frac_points = [(round(px / w, 4), round(py / h, 4))
                           for px, py in lane["points"]]
            output.append({"name": lane["name"], "points": frac_points})

        with open(ZONES_SAVE_PATH, "w") as f:
            json.dump({"source": str(self.source),
                       "frame_w": w, "frame_h": h,
                       "lanes": output}, f, indent=2)

        # Also patch config.py automatically
        self._patch_config(output, w, h)

        print(f"\n  ✓ Lanes saved to: {ZONES_SAVE_PATH}")
        print(f"  ✓ config.py updated automatically")
        print(f"\n  Saved {len(output)} lane(s):")
        for lane in output:
            print(f"    → '{lane['name']}'  ({len(lane['points'])} points)")
        print("\n  You can now run file_detect.py or live_detect.py")
        print("  ENABLE_ZONES = True has been set in config.py\n")

    # ── Auto-patch config.py ──────────────────────────────────
    def _patch_config(self, lanes, w, h):
        """Write polygon zones into config.py so detector picks them up."""
        try:
            with open("config.py", "r", encoding="utf-8") as f:
                content = f.read()

            # Build new ZONES block (bounding boxes of each polygon)
            zones_lines = ["ZONES = {\n"]
            for lane in lanes:
                xs = [p[0] for p in lane["points"]]
                ys = [p[1] for p in lane["points"]]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                name = lane["name"].replace('"', "'")
                zones_lines.append(
                    f'    "{name}": ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}),\n'
                )
            zones_lines.append("}\n")
            new_zones = "".join(zones_lines)

            # Replace ENABLE_ZONES line
            import re
            content = re.sub(r"ENABLE_ZONES\s*=\s*\w+", "ENABLE_ZONES = True", content)
            # Replace whole ZONES block
            content = re.sub(r"ZONES\s*=\s*\{[^}]*\}", new_zones.rstrip(), content, flags=re.DOTALL)

            # Also save polygon data as comment for reference
            content += f"\n# Polygon zones (from lane_tool.py)\n"
            content += f"POLYGON_ZONES = {json.dumps({l['name']: l['points'] for l in lanes})}\n"

            with open("config.py", "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"  [WARNING] Could not auto-patch config.py: {e}")
            print("  Manually set ENABLE_ZONES = True in config.py")


# ── Polygon-based zone checker (used by detector.py) ─────────
def point_in_polygon(px, py, polygon_frac_points, frame_w, frame_h):
    """
    Check if pixel (px, py) is inside a polygon defined by
    fractional coordinates. Used by the detector for per-lane counting.
    """
    pts = np.array([[int(fx * frame_w), int(fy * frame_h)]
                    for fx, fy in polygon_frac_points], dtype=np.int32)
    result = cv2.pointPolygonTest(pts, (float(px), float(py)), False)
    return result >= 0


def load_polygon_zones():
    """Load saved polygon zones from data/lanes.json."""
    if not os.path.exists(ZONES_SAVE_PATH):
        return None
    with open(ZONES_SAVE_PATH) as f:
        data = json.load(f)
    return data.get("lanes", [])


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # Allow integer camera index
        if source.isdigit():
            source = int(source)
    else:
        print("\n[LANE TOOL]")
        print("You can use this with a video file OR a live camera.")
        print("Examples:")
        print("  python lane_tool.py videos/dhaka_road.mp4")
        print("  python lane_tool.py 0   (webcam)")
        source = input("\nEnter video path or camera index: ").strip()
        if source.isdigit():
            source = int(source)
        if not source:
            print("[ERROR] No source given.")
            sys.exit(1)

    tool = LaneTool(source)
    tool.run()
