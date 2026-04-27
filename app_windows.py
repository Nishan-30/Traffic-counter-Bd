# ================================================================
#  VELOXIS  —  app_windows.py  (v1.0)
#
#  Author  : Nishan, SUST CEE
#  Product : VELOXIS · an app of NextCity Tessera
#  Year    : 2026
#  License : MIT
#
#  Light/Dark mode: uses CTk native theming only — works instantly.
#  All colors from CTk theme system — no hardcoded hex.
# ================================================================

import tkinter as tk
import customtkinter as ctk
import threading, os, sys, datetime, json, re, glob, time, queue, math
import cv2, numpy as np, subprocess
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd

# ── Prefs ──────────────────────────────────────────────────────
PREFS_FILE = "data/user_prefs.json"

def load_prefs() -> dict:
    try:
        with open(PREFS_FILE, encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_prefs(d: dict):
    os.makedirs("data", exist_ok=True)
    p = load_prefs(); p.update(d)
    with open(PREFS_FILE, "w", encoding="utf-8") as f: json.dump(p, f, indent=2)

# ── Theme setup (CTk native — no hex needed) ──────────────────
_prefs = load_prefs()
_THEME = _prefs.get("theme", "dark")
ctk.set_appearance_mode(_THEME)
ctk.set_default_color_theme("blue")

# Accent colours that don't need to change with theme
# (used only for matplotlib and cv2 overlays)
ACC_BLUE   = "#3b82f6"
ACC_GREEN  = "#34d399"
ACC_AMBER  = "#fbbf24"
ACC_RED    = "#f87171"
ACC_PURPLE = "#a78bfa"
ACC_TEAL   = "#2dd4bf"
LANE_COLS  = ["#2dd4bf","#fbbf24","#f87171","#a78bfa","#34d399","#fb923c"]

# Vehicle icon map
V_ICONS = {
    "car":          ("🚙", ACC_TEAL),
    "rickshaw":     ("🛺", ACC_AMBER),
    "CNG/auto":     ("🛺", "#fb923c"),
    "rickshaw/CNG": ("🛺", ACC_AMBER),
    "motorcycle":   ("🏍", ACC_RED),
    "bus":          ("🚌", ACC_BLUE),
    "truck":        ("🚛", ACC_PURPLE),
    "bicycle":      ("🚲", ACC_GREEN),
    "train":        ("🚆", "#60a5fa"),
}


# ================================================================
#  AI COUNTING LINE DETECTOR
# ================================================================
class AILineDetector:
    def __init__(self, n=35):
        self.n=n; self.frames=[]; self.ready=False
        self.line_start=None; self.line_end=None; self.position=0.55
    def feed(self, frame):
        if self.ready: return True
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.frames.append(cv2.resize(gray,(320,180)))
        if len(self.frames)>=self.n: self._analyse(); return True
        return False
    def _analyse(self):
        vx_list, vy_list = [], []
        h_frame, w_frame = self.frames[0].shape[:2]

        # Only analyse bottom 60% of frame — ignore sky and background
        # Sky has more movement from camera shake, not from vehicles
        road_top = int(h_frame * 0.35)

        for i in range(0, len(self.frames)-1, 2):
            f1 = self.frames[i][road_top:, :]
            f2 = self.frames[i+1][road_top:, :]
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            # Only strong motion — likely vehicles not background drift
            thresh = np.percentile(mag, 80)
            mask = mag > thresh
            if mask.sum() < 50:   # too few moving pixels — skip
                continue
            vx_list.append(float(flow[...,0][mask].mean()))
            vy_list.append(float(flow[...,1][mask].mean()))

        if not vx_list:
            # Fallback: horizontal line at 60% of frame
            self.line_start = (0.0, 0.60)
            self.line_end   = (1.0, 0.60)
            self.ready = True
            return

        vx, vy = np.mean(vx_list), np.mean(vy_list)

        # Flow perpendicular = counting line direction
        perp = np.degrees(np.arctan2(vy, vx)) + 90
        rad  = np.radians(perp)
        dx, dy = np.cos(rad), np.sin(rad)

        # Centre of counting line: 60% down, middle horizontally
        cx = w_frame * 0.5
        cy = h_frame * 0.60   # well into road area, not sky
        t  = max(w_frame, h_frame) * 1.5

        # Clamp to frame bounds (fractions 0-1)
        x1 = max(0.0, min(1.0, (cx - dx*t) / w_frame))
        y1 = max(0.0, min(1.0, (cy - dy*t) / h_frame))
        x2 = max(0.0, min(1.0, (cx + dx*t) / w_frame))
        y2 = max(0.0, min(1.0, (cy + dy*t) / h_frame))

        # Sanity check: if line is mostly in top third, force it down
        mid_y = (y1 + y2) / 2
        if mid_y < 0.4:
            self.line_start = (0.0, 0.60)
            self.line_end   = (1.0, 0.60)
        else:
            self.line_start = (x1, y1)
            self.line_end   = (x2, y2)
        self.ready = True
    def get_line_px(self,w,h):
        if self.line_start:
            return (int(self.line_start[0]*w),int(self.line_start[1]*h)),\
                   (int(self.line_end[0]*w),  int(self.line_end[1]*h))
        ly=int(h*self.position); return (0,ly),(w,ly)
    def progress(self): return len(self.frames)/self.n


# ================================================================
#  DETECTION THREAD
# ================================================================
class DetectionThread(threading.Thread):
    def __init__(self, source, mode, on_status, on_done,
                 on_progress=None, use_ai=True, conf_ref=None,
                 manual_line=None):
        super().__init__(daemon=True)
        self.source=source; self.mode=mode
        self.on_status=on_status; self.on_done=on_done
        self.on_progress=on_progress; self.use_ai=use_ai
        self.conf_ref=conf_ref; self.manual_line=manual_line
        self._stop=threading.Event()
        # Only use AI if no manual line set
        self.ai=AILineDetector(35) if (use_ai and manual_line is None) else None
        self.calibrating=(use_ai and manual_line is None)
        self.frame_q=queue.Queue(maxsize=2)
        self.fps=0.0; self._ft=time.time(); self._ff=0

    def stop(self): self._stop.set()

    def _push(self, frame, summary):
        try: self.frame_q.put_nowait((frame,summary))
        except queue.Full:
            try: self.frame_q.get_nowait()
            except queue.Empty: pass
            try: self.frame_q.put_nowait((frame,summary))
            except queue.Full: pass
        self._ff+=1
        now=time.time()
        if now-self._ft>=1.0:
            self.fps=self._ff/(now-self._ft); self._ff=0; self._ft=now

    def run(self):
        try:
            import config
            from detector import VehicleDetector
        except ImportError as e:
            self.on_status(f"ERROR: {e}"); return
        self.on_status("Loading YOLO model…")
        lbl=f"{'live' if self.mode=='live' else 'file'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        det=VehicleDetector(session_label=lbl)

        # Apply manual line — takes priority over AI
        if self.manual_line is not None:
            det.manual_line_a = max(0.05, min(0.95, self.manual_line))
            self.on_status(f"Manual line at {int(self.manual_line*100)}% — Detecting…")

        cap=(cv2.VideoCapture(self.source,cv2.CAP_DSHOW)
             if isinstance(self.source,int) else cv2.VideoCapture(self.source))
        if not cap.isOpened(): self.on_status("Cannot open source."); return
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # Auto-detect video FPS and update config
        detected_fps = cap.get(cv2.CAP_PROP_FPS)
        if detected_fps and 5 < detected_fps < 120:
            import config as cfg
            cfg.VIDEO_FPS = detected_fps
            self.on_status(f"Video FPS auto-detected: {detected_fps:.1f}")

        fn=0
        while not self._stop.is_set():
            ret,frame=cap.read()
            if not ret:
                if self.mode=="live": time.sleep(0.03); continue
                break
            fn+=1
            if self.conf_ref:
                import config as cfg; cfg.CONFIDENCE=self.conf_ref[0]

            # AI calibration — only if no manual line
            if self.ai and self.calibrating and det.manual_line_a is None:
                done=self.ai.feed(frame)
                if not done:
                    pct=int(self.ai.progress()*100)
                    self.on_status(f"AI analysing traffic flow… {pct}%")
                    h,w=frame.shape[:2]
                    ov=frame.copy()
                    cv2.rectangle(ov,(0,0),(w,h),(10,14,22),cv2.FILLED)
                    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
                    bw=int(w*0.55); bx=(w-bw)//2; by=h//2
                    cv2.rectangle(frame,(bx,by-10),(bx+bw,by+10),(30,35,50),-1)
                    cv2.rectangle(frame,(bx,by-10),(bx+int(bw*pct/100),by+10),(45,180,140),-1)
                    cv2.putText(frame,f"AI calibrating…  {pct}%",
                                (bx,by-18),cv2.FONT_HERSHEY_SIMPLEX,0.62,(190,190,200),1)
                    self._push(frame,{}); continue
                else:
                    self.calibrating=False
                    # Only apply AI line if no manual line was set
                    if self.ai.line_start and det.manual_line_a is None:
                        det.ai_line_start=self.ai.line_start
                        det.ai_line_end=self.ai.line_end
                    self.on_status("Detecting…")
            ann,summary=det.process_frame(frame)
            cv2.putText(ann,f"FPS:{self.fps:.1f}",
                        (ann.shape[1]-88,18),cv2.FONT_HERSHEY_SIMPLEX,0.48,(57,197,187),1)
            self._push(ann,summary)
            if self.on_progress and self.mode=="file":
                self.on_progress(int(fn/total*100))
        cap.release()
        # Save session summary CSV with all metrics
        det.save_session_summary()
        self.on_done({
            "total_unique":  len(det.counted_ids),
            "by_type":       det.total_counts,
            "phf":           det.phf,
            "peak_rate":     det.peak_rate,
            "avg_headway":   det.avg_headway_sec,
            "saturation":    det.saturation_flow,
            "speed_85th":    det.speed_85th,
            "speed_mean":    det.speed_mean,
            "safety_events": det.safety_events,
        })
        self.on_status("Session complete ✓")


# ================================================================
#  REUSABLE WIDGETS  (all use CTk native theming)
# ================================================================

class StatCard(ctk.CTkFrame):
    """Premium stat card — accent bar at bottom, large number."""
    def __init__(self, master, label, value="—", accent=ACC_BLUE, icon="", **kw):
        super().__init__(master, corner_radius=14, **kw)
        self.grid_columnconfigure(0, weight=1)
        top=ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0,column=0,padx=16,pady=(14,0),sticky="ew")
        ctk.CTkLabel(top,text=icon,font=("Segoe UI",15)).pack(side="left",padx=(0,5))
        ctk.CTkLabel(top,text=label.upper(),
                     font=("Segoe UI",9,"bold")).pack(side="left")
        self._val=ctk.CTkLabel(self,text=str(value),
                               font=("Segoe UI",28,"bold"),text_color=accent)
        self._val.grid(row=1,column=0,padx=16,pady=(2,10),sticky="w")
        ctk.CTkFrame(self,fg_color=accent,height=3,corner_radius=0
                    ).grid(row=2,column=0,sticky="ew")
    def set(self,v): self._val.configure(text=str(v))


class VideoCanvas(ctk.CTkLabel):
    """Video display — shows frames. Subclass ClickableVideoCanvas for line setting."""
    def __init__(self, master, placeholder="Press Start to begin", **kw):
        super().__init__(master, text=placeholder, corner_radius=14,
                         font=("Segoe UI",13), **kw)
        self._img = None

    def update_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        ww = max(self.winfo_width(), 640)
        wh = max(self.winfo_height(), 360)
        scale = min(ww/w, wh/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        rgb = cv2.cvtColor(cv2.resize(frame,(nw,nh)), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._img = ctk.CTkImage(light_image=img, dark_image=img, size=(nw,nh))
        self.configure(image=self._img, text="")


class ClickableVideoCanvas(ctk.CTkLabel):
    """
    Video display with click-to-set counting line.
    - Click anywhere on video → sets horizontal counting line at that Y position
    - Drag to move the line
    - Right-click → clear the line (revert to AI/default)
    """
    def __init__(self, master, on_line_set=None,
                 placeholder="Browse a video → then click on frame to set counting line", **kw):
        super().__init__(master, text=placeholder, corner_radius=14,
                         font=("Segoe UI",13), **kw)
        self._img        = None
        self._line_frac  = None
        self._frame_orig = None
        self.on_line_set = on_line_set
        # Stored after each _render so _set_line_at can compute correct fraction
        self._nh = None   # scaled frame height (pixels)
        self._nw = None   # scaled frame width  (pixels)
        self.bind("<Button-1>",   self._on_click)
        self.bind("<B1-Motion>",  self._on_drag)
        self.bind("<Button-3>",   self._on_right)   # right-click = clear
        self.configure(cursor="crosshair")

    def update_frame(self, frame: np.ndarray):
        self._frame_orig = frame.copy()
        self._render(frame)

    def _render(self, frame):
        h, w = frame.shape[:2]
        ww = max(self.winfo_width(), 640)
        wh = max(self.winfo_height(), 360)
        scale = min(ww/w, wh/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        disp = cv2.resize(frame, (nw, nh))

        if self._line_frac is not None:
            ly = int(nh * self._line_frac)
            # Draw line with handles
            cv2.line(disp, (0, ly), (nw, ly), (57,197,187), 2)
            # Left handle circle
            cv2.circle(disp, (20, ly), 8, (57,197,187), -1)
            # Right handle circle
            cv2.circle(disp, (nw-20, ly), 8, (57,197,187), -1)
            # Label
            cv2.rectangle(disp, (4, max(ly-22,0)), (260, max(ly-2,20)), (10,30,40), -1)
            cv2.putText(disp, f"Counting line ({int(self._line_frac*100)}%) — drag to move",
                        (8, max(ly-6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (57,197,187), 1)
        else:
            # Show hint overlay
            ov = disp.copy()
            cv2.rectangle(ov, (0, nh//2-22), (nw, nh//2+22), (10,20,30), -1)
            cv2.addWeighted(ov, 0.6, disp, 0.4, 0, disp)
            cv2.putText(disp, "Click anywhere on video to set counting line",
                        (nw//2-220, nh//2+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (57,197,187), 1)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._img = ctk.CTkImage(light_image=img, dark_image=img, size=(nw,nh))
        self.configure(image=self._img, text="")
        # Store scaled dimensions so _set_line_at can compute correct fraction
        self._nh = nh
        self._nw = nw

    def _on_click(self, e):
        self._set_line_at(e.y)

    def _on_drag(self, e):
        self._set_line_at(e.y)

    def _on_right(self, e):
        """Right-click clears the manual line."""
        self._line_frac = None
        if self._frame_orig is not None:
            self._render(self._frame_orig)
        if self.on_line_set:
            self.on_line_set(None)

    def _set_line_at(self, y_px):
        # Use stored scaled frame height (nh) — NOT winfo_height() (widget height).
        # When the video frame doesn't fill the full widget vertically, dividing by
        # winfo_height causes the drawn line to appear above where the user clicked.
        nh = self._nh if self._nh else self.winfo_height()
        if nh <= 0: return
        # Clamp click to within the frame area, then compute fraction of frame height
        frac = max(0.05, min(0.95, y_px / nh))
        self._line_frac = frac
        if self._frame_orig is not None:
            self._render(self._frame_orig)
        if self.on_line_set:
            self.on_line_set(frac)

    def get_line_frac(self):
        return self._line_frac

    def clear(self):
        self._line_frac = None


class NavBtn(ctk.CTkButton):
    def __init__(self, master, icon, label, cmd, **kw):
        super().__init__(master,text=f"  {icon}   {label}",anchor="w",
                         fg_color="transparent",font=("Segoe UI",13),
                         height=44,corner_radius=10,command=cmd,**kw)
    def set_active(self,v):
        self.configure(
            fg_color=(["#1e3a5f","#dbeafe"][ctk.get_appearance_mode()=="Light"] if v
                      else "transparent"),
            font=("Segoe UI",13,"bold" if v else "normal"))


class SLabel(ctk.CTkLabel):
    def __init__(self,master,text,**kw):
        super().__init__(master,text=text.upper(),
                         font=("Segoe UI",9,"bold"),**kw)


class StatusBar(ctk.CTkFrame):
    def __init__(self,master,**kw):
        super().__init__(master,height=28,corner_radius=0,**kw)
        self.grid_columnconfigure(1,weight=1)
        self._dot=ctk.CTkLabel(self,text="●",font=("Segoe UI",10),width=18)
        self._dot.grid(row=0,column=0,padx=(10,4),sticky="w")
        self._msg=ctk.CTkLabel(self,text="Ready",font=("Segoe UI",11))
        self._msg.grid(row=0,column=1,sticky="w")
        p=load_prefs()
        name=p.get("author_name","Nishan")
        inst=p.get("institution","SUST · CEE")
        self._right=ctk.CTkLabel(self,
            text=f"VELOXIS  ·  {name}, {inst}  ·  NextCity Tessera  ·  © 2026",
            font=("Segoe UI",10))
        self._right.grid(row=0,column=2,padx=12,sticky="e")
    def set(self,msg,state="idle"):
        colours={"idle":None,"running":ACC_GREEN,"warn":ACC_AMBER,"error":ACC_RED}
        c=colours.get(state)
        if c: self._dot.configure(text_color=c)
        else: self._dot.configure(text_color=["#1f2937","#94a3b8"][ctk.get_appearance_mode()=="Light"])
        self._msg.configure(text=msg)


class Page(ctk.CTkFrame):
    def __init__(self,master):
        super().__init__(master,corner_radius=0)
        self.grid_columnconfigure(0,weight=1)
    def page_header(self,icon,title,subtitle):
        hf=ctk.CTkFrame(self,fg_color="transparent")
        hf.grid(row=0,column=0,padx=32,pady=(26,16),sticky="ew")
        hf.grid_columnconfigure(1,weight=1)
        ic=ctk.CTkFrame(hf,width=48,height=48,corner_radius=14)
        ic.grid(row=0,column=0,rowspan=2,padx=(0,14)); ic.grid_propagate(False)
        ctk.CTkLabel(ic,text=icon,font=("Segoe UI",22)
                    ).place(relx=0.5,rely=0.5,anchor="center")
        ctk.CTkLabel(hf,text=title,font=("Segoe UI",20,"bold")
                    ).grid(row=0,column=1,sticky="w")
        ctk.CTkLabel(hf,text=subtitle,font=("Segoe UI",12)
                    ).grid(row=1,column=1,sticky="w")


class DetachedWindow(tk.Toplevel):
    def __init__(self,master):
        super().__init__(master)
        self.title("TrafficCounter BD — Live Feed")
        self.geometry("960x580"); self._img=None; self._frame=None
        theme=ctk.get_appearance_mode()
        bg="#0e1117" if theme=="Dark" else "#f0f4f8"
        self.configure(bg=bg)
        self.protocol("WM_DELETE_WINDOW",self._close); self._closed=False
        bar=tk.Frame(self,bg="#161b27" if theme=="Dark" else "#fff",height=38)
        bar.pack(fill="x")
        tk.Label(bar,text="  📹  Live Camera Feed",bg=bar["bg"],
                 fg="#e8eaf0" if theme=="Dark" else "#0f172a",
                 font=("Segoe UI",11,"bold")).pack(side="left",pady=8)
        self.fps_l=tk.Label(bar,text="FPS: —",bg=bar["bg"],
                             fg=ACC_TEAL,font=("Consolas",10))
        self.fps_l.pack(side="right",padx=12)
        self.cnt_l=tk.Label(bar,text="Total: 0",bg=bar["bg"],
                             fg=ACC_GREEN,font=("Segoe UI",10,"bold"))
        self.cnt_l.pack(side="right",padx=6)
        tk.Button(bar,text="📸 Snapshot",bg="#1e2535" if theme=="Dark" else "#e8edf2",
                  fg=ACC_BLUE,relief="flat",font=("Segoe UI",10),
                  command=self._snap).pack(side="right",padx=6,pady=5)
        self.lbl=tk.Label(self,bg="#0a0d14" if theme=="Dark" else "#f8fafc",
                          text="Waiting…",fg="#64748b",font=("Segoe UI",12))
        self.lbl.pack(fill="both",expand=True)
    def update_frame(self,frame,summary):
        if self._closed: return
        self._frame=frame.copy()
        h,w=frame.shape[:2]
        lw=max(self.lbl.winfo_width(),640); lh=max(self.lbl.winfo_height(),360)
        sc=min(lw/w,lh/h,1.0); nw,nh=int(w*sc),int(h*sc)
        rgb=cv2.cvtColor(cv2.resize(frame,(nw,nh)),cv2.COLOR_BGR2RGB)
        self._img=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.lbl.configure(image=self._img,text="")
        self.cnt_l.configure(text=f"Total: {summary.get('total_unique',0)}")
    def set_fps(self,fps): self.fps_l.configure(text=f"FPS: {fps:.1f}")
    def _snap(self):
        if self._frame is None: return
        os.makedirs("data/snapshots",exist_ok=True)
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"data/snapshots/snap_{ts}.jpg",self._frame)
        self.cnt_l.configure(text="Snapshot saved!")
    def _close(self): self._closed=True; self.destroy()
    @property
    def closed(self): return self._closed


# ================================================================
#  HOME PAGE
# ================================================================
class HomePage(Page):
    def __init__(self,master):
        super().__init__(master)
        self.grid_rowconfigure(4,weight=1)
        self.page_header("🏠","Dashboard","Session overview & live stats")
        # Row 1 — primary counts
        r1=ctk.CTkFrame(self,fg_color="transparent")
        r1.grid(row=1,column=0,padx=32,sticky="ew")
        r1.grid_columnconfigure((0,1,2,3),weight=1)
        self.ct =StatCard(r1,"Total",   "—",ACC_BLUE,  "🚗")
        self.cc =StatCard(r1,"Cars",    "—",ACC_TEAL,  "🚙")
        self.cr =StatCard(r1,"Rickshaws","—",ACC_AMBER,"🛺")
        self.cm =StatCard(r1,"Motorcycles","—",ACC_RED,"🏍")
        for i,c in enumerate([self.ct,self.cc,self.cr,self.cm]):
            c.grid(row=0,column=i,padx=(0 if i==0 else 10,0),sticky="ew")
        # Row 2 — secondary counts
        r2=ctk.CTkFrame(self,fg_color="transparent")
        r2.grid(row=2,column=0,padx=32,pady=(10,0),sticky="ew")
        r2.grid_columnconfigure((0,1,2,3),weight=1)
        self.cbus =StatCard(r2,"Buses",    "—",ACC_BLUE,  "🚌")
        self.ctrk =StatCard(r2,"Trucks",   "—",ACC_PURPLE,"🚛")
        self.cbike=StatCard(r2,"Bicycles", "—",ACC_GREEN, "🚲")
        self.csess=StatCard(r2,"Sessions", "—","#94a3b8",  "📁")
        for i,c in enumerate([self.cbus,self.ctrk,self.cbike,self.csess]):
            c.grid(row=0,column=i,padx=(0 if i==0 else 10,0),sticky="ew")
        SLabel(self,"Session Log").grid(row=3,column=0,padx=32,pady=(18,6),sticky="w")
        self.log=ctk.CTkTextbox(self,font=("Consolas",12),corner_radius=12,
                                 border_width=1)
        self.log.grid(row=4,column=0,padx=32,pady=(0,24),sticky="nsew")
        self._load_stats()

    def _get_df(self):
        files=glob.glob(os.path.join("data","log_*.csv"))
        if not files: return None
        dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
        return pd.concat(dfs,ignore_index=True) if dfs else None

    def _load_stats(self):
        df=self._get_df()
        if df is None:
            self.log.insert("end","No sessions yet. Run detection to start.\n")
            return
        try:
            bt=df["vehicle_type"].value_counts().to_dict() if "vehicle_type" in df.columns else {}
            self.ct.set(len(df)); self.cc.set(bt.get("car",0))
            self.cr.set(bt.get("rickshaw",0) or bt.get("rickshaw/CNG",0))
            self.cm.set(bt.get("motorcycle",0))
            self.cbus.set(bt.get("bus",0)); self.ctrk.set(bt.get("truck",0))
            self.cbike.set(bt.get("bicycle",0))
            nsess = df["session"].nunique() if "session" in df.columns else 0
            self.csess.set(nsess)
            # Load recent session history into log
            self.log.insert("end","── Previous Sessions ──────────────────────\n")
            if "session" in df.columns and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                for sess, grp in df.groupby("session"):
                    t0 = grp["timestamp"].min()
                    date_str = t0.strftime("%Y-%m-%d %H:%M") if pd.notna(t0) else "unknown"
                    vt = grp["vehicle_type"].value_counts().to_dict() if "vehicle_type" in grp.columns else {}
                    top = ", ".join(f"{k}:{v}" for k,v in list(vt.items())[:4])
                    self.log.insert("end",f"[{date_str}]  {sess}  — {len(grp)} vehicles  ({top})\n")
            self.log.see("end")
        except Exception as e:
            self.log.insert("end",f"Could not load history: {e}\n")

    def update_stats(self,s):
        bt=s.get("by_type",{})
        self.ct.set(s.get("total_unique",0)); self.cc.set(bt.get("car",0))
        self.cr.set(bt.get("rickshaw",0) or bt.get("rickshaw/CNG",0))
        self.cm.set(bt.get("motorcycle",0))
        self.cbus.set(bt.get("bus",0)); self.ctrk.set(bt.get("truck",0))
        self.cbike.set(bt.get("bicycle",0))

    def log_msg(self,msg):
        ts=datetime.datetime.now().strftime("%H:%M:%S")
        self.log.insert("end",f"[{ts}]  {msg}\n"); self.log.see("end")


# ================================================================
#  LIVE PAGE
# ================================================================
class LivePage(Page):
    def __init__(self,master,status_bar=None,home_page=None):
        super().__init__(master)
        self.thread=None; self.status_bar=status_bar; self.home_page=home_page
        self._detached=None; self._t0=None; self._last_frame=None
        self._conf_ref=[0.40]
        self.grid_rowconfigure(3,weight=1)
        self.page_header("📹","Live Detection","Real-time · AI line · any camera angle")

        # ── Source card ───────────────────────────────────────
        src=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        src.grid(row=1,column=0,padx=32,pady=(0,10),sticky="ew")
        src.grid_columnconfigure((0,1,2,3),weight=1)
        SLabel(src,"Camera Source").grid(row=0,column=0,columnspan=4,padx=18,pady=(14,10),sticky="w")

        self.src_var=tk.StringVar(value=load_prefs().get("last_source","webcam"))
        opts=[("webcam","💻","Laptop\nWebcam"),("usb","🔌","USB\nCamera"),
              ("droidcam","📱","DroidCam\n(WiFi)"),("custom","🔗","Custom\nURL")]
        self.opt_btns={}
        _inactive_border = "#2d3748"
        for col,(key,icon,label) in enumerate(opts):
            is_active = self.src_var.get()==key
            f=ctk.CTkFrame(src,corner_radius=8,border_width=2,
                            border_color=(ACC_BLUE if is_active else _inactive_border))
            f.grid(row=1,column=col,padx=(10 if col==0 else 4,10 if col==3 else 4),
                   pady=(0,8),sticky="ew")
            ctk.CTkRadioButton(f,text="",variable=self.src_var,value=key,
                               fg_color=ACC_BLUE,width=16,
                               command=self._src_ch).place(relx=0.88,rely=0.1)
            # Compact: icon small, label one line
            ctk.CTkLabel(f,text=icon,font=("Segoe UI",16)).pack(pady=(8,1))
            ctk.CTkLabel(f,text=label.replace("\n"," "),
                         font=("Segoe UI",9),justify="center").pack(pady=(0,8))
            self.opt_btns[key]=f

        # Detail panes
        self.det=ctk.CTkFrame(src,fg_color="transparent")
        self.det.grid(row=2,column=0,columnspan=4,padx=16,pady=(0,8),sticky="ew")

        self.wp=ctk.CTkFrame(self.det,fg_color="transparent")
        ctk.CTkLabel(self.wp,text="Index:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.cam_idx=ctk.CTkComboBox(self.wp,values=["0","1","2","3"],width=72)
        self.cam_idx.set("0"); self.cam_idx.pack(side="left",padx=(0,10))
        self.scan_btn=ctk.CTkButton(self.wp,text="🔍  Auto-scan cameras",width=160,height=30,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),command=self._scan)
        self.scan_btn.pack(side="left",padx=(0,8))
        self.scan_lbl=ctk.CTkLabel(self.wp,text="",font=("Segoe UI",11))
        self.scan_lbl.pack(side="left")

        self.dp2=ctk.CTkFrame(self.det,fg_color="transparent")
        ctk.CTkLabel(self.dp2,text="Phone IP:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.ip_var=tk.StringVar(value=load_prefs().get("droidcam_ip",""))
        self.ip_e=ctk.CTkEntry(self.dp2,textvariable=self.ip_var,
            placeholder_text="192.168.1.5",width=165)
        self.ip_e.pack(side="left",padx=(0,8))
        ctk.CTkLabel(self.dp2,text="Port:",font=("Segoe UI",12)).pack(side="left",padx=(0,4))
        self.port_var=tk.StringVar(value=load_prefs().get("droidcam_port","4747"))
        ctk.CTkEntry(self.dp2,textvariable=self.port_var,width=62
                    ).pack(side="left",padx=(0,10))
        self.droid_test=ctk.CTkButton(self.dp2,text="🔗  Test",width=90,height=30,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),command=self._test_droid)
        self.droid_test.pack(side="left",padx=(0,8))
        self.droid_lbl=ctk.CTkLabel(self.dp2,text="",font=("Segoe UI",11))
        self.droid_lbl.pack(side="left")

        self.up=ctk.CTkFrame(self.det,fg_color="transparent")
        ctk.CTkLabel(self.up,text="URL:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.url_var=tk.StringVar()
        ctk.CTkEntry(self.up,textvariable=self.url_var,
            placeholder_text="rtsp://...  or  http://...",width=420).pack(side="left")

        # Options
        opt=ctk.CTkFrame(src,fg_color="transparent")
        opt.grid(row=3,column=0,columnspan=4,padx=18,pady=(0,14),sticky="ew")
        self.ai_var=tk.BooleanVar(value=True)
        ctk.CTkSwitch(opt,text="",variable=self.ai_var,
                      button_color=ACC_TEAL,progress_color=ACC_TEAL,
                      width=44,height=22).pack(side="left",padx=(0,8))
        ctk.CTkLabel(opt,text="AI auto-detect counting line",
                     font=("Segoe UI",12,"bold"),text_color=ACC_TEAL).pack(side="left",padx=(0,24))
        ctk.CTkLabel(opt,text="Confidence:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.conf_sl=ctk.CTkSlider(opt,from_=0.1,to=0.9,width=130,
            button_color=ACC_BLUE,progress_color=ACC_BLUE,
            command=lambda v:[self._conf_ref.__setitem__(0,float(v)),
                               self.conf_lbl.configure(text=f"{int(float(v)*100)}%"),
                               self.conf_hint.configure(text=self._conf_hint(float(v)))])
        self.conf_sl.set(0.40); self.conf_sl.pack(side="left",padx=(0,4))
        self.conf_lbl=ctk.CTkLabel(opt,text="40%",font=("Segoe UI",11,"bold"),
                                    width=36,text_color=ACC_BLUE)
        self.conf_lbl.pack(side="left")
        self.conf_hint=ctk.CTkLabel(opt,text="· crowded road",
                                     font=("Segoe UI",9),text_color="#64748b")
        self.conf_hint.pack(side="left",padx=(4,0))
        self._src_ch()

        # ── Buttons ───────────────────────────────────────────
        br=ctk.CTkFrame(self,fg_color="transparent")
        br.grid(row=2,column=0,padx=32,pady=(0,10),sticky="w")
        self.start_btn=ctk.CTkButton(br,text="▶  Start Detection",
            width=180,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            command=self._start)
        self.start_btn.pack(side="left",padx=(0,10))
        self.stop_btn=ctk.CTkButton(br,text="■  Stop",
            width=110,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            fg_color="#7f1d1d",hover_color="#991b1b",state="disabled",command=self._stop)
        self.stop_btn.pack(side="left",padx=(0,10))
        self.detach_btn=ctk.CTkButton(br,text="⧉  Pop-out",
            width=120,height=44,font=("Segoe UI",13),corner_radius=10,
            fg_color="transparent",border_width=1,command=self._detach)
        self.detach_btn.pack(side="left",padx=(0,10))
        self.snap_btn=ctk.CTkButton(br,text="📸  Snapshot",
            width=130,height=44,font=("Segoe UI",13),corner_radius=10,
            fg_color="transparent",border_width=1,command=self._snap)
        self.snap_btn.pack(side="left",padx=(0,14))
        self.timer_lbl=ctk.CTkLabel(br,text="00:00:00",font=("Consolas",13))
        self.timer_lbl.pack(side="left")

        # ── Compact stats strip (one row, small) ─────────────
        self.grid_rowconfigure(3,weight=0)
        self.grid_rowconfigure(4,weight=1)

        strip=ctk.CTkFrame(self,fg_color="transparent",height=52)
        strip.grid(row=3,column=0,padx=32,pady=(0,4),sticky="ew")
        strip.grid_propagate(False)
        strip.grid_columnconfigure(list(range(12)),weight=1)

        self.lv_cards={}
        items=[
            ("total","Total","🚗",ACC_BLUE),
            ("car","Cars","🚙",ACC_TEAL),
            ("cng","CNG","🛺","#fb923c"),
            ("rickshaw","Rick.","🛺",ACC_AMBER),
            ("motorcycle","Moto","🏍",ACC_RED),
            ("bus","Bus","🚌","#60a5fa"),
            ("truck","Truck","🚛",ACC_PURPLE),
            ("bicycle","Bike","🚲",ACC_GREEN),
            ("_live","Live","📍","#60a5fa"),
            ("_occ","Occ%","📊","#f97316"),
            ("_queue","Queue","🚦","#818cf8"),
            ("_rate","Rate/hr","📈","#fbbf24"),
            ("_person","People","🚶","#a78bfa"),
            ("_safety","Safety","⚠️","#f87171"),
        ]
        for col,(key,lbl,icon,acc) in enumerate(items):
            f=ctk.CTkFrame(strip,corner_radius=8,border_width=1,height=48)
            f.grid(row=0,column=col,padx=(0 if col==0 else 2,0),sticky="ew")
            f.grid_propagate(False); f.grid_columnconfigure(0,weight=1)
            ctk.CTkLabel(f,text=f"{icon} {lbl}",font=("Segoe UI",8),
                         text_color="#64748b").grid(row=0,column=0,pady=(4,0))
            val_lbl=ctk.CTkLabel(f,text="0",font=("Segoe UI",13,"bold"),
                                  text_color=acc)
            val_lbl.grid(row=1,column=0,pady=(0,4))
            ctk.CTkFrame(f,fg_color=acc,height=2,corner_radius=0
                        ).grid(row=2,column=0,sticky="ew")
            self.lv_cards[key]=val_lbl
        strip.grid_columnconfigure(list(range(14)),weight=1)

        self._live_manual_line = None
        self.video=ClickableVideoCanvas(self,
            on_line_set=self._live_line_set,
            placeholder="Camera feed will appear here · Click to set counting line")
        self.video.grid(row=4,column=0,padx=32,pady=(0,24),sticky="nsew")

    def _src_ch(self):
        v=self.src_var.get()
        _inactive = "#2d3748"
        for k,f in self.opt_btns.items():
            f.configure(border_color=(ACC_BLUE if k==v else _inactive))
        for p in [self.wp,self.dp2,self.up]: p.pack_forget()
        if v in ("webcam","usb"):
            self.wp.pack(fill="x")
            self.cam_idx.set("1" if v=="usb" else "0")
        elif v=="droidcam": self.dp2.pack(fill="x")
        elif v=="custom": self.up.pack(fill="x")

    def _scan(self):
        self.scan_lbl.configure(text="Scanning…",text_color=ACC_AMBER); self.update()
        found=[]
        for i in range(4):
            cap=cv2.VideoCapture(i,cv2.CAP_DSHOW)
            if cap.isOpened():
                ret,_=cap.read()
                if ret: found.append(str(i))
                cap.release()
        if found:
            self.scan_lbl.configure(text=f"✓ Found: {', '.join(found)}",text_color=ACC_GREEN)
            self.cam_idx.set(found[0])
        else: self.scan_lbl.configure(text="✗ None found",text_color=ACC_RED)

    def _test_droid(self):
        ip=self.ip_var.get().strip() or "192.168.1.5"
        port=self.port_var.get().strip() or "4747"
        self.droid_lbl.configure(text="Testing…",text_color=ACC_AMBER); self.update()
        for suf in ["/video","/mjpegfeed","/videofeed"]:
            url=f"http://{ip}:{port}{suf}"
            cap=cv2.VideoCapture(url)
            if cap.isOpened():
                ret,_=cap.read(); cap.release()
                if ret:
                    self.droid_lbl.configure(text="✓ Connected!",text_color=ACC_GREEN)
                    self._droid_url=url
                    save_prefs({"droidcam_ip":ip,"droidcam_port":port}); return
        self.droid_lbl.configure(text="✗ Cannot connect — check IP & WiFi",text_color=ACC_RED)

    def _get_src(self):
        v=self.src_var.get()
        if v in ("webcam","usb"): return int(self.cam_idx.get() or "0")
        if v=="droidcam":
            if hasattr(self,"_droid_url"): return self._droid_url
            ip=self.ip_var.get().strip() or "192.168.1.5"
            port=self.port_var.get().strip() or "4747"
            return f"http://{ip}:{port}/video"
        return self.url_var.get().strip()

    def _start(self):
        src=self._get_src()
        if not src and src!=0: return
        save_prefs({"last_source":self.src_var.get()})
        self._t0=time.time(); self._conf_ref[0]=self.conf_sl.get()
        if self.status_bar: self.status_bar.set("Detection running…","running")
        # Use manual line if user clicked on video
        manual = getattr(self,'_live_manual_line',None)
        self.thread=DetectionThread(src,"live",
            on_status=lambda m: self.after(0,lambda mm=m: self._sts(mm)),
            on_done  =lambda s: self.after(0,lambda ss=s: self._done(ss)),
            use_ai=self.ai_var.get() and manual is None,
            conf_ref=self._conf_ref,
            manual_line=manual)
        self.thread.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._poll(); self._tick()

    def _poll(self):
        if self.thread and self.thread.is_alive():
            try:
                frame,summary=self.thread.frame_q.get_nowait()
                self._last_frame=frame; self.video.update_frame(frame)
                if self._detached and not self._detached.closed:
                    self._detached.update_frame(frame,summary)
                    self._detached.set_fps(self.thread.fps)
                if summary:
                    bt=summary.get("by_type",{})
                    total=summary.get("total_unique",0)
                    self.lv_cards["total"].configure(text=str(total))
                    for k in ["car","cng","rickshaw","motorcycle","bus","truck","bicycle"]:
                        self.lv_cards[k].configure(text=str(bt.get(k,0)))
                    self.lv_cards["_live"].configure(text=str(summary.get("live_vehicles",0)))
                    self.lv_cards["_occ"].configure(text=f"{summary.get('occupancy_pct',0):.0f}%")
                    self.lv_cards["_queue"].configure(text=str(summary.get("queue_length",0)))
                    self.lv_cards["_rate"].configure(text=str(summary.get("current_rate",0)))
                    self.lv_cards["_person"].configure(text=str(summary.get("person_count",0)))
                    self.lv_cards["_safety"].configure(text=str(summary.get("safety_events",0)))
                    if total and self.home_page:
                        self.home_page.update_stats(summary)
            except queue.Empty: pass
            self.after(33,self._poll)

    def _tick(self):
        if self.thread and self.thread.is_alive() and self._t0:
            e=int(time.time()-self._t0)
            self.timer_lbl.configure(text=f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")
            self.after(1000,self._tick)

    def _sts(self,m):
        if self.status_bar:
            self.status_bar.set(m,"error" if "ERROR" in m or "Cannot" in m else "running")

    def _done(self,s):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if self.status_bar: self.status_bar.set("Session complete ✓","idle")
        if self.home_page:
            self.home_page.update_stats(s)
            self.home_page.log_msg(f"Live session — {s.get('total_unique',0)} vehicles")

    def _stop(self):
        if self.thread: self.thread.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if self.status_bar: self.status_bar.set("Stopped","idle")

    def _detach(self):
        if self._detached and not self._detached.closed:
            self._detached.destroy(); self._detached=None
            self.detach_btn.configure(text="⧉  Pop-out")
        else:
            self._detached=DetachedWindow(self)
            self.detach_btn.configure(text="✕  Close pop-out")

    @staticmethod
    def _conf_hint(v):
        if v < 0.30: return "night/dark"
        if v < 0.40: return "crowded road"
        if v < 0.55: return "daylight clear"
        if v < 0.70: return "strict"
        return "very strict"

    def _live_line_set(self, frac):
        self._live_manual_line = frac
        if frac is not None:
            self.ai_var.set(False)
            if self.status_bar:
                self.status_bar.set(
                    f"Line set at {int(frac*100)}% — right-click to clear — click Start","idle")
        else:
            if self.status_bar:
                self.status_bar.set("Line cleared","idle")

    def _snap(self):
        if self._last_frame is None: return
        os.makedirs("data/snapshots",exist_ok=True)
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p=f"data/snapshots/snap_{ts}.jpg"
        cv2.imwrite(p,self._last_frame)
        if self.status_bar: self.status_bar.set(f"Snapshot saved: {p}","idle")
# ================================================================
class FilePage(Page):
    def __init__(self,master,status_bar=None,home_page=None):
        super().__init__(master); self.thread=None; self._last_frame=None
        self.status_bar=status_bar; self.home_page=home_page
        self._cap_preview=None; self._total_frames=1
        self.grid_rowconfigure(4,weight=1)
        self.page_header("🎬","File Detection","Analyse a recorded road video")

        pick=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        pick.grid(row=1,column=0,padx=32,pady=(0,10),sticky="ew")
        pick.grid_columnconfigure(0,weight=1)
        SLabel(pick,"Video File").grid(row=0,column=0,columnspan=3,padx=18,pady=(14,8),sticky="w")
        pr=ctk.CTkFrame(pick,fg_color="transparent")
        pr.grid(row=1,column=0,columnspan=3,padx=16,pady=(0,8),sticky="ew")
        pr.grid_columnconfigure(0,weight=1)
        self.pv=tk.StringVar()
        ctk.CTkEntry(pr,textvariable=self.pv,placeholder_text="No video selected…"
                    ).grid(row=0,column=0,sticky="ew",padx=(0,10))
        ctk.CTkButton(pr,text="📂  Browse",width=120,height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._browse).grid(row=0,column=1)

        # ── Video seek slider ──────────────────────────────────
        seek_row=ctk.CTkFrame(pick,fg_color="transparent")
        seek_row.grid(row=2,column=0,columnspan=3,padx=16,pady=(0,8),sticky="ew")
        seek_row.grid_columnconfigure(1,weight=1)
        ctk.CTkLabel(seek_row,text="Preview frame:",font=("Segoe UI",11)).grid(row=0,column=0,padx=(0,8))
        self.seek_var=tk.IntVar(value=0)
        self.seek_slider=ctk.CTkSlider(seek_row,from_=0,to=100,
            variable=self.seek_var,command=self._seek_preview,
            button_color=ACC_TEAL,progress_color=ACC_TEAL,state="disabled")
        self.seek_slider.grid(row=0,column=1,sticky="ew",padx=(0,8))
        self.seek_lbl=ctk.CTkLabel(seek_row,text="0 / 0",font=("Segoe UI",11),width=80)
        self.seek_lbl.grid(row=0,column=2)

        ai_r=ctk.CTkFrame(pick,fg_color="transparent")
        ai_r.grid(row=3,column=0,columnspan=3,padx=16,pady=(0,14),sticky="w")
        self.ai_var=tk.BooleanVar(value=False)   # default OFF — user clicks on video
        ctk.CTkSwitch(ai_r,text="",variable=self.ai_var,
                      button_color=ACC_TEAL,progress_color=ACC_TEAL,
                      width=44,height=22).pack(side="left",padx=(0,8))
        ctk.CTkLabel(ai_r,text="AI auto-detect line",
                     font=("Segoe UI",12,"bold"),text_color=ACC_TEAL).pack(side="left",padx=(0,12))
        ctk.CTkLabel(ai_r,
                     text="← OFF: Click on the video frame below to draw your counting line  |  Right-click to clear",
                     font=("Segoe UI",11),text_color="#64748b").pack(side="left")

        br=ctk.CTkFrame(self,fg_color="transparent")
        br.grid(row=2,column=0,padx=32,pady=(0,8),sticky="w")
        self.run_btn=ctk.CTkButton(br,text="▶  Analyse Video",
            width=180,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            state="disabled",command=self._run)
        self.run_btn.pack(side="left",padx=(0,10))
        self.stop_btn=ctk.CTkButton(br,text="■  Stop",
            width=110,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            fg_color="#7f1d1d",hover_color="#991b1b",state="disabled",command=self._stop)
        self.stop_btn.pack(side="left",padx=(0,10))
        ctk.CTkButton(br,text="📸  Snapshot",width=130,height=44,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._snap).pack(side="left",padx=(0,10))
        ctk.CTkButton(br,text="📊  Export CSV",width=130,height=44,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._export).pack(side="left",padx=(0,6))
        ctk.CTkButton(br,text="🏙  Vissim Export",width=140,height=44,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            text_color=ACC_TEAL,command=self._export_vissim).pack(side="left",padx=(0,10))
        self.prog_lbl=ctk.CTkLabel(br,text="",font=("Segoe UI",12))
        self.prog_lbl.pack(side="left")

        self.prog=ctk.CTkProgressBar(self,height=6,corner_radius=3,
                                      progress_color=ACC_BLUE)
        self.prog.set(0); self.prog.grid(row=3,column=0,padx=32,pady=(0,6),sticky="ew")

        mid=ctk.CTkFrame(self,fg_color="transparent")
        mid.grid(row=4,column=0,padx=32,pady=(0,24),sticky="nsew")
        mid.grid_columnconfigure(0,weight=3); mid.grid_columnconfigure(1,weight=1)
        mid.grid_rowconfigure(0,weight=1)
        self.video=ClickableVideoCanvas(mid, on_line_set=self._on_line_set)
        self.video.grid(row=0,column=0,padx=(0,12),sticky="nsew")
        self._manual_line_frac = None

        res=ctk.CTkScrollableFrame(mid,corner_radius=14,border_width=1,width=180)
        res.grid(row=0,column=1,sticky="nsew")
        SLabel(res,"Results").pack(anchor="w",padx=14,pady=(14,6))
        self.rcards={}
        for key,label,icon,acc in [
            ("total","Total","🚗",ACC_BLUE),("car","Cars","🚙",ACC_TEAL),
            ("rickshaw","Rickshaws","🛺",ACC_AMBER),
            ("CNG/auto","CNGs","🛺","#fb923c"),
            ("motorcycle","Motorcycles","🏍",ACC_RED),
            ("bus","Buses","🚌","#60a5fa"),("truck","Trucks","🚛",ACC_PURPLE),
            ("bicycle","Bicycles","🚲",ACC_GREEN)]:
            c=StatCard(res,label,"—",acc,icon)
            c.pack(fill="x",padx=10,pady=4)
            self.rcards[key]=c
        ctk.CTkButton(res,text="📊 Export",height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),
            command=self._export).pack(fill="x",padx=10,pady=(4,14))

    def _seek_preview(self, val):
        """Show frame at slider position without running detection."""
        if self._cap_preview is None: return
        idx = int(float(val))
        self._cap_preview.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap_preview.read()
        if ret:
            self.video.update_frame(frame)
            self.seek_lbl.configure(text=f"{idx} / {self._total_frames}")

    def _browse(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(title="Open Video",initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv *.wmv"),("All","*.*")])
        if not p: return
        self.pv.set(p)
        self.run_btn.configure(state="normal")
        # Load video for seek preview
        if self._cap_preview: self._cap_preview.release()
        self._cap_preview = cv2.VideoCapture(p)
        self._total_frames = max(int(self._cap_preview.get(cv2.CAP_PROP_FRAME_COUNT))-1, 1)
        self.seek_slider.configure(to=self._total_frames, state="normal")
        self.seek_var.set(0)
        # Show first frame as preview
        self._cap_preview.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap_preview.read()
        if ret:
            self.video.update_frame(frame)
            self.seek_lbl.configure(text=f"0 / {self._total_frames}")

    def _on_line_set(self, frac):
        self._manual_line_frac = frac
        if frac is not None:
            # Auto-disable AI when user manually sets line
            self.ai_var.set(False)
            if self.status_bar:
                self.status_bar.set(
                    f"Manual line set at {int(frac*100)}% — right-click to clear — press Analyse Video",
                    "idle")
        else:
            # Line cleared — re-enable AI option
            if self.status_bar:
                self.status_bar.set("Line cleared — AI will auto-detect, or click again to set manually", "idle")

    def _run(self):
        p=self.pv.get().strip()
        if not p: return
        self.prog.set(0)
        if self.status_bar: self.status_bar.set("Analysing…","running")
        manual_line = getattr(self, '_manual_line_frac', None)
        self.thread=DetectionThread(p,"file",
            on_status=lambda m: self.after(0,lambda mm=m:
                [self.prog_lbl.configure(text=mm),
                 self.status_bar and self.status_bar.set(mm,"running")]),
            on_done  =lambda s: self.after(0,lambda ss=s: self._done(ss)),
            on_progress=lambda v: self.after(0,lambda vv=v:
                [self.prog.set(vv/100),self.prog_lbl.configure(text=f"{vv}%")]),
            use_ai=self.ai_var.get() and manual_line is None,
            manual_line=manual_line)
        self.thread.start()
        self.run_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self._poll()

    def _poll(self):
        if self.thread and self.thread.is_alive():
            try:
                frame,_=self.thread.frame_q.get_nowait()
                self._last_frame=frame; self.video.update_frame(frame)
            except queue.Empty: pass
            self.after(33,self._poll)

    def _stop(self):
        if self.thread: self.thread.stop()
        self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")

    def _done(self,s):
        self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
        self.prog.set(1)
        bt=s.get("by_type",{})
        self.rcards["total"].set(s.get("total_unique",0))
        for k in ["car","rickshaw","CNG/auto","motorcycle","bus","truck","bicycle"]:
            self.rcards[k].set(bt.get(k,0))
        if self.status_bar: self.status_bar.set("Analysis complete ✓","idle")
        if self.home_page:
            self.home_page.update_stats(s)
            self.home_page.log_msg(f"File analysis — {s.get('total_unique',0)} vehicles")

    def _snap(self):
        if self._last_frame is None: return
        os.makedirs("data/snapshots",exist_ok=True)
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"data/snapshots/snap_{ts}.jpg",self._last_frame)
        if self.status_bar: self.status_bar.set("Snapshot saved","idle")

    def _export(self):
        from tkinter import filedialog, messagebox
        path=filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")],initialfile="traffic_report.csv")
        if not path: return
        try:
            files=glob.glob(os.path.join("data","log_*.csv"))
            dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
            if not dfs: messagebox.showinfo("No data","Run detection first."); return
            pd.concat(dfs,ignore_index=True).to_csv(path,index=False)
            messagebox.showinfo("Exported ✓",f"Saved to:\n{path}")
        except Exception as e: messagebox.showerror("Error",str(e))

    def _export_vissim(self):
        """Export session data in PTV Vissim / Aimsun compatible format."""
        from tkinter import filedialog, messagebox
        try:
            files=glob.glob(os.path.join("data","log_*.csv"))
            dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
            if not dfs: messagebox.showinfo("No data","Run detection first."); return
            df=pd.concat(dfs,ignore_index=True)
            if "timestamp" in df.columns:
                df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
            duration_hrs=max(
                (df["timestamp"].max()-df["timestamp"].min()).total_seconds()/3600
                if "timestamp" in df.columns else 1, 0.0167)
            rows=[]
            for vtype in df["vehicle_type"].dropna().unique():
                sub=df[df["vehicle_type"]==vtype]
                count=len(sub)
                vol_hr=round(count/duration_hrs,1)
                fwd=len(sub[sub["direction"].str.contains("FWD|Forward|→",na=False,regex=True)]) \
                    if "direction" in sub.columns else 0
                bwd=count-fwd
                rows.append({"VehicleType":vtype,"TotalCount":count,
                             "Volume_veh_per_hour":vol_hr,
                             "Forward_count":fwd,"Backward_count":bwd,
                             "Duration_hours":round(duration_hrs,3)})
            out=pd.DataFrame(rows)
            path=filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV","*.csv")],initialfile="vissim_input.csv")
            if not path: return
            out.to_csv(path,index=False)
            messagebox.showinfo("Vissim Export ✓",
                f"Saved to:\n{path}\n\n"
                "Import into PTV Vissim:\n"
                "  Traffic Demand → Vehicle Inputs → Import CSV\n\n"
                "Import into Aimsun:\n"
                "  Traffic State → Origin/Destination → Import")
        except Exception as e: messagebox.showerror("Error",str(e))


# ================================================================
#  CALIBRATE SPEED PAGE  (Homography calibration)
# ================================================================
class CalibratePage(Page):
    """
    4-point homography calibration for accurate speed estimation.
    User loads a video frame, clicks 4 road points, enters real distances.
    Saves homography.npy for detector to use automatically.
    """
    def __init__(self,master,status_bar=None):
        super().__init__(master)
        self.status_bar=status_bar
        self._cap=None; self._frame=None; self._pts=[]; self._photo=None
        self.grid_rowconfigure(3,weight=1)
        self.page_header("📐","Speed Calibration",
            "Click 4 road points → enter real distances → accurate speed for any camera angle")

        # Instructions
        inst=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        inst.grid(row=1,column=0,padx=32,pady=(0,8),sticky="ew")
        ctk.CTkLabel(inst,justify="left",font=("Segoe UI",12),
            text=(
                "HOW TO CALIBRATE:\n"
                "1. Load a video frame where road markings are visible\n"
                "2. Click 4 corners of a known rectangle on the road "
                "(e.g. lane markings, road edge)\n"
                "3. Enter the real-world width and length of that rectangle in metres\n"
                "4. Click Calibrate — homography.npy saved, speed becomes accurate"
            )).pack(padx=18,pady=12,anchor="w")

        # Controls
        cr=ctk.CTkFrame(self,fg_color="transparent")
        cr.grid(row=2,column=0,padx=32,pady=(0,8),sticky="ew")
        ctk.CTkButton(cr,text="📂  Load Video Frame",width=170,height=38,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._load).pack(side="left",padx=(0,10))
        ctk.CTkLabel(cr,text="Real width (m):",font=("Segoe UI",12)).pack(side="left",padx=(10,4))
        self.w_var=tk.StringVar(value="3.5")
        ctk.CTkEntry(cr,textvariable=self.w_var,width=65).pack(side="left",padx=(0,10))
        ctk.CTkLabel(cr,text="Real length (m):",font=("Segoe UI",12)).pack(side="left",padx=(0,4))
        self.l_var=tk.StringVar(value="8.0")
        ctk.CTkEntry(cr,textvariable=self.l_var,width=65).pack(side="left",padx=(0,10))
        ctk.CTkButton(cr,text="✕  Clear points",width=120,height=38,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),
            command=self._clear).pack(side="left",padx=(0,8))
        self.cal_btn=ctk.CTkButton(cr,text="📐  Calibrate",width=120,height=38,
            fg_color=ACC_TEAL,hover_color="#0d9488",
            font=("Segoe UI",13,"bold"),state="disabled",
            command=self._calibrate)
        self.cal_btn.pack(side="left",padx=(0,8))
        self.status_lbl=ctk.CTkLabel(cr,text="Load a video, then click 4 corners on the road.",
                                      font=("Segoe UI",11),text_color="#64748b")
        self.status_lbl.pack(side="left",padx=(6,0))

        # Canvas
        cf=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        cf.grid(row=3,column=0,padx=32,pady=(0,24),sticky="nsew")
        self.canvas=tk.Canvas(cf,bg="#0a0d14",highlightthickness=0,cursor="crosshair")
        self.canvas.pack(fill="both",expand=True,padx=2,pady=2)
        self.canvas.bind("<Button-1>",self._click)

    def _load(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not p: return
        cap=cv2.VideoCapture(p)
        # Get middle frame for best road visibility
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES,total//2)
        ret,frame=cap.read(); cap.release()
        if ret:
            self._frame=frame.copy(); self._pts=[]; self._redraw()
            self.status_lbl.configure(text="Click 4 corners of a known rectangle on the road.")

    def _click(self,e):
        if self._frame is None or len(self._pts)>=4: return
        # Convert canvas coords back to frame coords
        h,w=self._frame.shape[:2]
        cw=max(self.canvas.winfo_width(),640); ch=max(self.canvas.winfo_height(),360)
        sc=min(cw/w,ch/h,1.0); nw,nh=int(w*sc),int(h*sc)
        ox=(cw-nw)//2; oy=(ch-nh)//2
        fx=max(0,min(w-1,int((e.x-ox)/sc)))
        fy=max(0,min(h-1,int((e.y-oy)/sc)))
        self._pts.append((fx,fy))
        self._redraw()
        n=len(self._pts)
        if n<4:
            self.status_lbl.configure(text=f"Point {n}/4 set. Click next corner.")
        else:
            self.status_lbl.configure(text="4 points set! Enter dimensions and click Calibrate.")
            self.cal_btn.configure(state="normal")

    def _redraw(self):
        self.canvas.delete("all")
        if self._frame is None: return
        h,w=self._frame.shape[:2]
        cw=max(self.canvas.winfo_width(),640); ch=max(self.canvas.winfo_height(),360)
        sc=min(cw/w,ch/h,1.0); nw,nh=int(w*sc),int(h*sc)
        ox=(cw-nw)//2; oy=(ch-nh)//2
        self._ox=ox;self._oy=oy;self._sc=sc
        disp=cv2.resize(self._frame,(nw,nh))
        from PIL import Image,ImageTk
        rgb=cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
        self._photo=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(ox,oy,anchor="nw",image=self._photo)
        cols=["#2dd4bf","#fbbf24","#f87171","#a78bfa"]
        labels=["TL","TR","BR","BL"]
        for i,(fx,fy) in enumerate(self._pts):
            px=ox+int(fx*sc); py=oy+int(fy*sc)
            c=cols[i]
            self.canvas.create_oval(px-8,py-8,px+8,py+8,fill=c,outline="white",width=2)
            self.canvas.create_text(px+12,py,text=labels[i],fill=c,
                                    font=("Segoe UI",11,"bold"))
        if len(self._pts)==4:
            pts=[(ox+int(fx*sc),oy+int(fy*sc)) for fx,fy in self._pts]
            flat=[c for xy in pts+[pts[0]] for c in xy]
            self.canvas.create_line(flat,fill="#2dd4bf",width=2,dash=(6,3))

    def _clear(self):
        self._pts=[]; self.cal_btn.configure(state="disabled")
        self.status_lbl.configure(text="Points cleared. Click 4 corners.")
        self._redraw()

    def _calibrate(self):
        if len(self._pts)!=4:
            self.status_lbl.configure(text="Need exactly 4 points!",text_color=ACC_RED)
            return
        try:
            W=float(self.w_var.get()); L=float(self.l_var.get())
        except ValueError:
            self.status_lbl.configure(text="Invalid dimensions.",text_color=ACC_RED)
            return
        # Image points
        img_pts=np.float32(self._pts)
        # World points: TL, TR, BR, BL of rectangle W×L metres
        wld_pts=np.float32([[0,0],[W,0],[W,L],[0,L]])
        H,mask=cv2.findHomography(img_pts,wld_pts,cv2.RANSAC,5.0)
        if H is None:
            self.status_lbl.configure(text="Calibration failed — try different points.",
                                       text_color=ACC_RED)
            return
        os.makedirs("data",exist_ok=True)
        np.save("data/homography.npy",H)
        inliers=int(mask.sum()) if mask is not None else 4
        self.status_lbl.configure(
            text=f"✓ Calibrated! ({inliers}/4 inliers) — homography.npy saved. "
                 f"Restart detection for accurate speed.",
            text_color=ACC_GREEN)
        self.cal_btn.configure(state="disabled")
        if self.status_bar:
            self.status_bar.set("Homography calibrated ✓ — speed will be accurate","idle")


# ================================================================
#  ANALYTICS DASHBOARD
# ================================================================
class DashboardPage(Page):
    def __init__(self,master):
        super().__init__(master); self.grid_rowconfigure(2,weight=1)
        self._ever_shown=False
        self.page_header("📊","Analytics Dashboard","Embedded charts · zoom · pan · export")

        fbar=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        fbar.grid(row=1,column=0,padx=32,pady=(0,10),sticky="ew")

        # Row 1: chart type buttons
        r1=ctk.CTkFrame(fbar,fg_color="transparent"); r1.pack(padx=16,pady=(10,4),fill="x")
        self.cvar=tk.StringVar(value="Daily")
        for lbl in ["Daily","Hourly","Monthly","Types","Direction","By Zone"]:
            ctk.CTkRadioButton(r1,text=lbl,variable=self.cvar,value=lbl,
                               font=("Segoe UI",12),command=self._render_chart
            ).pack(side="left",padx=(0,14))

        # Row 2: filters
        r2=ctk.CTkFrame(fbar,fg_color="transparent"); r2.pack(padx=16,pady=(0,10),fill="x")

        ctk.CTkLabel(r2,text="From:",font=("Segoe UI",11)).pack(side="left",padx=(0,4))
        self.sv=tk.StringVar()
        sv_entry=ctk.CTkEntry(r2,textvariable=self.sv,placeholder_text="YYYY-MM-DD",width=112)
        sv_entry.pack(side="left",padx=(0,4))
        ctk.CTkButton(r2,text="📅",width=28,height=28,font=("Segoe UI",12),
            fg_color="transparent",border_width=1,
            command=lambda: self._pick_date(self.sv)).pack(side="left",padx=(0,10))

        ctk.CTkLabel(r2,text="To:",font=("Segoe UI",11)).pack(side="left",padx=(0,4))
        self.ev=tk.StringVar()
        ev_entry=ctk.CTkEntry(r2,textvariable=self.ev,placeholder_text="YYYY-MM-DD",width=112)
        ev_entry.pack(side="left",padx=(0,4))
        ctk.CTkButton(r2,text="📅",width=28,height=28,font=("Segoe UI",12),
            fg_color="transparent",border_width=1,
            command=lambda: self._pick_date(self.ev)).pack(side="left",padx=(0,14))

        ctk.CTkLabel(r2,text="Session:",font=("Segoe UI",11)).pack(side="left",padx=(0,4))
        self.sess_var=tk.StringVar(value="All")
        self.sess_cb=ctk.CTkComboBox(r2,variable=self.sess_var,values=["All"],
            width=160,command=lambda _: self._render_chart())
        self.sess_cb.pack(side="left",padx=(0,10))

        ctk.CTkButton(r2,text="Clear filters",width=90,height=28,font=("Segoe UI",11),
            fg_color="transparent",border_width=1,
            command=self._clear_filters).pack(side="left",padx=(0,8))
        ctk.CTkButton(r2,text="Refresh",width=80,height=28,font=("Segoe UI",12),
            command=self._render_chart).pack(side="left")

        cf=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        cf.grid(row=2,column=0,padx=32,pady=(0,24),sticky="nsew")
        # Use grid inside cf so the canvas always fills remaining space
        cf.grid_rowconfigure(1,weight=1)
        cf.grid_columnconfigure(0,weight=1)
        tb_f=ctk.CTkFrame(cf,corner_radius=0,height=32)
        tb_f.grid(row=0,column=0,sticky="ew",padx=2,pady=(2,0))
        tb_f.grid_propagate(False)

        dark=ctk.get_appearance_mode()=="Dark"
        bg=("#0e1117" if dark else "#ffffff")
        self.fig=Figure(facecolor=bg)
        self.fig.set_tight_layout(False)
        self.fig.subplots_adjust(left=0.08,right=0.97,top=0.88,bottom=0.18)
        self.ax=self.fig.add_subplot(111)
        self._style_ax()
        self.canvas=FigureCanvasTkAgg(self.fig,master=cf)
        self.canvas.get_tk_widget().grid(row=1,column=0,sticky="nsew",padx=4,pady=4)
        self.toolbar=NavigationToolbar2Tk(self.canvas,tb_f)
        self.toolbar.update()

    def _pick_date(self, var):
        """Simple date picker popup."""
        import datetime as dt
        top = tk.Toplevel(self); top.title("Pick date"); top.geometry("260x200")
        top.configure(bg="#1e2535"); top.resizable(False,False)
        today = dt.date.today()
        tk.Label(top,text="Enter date (YYYY-MM-DD):",bg="#1e2535",fg="#e8eaf0",
                 font=("Segoe UI",11)).pack(pady=(18,6))
        e=tk.Entry(top,font=("Segoe UI",13),width=16,justify="center")
        e.insert(0,str(today)); e.pack(pady=4)
        def quick(days):
            d=today-dt.timedelta(days=days)
            e.delete(0,"end"); e.insert(0,str(d))
        bf=tk.Frame(top,bg="#1e2535"); bf.pack(pady=6)
        for label,days in [("Today",0),("7d",7),("30d",30)]:
            tk.Button(bf,text=label,bg="#2d3748",fg="#94a3b8",relief="flat",
                      command=lambda d=days: quick(d)).pack(side="left",padx=4)
        def ok():
            var.set(e.get().strip()); top.destroy(); self._render_chart()
        tk.Button(top,text="OK",bg="#3b82f6",fg="white",font=("Segoe UI",12),
                  relief="flat",padx=20,command=ok).pack(pady=8)

    def _clear_filters(self):
        self.sv.set(""); self.ev.set("")
        self.sess_var.set("All"); self._render_chart()

    def _update_sessions(self, df):
        """Populate session dropdown from loaded data."""
        if df is None or df.empty or "session" not in df.columns:
            self.sess_cb.configure(values=["All"]); return
        sessions = ["All"] + sorted(df["session"].dropna().unique().tolist())
        self.sess_cb.configure(values=sessions)

    def _style_ax(self):
        dark=ctk.get_appearance_mode()=="Dark"
        bg=("#0e1117" if dark else "#ffffff")
        fg=("#e8eaf0" if dark else "#0f172a")
        grid=("#1f2937" if dark else "#e2e8f0")
        self.ax.set_facecolor(bg)
        self.ax.tick_params(colors="#64748b",labelsize=10)
        self.ax.title.set_color(fg)
        for sp in self.ax.spines.values(): sp.set_color(grid)
        self.ax.grid(True,color=grid,linewidth=0.5,linestyle="--",alpha=0.7)

    def _df(self):
        files=glob.glob(os.path.join("data","log_*.csv"))
        if not files: return pd.DataFrame()
        dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
        if not dfs: return pd.DataFrame()
        df=pd.concat(dfs,ignore_index=True)
        if "timestamp" in df.columns:
            df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
            df["date"]=df["timestamp"].dt.date
            df["hour"]=df["timestamp"].dt.hour
            df["month"]=df["timestamp"].dt.to_period("M").astype(str)
        # Update session dropdown with all available sessions
        self._update_sessions(df)
        # Apply date filters
        s=self.sv.get().strip(); e=self.ev.get().strip()
        if s and "date" in df.columns:
            try: df=df[df["date"]>=datetime.date.fromisoformat(s)]
            except: pass
        if e and "date" in df.columns:
            try: df=df[df["date"]<=datetime.date.fromisoformat(e)]
            except: pass
        # Apply session filter
        sess=getattr(self,'sess_var',None)
        if sess:
            sv=sess.get()
            if sv and sv!="All" and "session" in df.columns:
                df=df[df["session"]==sv]
        return df

    def _render_chart(self):
        df=self._df()
        self.ax.cla()
        # Re-apply BOTH after cla() — matplotlib resets these on clear
        self.fig.set_tight_layout(False)
        self.fig.subplots_adjust(left=0.08,right=0.97,top=0.88,bottom=0.18)
        self._style_ax()
        dark=ctk.get_appearance_mode()=="Dark"
        fg=("#e8eaf0" if dark else "#0f172a")
        mut="#64748b"
        chart=self.cvar.get()
        if df.empty:
            self.ax.text(0.5,0.5,"No data yet. Run detection first.",
                ha="center",va="center",color=mut,fontsize=14,transform=self.ax.transAxes)
        elif chart=="Daily" and "date" in df.columns:
            c=df.groupby("date").size()
            bars=self.ax.bar(range(len(c)),c.values,color=ACC_BLUE,alpha=0.85,width=0.7)
            self.ax.set_xticks(range(len(c)))
            self.ax.set_xticklabels([str(d) for d in c.index],rotation=30,ha="right",fontsize=9)
            self.ax.set_title("Daily Vehicle Count",color=fg,fontsize=13,pad=10)
            self.ax.set_ylabel("Vehicles",color=mut)
            for b in bars:
                h=b.get_height()
                if h>0: self.ax.text(b.get_x()+b.get_width()/2,h+.3,str(int(h)),
                                     ha="center",va="bottom",color=mut,fontsize=8)
        elif chart=="Hourly" and "hour" in df.columns:
            c=df.groupby("hour").size().reindex(range(24),fill_value=0)
            self.ax.bar(c.index,c.values,color=ACC_AMBER,alpha=0.85,width=0.8)
            self.ax.set_xticks(range(0,24,2))
            self.ax.set_xticklabels([f"{h:02d}:00" for h in range(0,24,2)],rotation=30,ha="right",fontsize=9)
            self.ax.set_title("Traffic by Hour of Day",color=fg,fontsize=13,pad=10)
            self.ax.set_ylabel("Vehicles",color=mut)
        elif chart=="Monthly" and "month" in df.columns:
            c=df.groupby("month").size()
            self.ax.bar(range(len(c)),c.values,color=ACC_GREEN,alpha=0.85,width=0.7)
            self.ax.set_xticks(range(len(c)))
            self.ax.set_xticklabels(c.index.astype(str),rotation=20,ha="right",fontsize=10)
            self.ax.set_title("Monthly Trend",color=fg,fontsize=13,pad=10)
            self.ax.set_ylabel("Vehicles",color=mut)
        elif chart=="Types" and "vehicle_type" in df.columns:
            c=df["vehicle_type"].value_counts()
            cols=[ACC_BLUE,ACC_AMBER,ACC_RED,ACC_GREEN,ACC_PURPLE,ACC_TEAL,"#fb923c"][:len(c)]
            _,_,auts=self.ax.pie(c.values,labels=c.index,autopct="%1.0f%%",colors=cols,
                startangle=140,wedgeprops={"linewidth":0.5,"edgecolor":"#0e1117"},
                textprops={"color":fg,"fontsize":11})
            for at in auts: at.set_color("#0e1117"); at.set_fontsize(10)
            self.ax.set_title("Vehicle Type Distribution",color=fg,fontsize=13,pad=10)
        elif chart=="Direction":
            dark = ctk.get_appearance_mode()=="Dark"
            chart_bg = "#0e1117" if dark else "#ffffff"
            if "direction" not in df.columns or df.empty:
                self.ax.text(0.5,0.5,
                    "No direction data yet.\nRun detection first.",
                    ha="center",va="center",color=mut,fontsize=13,
                    transform=self.ax.transAxes)
            else:
                fwd = df[df["direction"].str.contains("FWD|Forward|→",na=False,regex=True)]
                bwd = df[df["direction"].str.contains("BWD|Backward|←",na=False,regex=True)]
                # Check if session filter is active
                sess = getattr(self,'sess_var',None)
                sess_val = sess.get() if sess else "All"

                if sess_val != "All" and "session" in df.columns:
                    # Session selected — show by vehicle type for that session
                    types = sorted(df["vehicle_type"].dropna().unique().tolist())
                    if not types:
                        self.ax.text(0.5,0.5,"No vehicle data.",ha="center",va="center",
                            color=mut,fontsize=13,transform=self.ax.transAxes)
                    else:
                        fwd_c=[len(fwd[fwd["vehicle_type"]==t]) for t in types]
                        bwd_c=[len(bwd[bwd["vehicle_type"]==t]) for t in types]
                        x=list(range(len(types))); w2=0.36
                        b1=self.ax.bar([i-w2/2 for i in x],fwd_c,w2,
                                       label="FWD",color=ACC_BLUE,alpha=0.85)
                        b2=self.ax.bar([i+w2/2 for i in x],bwd_c,w2,
                                       label="BWD",color=ACC_AMBER,alpha=0.85)
                        self.ax.set_xticks(x)
                        self.ax.set_xticklabels(types,rotation=20,ha="right",fontsize=10)
                        self.ax.set_title(f"Forward vs Backward — {sess_val}",
                                          color=fg,fontsize=13,pad=10)
                        self.ax.set_ylabel("Vehicles",color=mut)
                        self.ax.legend(facecolor=chart_bg,labelcolor=fg,fontsize=10)
                        for bar in list(b1)+list(b2):
                            h=bar.get_height()
                            if h>0: self.ax.text(bar.get_x()+bar.get_width()/2,h+0.2,
                                str(int(h)),ha="center",va="bottom",color=mut,fontsize=8)
                else:
                    # All sessions — show session-wise FWD/BWD totals
                    if "session" in df.columns:
                        sessions = sorted(df["session"].dropna().unique().tolist())
                        if len(sessions) > 1:
                            fwd_by_sess = [len(fwd[fwd["session"]==s]) for s in sessions]
                            bwd_by_sess = [len(bwd[bwd["session"]==s]) for s in sessions]
                            x=list(range(len(sessions))); w2=0.38
                            b1=self.ax.bar([i-w2/2 for i in x],fwd_by_sess,w2,
                                           label="FWD",color=ACC_BLUE,alpha=0.85)
                            b2=self.ax.bar([i+w2/2 for i in x],bwd_by_sess,w2,
                                           label="BWD",color=ACC_AMBER,alpha=0.85)
                            short=[s[:12]+"…" if len(s)>14 else s for s in sessions]
                            self.ax.set_xticks(x)
                            self.ax.set_xticklabels(short,rotation=30,ha="right",fontsize=9)
                            self.ax.set_title("Forward vs Backward — All Sessions",
                                              color=fg,fontsize=13,pad=10)
                            self.ax.set_ylabel("Vehicles",color=mut)
                            self.ax.legend(facecolor=chart_bg,labelcolor=fg,fontsize=10)
                            for bar in list(b1)+list(b2):
                                h=bar.get_height()
                                if h>0: self.ax.text(bar.get_x()+bar.get_width()/2,h+0.2,
                                    str(int(h)),ha="center",va="bottom",color=mut,fontsize=8)
                        else:
                            # Only one session — show by vehicle type
                            types=sorted(df["vehicle_type"].dropna().unique().tolist())
                            fwd_c=[len(fwd[fwd["vehicle_type"]==t]) for t in types]
                            bwd_c=[len(bwd[bwd["vehicle_type"]==t]) for t in types]
                            x=list(range(len(types))); w2=0.36
                            b1=self.ax.bar([i-w2/2 for i in x],fwd_c,w2,
                                           label="FWD",color=ACC_BLUE,alpha=0.85)
                            b2=self.ax.bar([i+w2/2 for i in x],bwd_c,w2,
                                           label="BWD",color=ACC_AMBER,alpha=0.85)
                            self.ax.set_xticks(x)
                            self.ax.set_xticklabels(types,rotation=20,ha="right",fontsize=10)
                            self.ax.set_title("Forward vs Backward by Vehicle Type",
                                              color=fg,fontsize=13,pad=10)
                            self.ax.set_ylabel("Vehicles",color=mut)
                            self.ax.legend(facecolor=chart_bg,labelcolor=fg,fontsize=10)
        elif chart=="By Zone" and "zone" in df.columns:
            c=df.groupby("zone").size().sort_values(ascending=True)
            if len(c)>1 and "all" not in c.index:
                self.ax.barh(c.index,c.values,
                             color=LANE_COLS[:len(c)],alpha=0.85)
                self.ax.set_title("By Road / Zone",color=fg,fontsize=13,pad=10)
                self.ax.set_xlabel("Vehicles",color=mut)
            else:
                self.ax.text(0.5,0.5,"Draw lanes first\n(Lane Drawing page).",
                    ha="center",va="center",color=mut,fontsize=13,transform=self.ax.transAxes)
        self.canvas.draw_idle()

    def refresh(self):
        if not self._ever_shown: self._ever_shown=True; self._render_chart()


# ================================================================
#  LANE PAGE
# ================================================================
class LanePage(Page):
    COLS=LANE_COLS
    def __init__(self,master):
        super().__init__(master); self.cap=None; self.lanes=[]; self.cur_pts=[]; self.fphoto=None
        self.grid_rowconfigure(3,weight=1)
        self.page_header("🗺","Lane Drawing","Click to define road zones — any shape, any angle")

        r1=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        r1.grid(row=1,column=0,padx=32,pady=(0,8),sticky="ew")
        rr=ctk.CTkFrame(r1,fg_color="transparent"); rr.pack(padx=14,pady=12,fill="x")
        ctk.CTkButton(rr,text="📂  Load Video",width=140,height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._load).pack(side="left",padx=(0,14))
        ctk.CTkLabel(rr,text="Seek:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.svar=tk.IntVar(value=0)
        self.slider=ctk.CTkSlider(rr,from_=0,to=100,variable=self.svar,
            command=self._seek,width=280,state="disabled")
        self.slider.pack(side="left",padx=(0,10))
        self.flbl=ctk.CTkLabel(rr,text="Frame 0",font=("Segoe UI",11),width=75)
        self.flbl.pack(side="left")

        ctk.CTkFrame(self,corner_radius=10,border_width=1,height=30
                    ).grid(row=2,column=0,padx=32,pady=(0,6),sticky="ew")
        ctk.CTkLabel(self,text="  LEFT CLICK = add point   ·   RIGHT CLICK = remove last point   ·   Name it → Finish Lane",
            font=("Segoe UI",11)).grid(row=2,column=0,padx=48,pady=0)

        cvf=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        cvf.grid(row=3,column=0,padx=32,pady=(0,8),sticky="nsew")
        self.canvas=tk.Canvas(cvf,bg="#0a0d14",highlightthickness=0,cursor="crosshair")
        self.canvas.pack(fill="both",expand=True,padx=2,pady=2)
        self.canvas.bind("<Button-1>",self._click); self.canvas.bind("<Button-3>",self._rclick)

        r2=ctk.CTkFrame(self,fg_color="transparent"); r2.grid(row=4,column=0,padx=32,pady=(0,24),sticky="ew")
        self.ne=ctk.CTkEntry(r2,placeholder_text="Lane name (e.g. North Road)",width=240)
        self.ne.pack(side="left",padx=(0,10))
        for txt,fn,kw in [
            ("✓  Finish Lane",self._finish,{"fg_color":ACC_BLUE,"hover_color":"#2563eb","font":("Segoe UI",13,"bold")}),
            ("↩  Undo",self._undo,{"fg_color":"transparent","border_width":1}),
            ("✕  Clear",self._clr,{"fg_color":"transparent","border_width":1}),
        ]:
            ctk.CTkButton(r2,text=txt,width=120,height=38,command=fn,**kw
                         ).pack(side="left",padx=(0,8))
        ctk.CTkButton(r2,text="💾  Save All Lanes",width=160,height=38,
            fg_color="#064e3b",hover_color="#047857",border_width=1,
            border_color="#047857",text_color=ACC_GREEN,
            font=("Segoe UI",13,"bold"),command=self._save).pack(side="left")
        self.ll=ctk.CTkLabel(r2,text="0 lanes",font=("Segoe UI",12))
        self.ll.pack(side="left",padx=14)

    def _load(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not p: return
        self.cap=cv2.VideoCapture(p)
        tf=max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))-1,1)
        self.slider.configure(to=tf,state="normal"); self._seek(0)
    def _seek(self,v):
        if not self.cap: return
        idx=int(float(v)); self.cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ret,frame=self.cap.read()
        if ret: self.cur_frame=frame.copy(); self.flbl.configure(text=f"Frame {idx}"); self._redraw()
    def _redraw(self):
        self.canvas.delete("all")
        if not hasattr(self,"cur_frame"): return
        h,w=self.cur_frame.shape[:2]; cw=max(self.canvas.winfo_width(),640); ch=max(self.canvas.winfo_height(),360)
        sc=min(cw/w,ch/h,1.0); nw,nh=int(w*sc),int(h*sc); ox,oy=(cw-nw)//2,(ch-nh)//2
        self._ox=ox; self._oy=oy; self._nw=nw; self._nh=nh
        rgb=cv2.cvtColor(cv2.resize(self.cur_frame,(nw,nh)),cv2.COLOR_BGR2RGB)
        self.fphoto=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(ox,oy,anchor="nw",image=self.fphoto)
        for i,lane in enumerate(self.lanes):
            col=self.COLS[i%len(self.COLS)]; pts=[(ox+fx*nw,oy+fy*nh) for fx,fy in lane["points"]]
            flat=[c for xy in pts for c in xy]
            if len(flat)>=4: self.canvas.create_polygon(flat,fill=col,stipple="gray25",outline=col,width=2)
            self.canvas.create_text(sum(p[0] for p in pts)/len(pts)+1,sum(p[1] for p in pts)/len(pts)+1,
                text=lane["name"],fill="#000",font=("Segoe UI",11,"bold"))
            self.canvas.create_text(sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts),
                text=lane["name"],fill="white",font=("Segoe UI",11,"bold"))
        if self.cur_pts:
            col=self.COLS[len(self.lanes)%len(self.COLS)]
            flat=[c for xy in self.cur_pts for c in xy]
            for px,py in self.cur_pts: self.canvas.create_oval(px-6,py-6,px+6,py+6,fill=col,outline="white")
            if len(self.cur_pts)>1: self.canvas.create_line(flat,fill=col,width=2,dash=(6,3))
            if len(self.cur_pts)>=3:
                self.canvas.create_line(self.cur_pts[-1][0],self.cur_pts[-1][1],
                    self.cur_pts[0][0],self.cur_pts[0][1],fill=col,width=1,dash=(4,4))

        # Draw auto counting line preview (midpoint of all lanes)
        if self.lanes:
            all_ys=[]
            for lane in self.lanes:
                all_ys.extend([oy+fy*nh for fx,fy in lane["points"]])
            if all_ys:
                road_top=min(all_ys); road_bot=max(all_ys)
                mid_y = (road_top+road_bot)/2
                line_a = road_top+(road_bot-road_top)*0.33
                line_b = road_top+(road_bot-road_top)*0.67
                x0=ox; x1=ox+nw
                # Single line (teal)
                self.canvas.create_line(x0,mid_y,x1,mid_y,fill="#2dd4bf",width=2,dash=(8,4))
                self.canvas.create_text(x0+6,mid_y-10,text="Counting line",
                    fill="#2dd4bf",font=("Segoe UI",9),anchor="w")
                # Dual lines (amber)
                self.canvas.create_line(x0,line_a,x1,line_a,fill="#fbbf24",width=1,dash=(6,4))
                self.canvas.create_line(x0,line_b,x1,line_b,fill="#fbbf24",width=1,dash=(6,4))
                self.canvas.create_text(x0+6,line_a-10,text="Line A (Fwd)",
                    fill="#fbbf24",font=("Segoe UI",8),anchor="w")
                self.canvas.create_text(x0+6,line_b-10,text="Line B (Bwd)",
                    fill="#fbbf24",font=("Segoe UI",8),anchor="w")
    def _click(self,e): self.cur_pts.append((e.x,e.y)); self._redraw()
    def _rclick(self,e):
        if self.cur_pts: self.cur_pts.pop(); self._redraw()
    def _finish(self):
        if len(self.cur_pts)<3:
            from tkinter import messagebox; messagebox.showwarning("Need points","Place ≥ 3 points."); return
        name=self.ne.get().strip() or f"Lane {len(self.lanes)+1}"
        nw=self._nw or 640; nh=self._nh or 360; ox=self._ox or 0; oy=self._oy or 0
        frac=[((px-ox)/max(nw,1),(py-oy)/max(nh,1)) for px,py in self.cur_pts]
        self.lanes.append({"name":name,"points":frac}); self.cur_pts=[]; self.ne.delete(0,"end")
        self.ll.configure(text=f"{len(self.lanes)} lane(s)"); self._redraw()
    def _undo(self):
        if self.lanes: self.lanes.pop(); self.ll.configure(text=f"{len(self.lanes)} lane(s)"); self._redraw()
    def _clr(self): self.cur_pts=[]; self._redraw()
    def _save(self):
        if not self.lanes:
            from tkinter import messagebox
            messagebox.showwarning("No lanes","Draw at least one."); return
        os.makedirs("data", exist_ok=True)
        with open("data/lanes.json","w",encoding="utf-8") as f:
            json.dump({"lanes":self.lanes}, f, indent=2)
        try:
            try:
                with open("config.py",encoding="utf-8") as f: c=f.read()
            except UnicodeDecodeError:
                with open("config.py",encoding="latin-1") as f: c=f.read()
            # Build ZONES from lane polygons
            zl=["ZONES = {\n"]
            for lane in self.lanes:
                xs=[p[0] for p in lane["points"]]
                ys=[p[1] for p in lane["points"]]
                zl.append(f'    "{lane["name"]}": ({min(xs):.3f},{min(ys):.3f},{max(xs):.3f},{max(ys):.3f}),\n')
            zl.append("}\n")
            c=re.sub(r"ENABLE_ZONES\s*=\s*\w+","ENABLE_ZONES = True",c)
            c=re.sub(r"ZONES\s*=\s*\{[^}]*\}","".join(zl).rstrip(),c,flags=re.DOTALL)

            # Auto-set counting line to road mid-point based on drawn polygon
            # Uses the vertical midpoint of ALL drawn lanes combined
            all_ys = []
            for lane in self.lanes:
                all_ys.extend([p[1] for p in lane["points"]])
            if all_ys:
                road_top    = min(all_ys)
                road_bottom = max(all_ys)
                # Single line: midpoint of road
                mid_y = round((road_top + road_bottom) / 2, 3)
                # Dual lines: 33% and 67% within road height
                line_a = round(road_top + (road_bottom - road_top) * 0.33, 3)
                line_b = round(road_top + (road_bottom - road_top) * 0.67, 3)
                # Update config
                c=re.sub(r"COUNTING_LINE_POSITION\s*=\s*[\d.]+",
                          f"COUNTING_LINE_POSITION = {mid_y}", c)
                c=re.sub(r"LINE_POS_A\s*=\s*[\d.]+",
                          f"LINE_POS_A = {line_a}", c)
                c=re.sub(r"LINE_POS_B\s*=\s*[\d.]+",
                          f"LINE_POS_B = {line_b}", c)

            with open("config.py","w",encoding="utf-8") as f: f.write(c)
        except Exception as e:
            pass

        from tkinter import messagebox
        messagebox.showinfo(
            "Saved ✓",
            f"{len(self.lanes)} lane(s) saved.\n"
            "Counting line auto-set to road midpoint.\n"
            "config.py updated."
        )


# ================================================================
#  SETTINGS PAGE
# ================================================================
class SettingsPage(Page):
    def __init__(self,master):
        super().__init__(master); self.grid_rowconfigure(1,weight=1)
        self.page_header("⚙️","Settings","Detection parameters & profile")
        scroll=ctk.CTkScrollableFrame(self,corner_radius=0)
        scroll.grid(row=1,column=0,padx=32,pady=(0,24),sticky="nsew")
        scroll.grid_columnconfigure(0,weight=1)

        def sec(t):
            f=ctk.CTkFrame(scroll,corner_radius=14,border_width=1)
            f.pack(fill="x",pady=(0,14)); SLabel(f,t).pack(anchor="w",padx=18,pady=(14,10)); return f
        def row(p,label,w,hint=""):
            r=ctk.CTkFrame(p,fg_color="transparent"); r.pack(fill="x",padx=16,pady=(0,10))
            ctk.CTkLabel(r,text=label,font=("Segoe UI",13),width=270,anchor="w").pack(side="left")
            w.pack(side="left",padx=(0,8))
            if hint: ctk.CTkLabel(r,text=hint,font=("Segoe UI",11)).pack(side="left")

        # Profile
        s0=sec("Profile — Your Identity")
        p=load_prefs()
        self.name_e=ctk.CTkEntry(s0,width=200); self.name_e.insert(0,p.get("author_name","Nishan"))
        row(s0,"Your name",self.name_e,"Shown in sidebar, watermark & reports")
        self.inst_e=ctk.CTkEntry(s0,width=280); self.inst_e.insert(0,p.get("institution","SUST · CEE Dept."))
        row(s0,"Institution",self.inst_e)

        # Model
        s1=sec("Detection Model")
        self.model_cb=ctk.CTkComboBox(s1,width=320,
            values=["bd_vehicles_yolo11.pt  (custom BD · YOLOv11)",
                    "bd_vehicles_best.pt  (custom BD · YOLOv8)",
                    "yolo11n.pt  (fastest — no custom model)",
                    "yolo11s.pt  (balanced — no custom model)",
                    "yolov8n.pt  (YOLOv8 fastest)",
                    "yolov8s.pt  (YOLOv8 balanced)"])
        row(s1,"YOLO Model",self.model_cb,"bd_vehicles_yolo11.pt = your trained custom model")
        self.conf=ctk.CTkSlider(s1,from_=0.1,to=0.9,width=260)
        self.conf.set(0.40); row(s1,"Confidence (0.1–0.9)",self.conf,"Lower = detect more, Higher = fewer false positives")

        # Speed
        s2=sec("Speed Estimation")
        self.ppm=ctk.CTkEntry(s2,width=120); self.ppm.insert(0,"55")
        row(s2,"Pixels per metre",self.ppm,"Measure a known distance in your video frame")
        self.fps_e=ctk.CTkEntry(s2,width=120); self.fps_e.insert(0,"25")
        row(s2,"Video FPS",self.fps_e)

        # Display
        s3=sec("Display Options")
        self.sw_sp=ctk.CTkSwitch(s3,text="Show speed (km/h)",onvalue=True,offvalue=False); self.sw_sp.select(); row(s3,"",self.sw_sp)
        self.sw_id=ctk.CTkSwitch(s3,text="Show track IDs",onvalue=True,offvalue=False); self.sw_id.select(); row(s3,"",self.sw_id)
        self.sw_zo=ctk.CTkSwitch(s3,text="Enable lane/zone counting",onvalue=True,offvalue=False); row(s3,"",self.sw_zo)

        # Performance
        s_perf=sec("Performance Mode")
        # Detect GPU now and show status
        try:
            import torch
            _gpu_ok = torch.cuda.is_available()
            _gpu_name = torch.cuda.get_device_name(0) if _gpu_ok else "Not detected"
        except Exception:
            _gpu_ok = False
            _gpu_name = "PyTorch not loaded yet"
        gpu_status = f"GPU: {_gpu_name}" if _gpu_ok else "GPU: None — integrated graphics / CPU only"
        gpu_color  = ACC_GREEN if _gpu_ok else ACC_AMBER
        ctk.CTkLabel(s_perf,text=f"  Hardware detected:  {gpu_status}",
                     font=("Segoe UI",11,"bold"),text_color=gpu_color
                     ).pack(anchor="w",padx=18,pady=(0,8))
        self.sw_cpu=ctk.CTkSwitch(s_perf,
                                   text="CPU Performance Mode",
                                   onvalue=True,offvalue=False,
                                   button_color=ACC_GREEN,progress_color=ACC_GREEN)
        if not _gpu_ok:
            self.sw_cpu.select()   # auto-ON if no GPU
        row(s_perf,"",self.sw_cpu)
        ctk.CTkLabel(s_perf,
                     text="  ON  → resize 416px, frame_skip=2  — smooth on HP Envy / Intel iGPU\n"
                          "  OFF → resize 640px, frame_skip=1  — full accuracy, use only with dedicated GPU",
                     font=("Segoe UI",10),justify="left",text_color="#64748b"
                     ).pack(anchor="w",padx=18,pady=(0,12))
        self.sw_dual=ctk.CTkSwitch(s3,text="Dual line mode (bidirectional road)",
                                    onvalue=True,offvalue=False,
                                    button_color=ACC_AMBER,progress_color=ACC_AMBER)
        row(s3,"",self.sw_dual)
        ctk.CTkLabel(s3,text="  Line A (teal) = upper = forward vehicles  |  Line B (amber) = lower = backward vehicles",
                     font=("Segoe UI",10),justify="left").pack(anchor="w",padx=16,pady=(0,6))
        self.lpos=ctk.CTkSlider(s3,from_=0.1,to=0.9,width=260); self.lpos.set(0.55)
        row(s3,"Single line position (when dual OFF)",self.lpos)
        self.lpos_a=ctk.CTkSlider(s3,from_=0.1,to=0.6,width=260); self.lpos_a.set(0.38)
        row(s3,"Line A position (upper, when dual ON)",self.lpos_a)
        self.lpos_b=ctk.CTkSlider(s3,from_=0.4,to=0.9,width=260); self.lpos_b.set(0.70)
        row(s3,"Line B position (lower, when dual ON)",self.lpos_b)

        # Custom model info
        s4=sec("Custom AI Model — Train for Bangladeshi Roads")
        ctk.CTkLabel(s4,font=("Segoe UI",12),
            text="Train YOLOv8 specifically on rickshaw, CNG, motorcycle, car, bus, truck, bicycle.\n"
                 "This will make detection angle-independent and more accurate for BD roads.",
            justify="left",wraplength=600).pack(anchor="w",padx=16,pady=(0,8))
        ctk.CTkButton(s4,text="🚀  Open Training Guide",width=200,height=36,
            fg_color=ACC_TEAL,hover_color="#0d9488",
            font=("Segoe UI",12,"bold"),
            command=lambda: subprocess.Popen([sys.executable,"train_custom_model.py"])
        ).pack(anchor="w",padx=16,pady=(0,14))

        ctk.CTkButton(scroll,text="💾  Save Settings",width=200,height=44,
            font=("Segoe UI",14,"bold"),corner_radius=10,command=self._save
        ).pack(anchor="w",pady=(4,0))
        self._load()

    def _load(self):
        try:
            import config
            m={"bd_vehicles_yolo11.pt":0,"bd_vehicles_best.pt":1,
                "yolo11n.pt":2,"yolo11s.pt":3,"yolov8n.pt":4,"yolov8s.pt":5}
            self.model_cb.set(self.model_cb.cget("values")[m.get(config.YOLO_MODEL,0)])
            self.conf.set(config.CONFIDENCE)
            self.ppm.delete(0,"end"); self.ppm.insert(0,str(config.PIXELS_PER_METER))
            self.fps_e.delete(0,"end"); self.fps_e.insert(0,str(config.VIDEO_FPS))
            (self.sw_sp.select if config.SHOW_SPEED else self.sw_sp.deselect)()
            (self.sw_id.select if config.SHOW_IDS else self.sw_id.deselect)()
            (self.sw_zo.select if config.ENABLE_ZONES else self.sw_zo.deselect)()
            cpu = getattr(config,'CPU_PERFORMANCE_MODE',True)
            (self.sw_cpu.select if cpu else self.sw_cpu.deselect)()
            dual = getattr(config,'USE_DUAL_LINES',False)
            (self.sw_dual.select if dual else self.sw_dual.deselect)()
            self.lpos.set(config.COUNTING_LINE_POSITION)
            self.lpos_a.set(getattr(config,'LINE_POS_A',0.38))
            self.lpos_b.set(getattr(config,'LINE_POS_B',0.70))
        except: pass

    def _save(self):
        try:
            # Read with utf-8, fallback to latin-1 if file has legacy encoding
            try:
                with open("config.py", encoding="utf-8") as f: c=f.read()
            except UnicodeDecodeError:
                with open("config.py", encoding="latin-1") as f: c=f.read()
            mn=["bd_vehicles_yolo11.pt","bd_vehicles_best.pt",
                "yolo11n.pt","yolo11s.pt","yolov8n.pt","yolov8s.pt"]
            idx=next((i for i,v in enumerate(self.model_cb.cget("values")) if self.model_cb.get() in v),0)
            for pat,rep in [
                (r'YOLO_MODEL\s*=\s*"[^"]*"',       f'YOLO_MODEL = "{mn[idx]}"'),
                (r'CONFIDENCE\s*=\s*[\d.]+',         f'CONFIDENCE = {self.conf.get():.2f}'),
                (r'PIXELS_PER_METER\s*=\s*\d+',      f'PIXELS_PER_METER = {self.ppm.get()}'),
                (r'VIDEO_FPS\s*=\s*\d+',             f'VIDEO_FPS = {self.fps_e.get()}'),
                (r'SHOW_SPEED\s*=\s*\w+',            f'SHOW_SPEED = {bool(self.sw_sp.get())}'),
                (r'SHOW_IDS\s*=\s*\w+',              f'SHOW_IDS = {bool(self.sw_id.get())}'),
                (r'ENABLE_ZONES\s*=\s*\w+',          f'ENABLE_ZONES = {bool(self.sw_zo.get())}'),
                (r'CPU_PERFORMANCE_MODE\s*=\s*\w+',  f'CPU_PERFORMANCE_MODE = {bool(self.sw_cpu.get())}'),
                (r'USE_DUAL_LINES\s*=\s*\w+',        f'USE_DUAL_LINES = {bool(self.sw_dual.get())}'),
                (r'COUNTING_LINE_POSITION\s*=\s*[\d.]+',f'COUNTING_LINE_POSITION = {self.lpos.get():.2f}'),
                (r'LINE_POS_A\s*=\s*[\d.]+',         f'LINE_POS_A = {self.lpos_a.get():.2f}'),
                (r'LINE_POS_B\s*=\s*[\d.]+',         f'LINE_POS_B = {self.lpos_b.get():.2f}'),
            ]: c=re.sub(pat,rep,c)
            with open("config.py","w",encoding="utf-8") as f: f.write(c)
            save_prefs({"author_name":self.name_e.get().strip() or "Nishan",
                        "institution":self.inst_e.get().strip() or "SUST · CEE Dept."})
            from tkinter import messagebox; messagebox.showinfo("Saved ✓","Settings saved ✓")
        except Exception as e:
            from tkinter import messagebox; messagebox.showerror("Error",str(e))


# ================================================================
#  ABOUT PAGE
# ================================================================
class AboutPage(Page):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure(1, weight=1)
        self.page_header("ℹ️", "About", "VELOXIS · Product information · NextCity Tessera")

        scroll = ctk.CTkScrollableFrame(self, corner_radius=0)
        scroll.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        # ── Hero ──────────────────────────────────────────────
        hero = ctk.CTkFrame(scroll, corner_radius=16, border_width=1)
        hero.pack(fill="x", padx=32, pady=(20, 0))
        hero.grid_columnconfigure(1, weight=1)

        icon_f = ctk.CTkFrame(hero, width=88, height=88, corner_radius=22)
        icon_f.grid(row=0, column=0, rowspan=3, padx=(28, 20), pady=28)
        icon_f.grid_propagate(False)
        ctk.CTkLabel(icon_f, text="🚦", font=("Segoe UI", 40)
                     ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(hero, text="VELOXIS",
                     font=("Segoe UI", 34, "bold"),
                     text_color=ACC_BLUE).grid(row=0, column=1, sticky="w", pady=(26, 2))
        ctk.CTkLabel(hero, text="AI-Powered Traffic Analysis Platform  ·  Bangladesh Edition",
                     font=("Segoe UI", 13)).grid(row=1, column=1, sticky="w")
        ctk.CTkLabel(hero,
                     text="Version 2.0  ·  YOLOv11 + BoTSORT  ·  2026  ·  MIT License",
                     font=("Segoe UI", 11),
                     text_color="#64748b").grid(row=2, column=1, sticky="w", pady=(2, 26))

        # ── Capability cards ──────────────────────────────────
        cap_f = ctk.CTkFrame(scroll, fg_color="transparent")
        cap_f.pack(fill="x", padx=32, pady=(16, 0))
        cap_f.grid_columnconfigure((0,1,2,3), weight=1)

        caps = [
            ("🎯", "Detection",    "YOLOv11 + BoTSORT\nCustom BD model\n45,862 images",  ACC_BLUE),
            ("🛺", "BD Vehicles",  "Rickshaw · CNG · Car\nMotorcycle · Bus · Truck\nBicycle · Easybike",  ACC_AMBER),
            ("📊", "Analytics",    "PHF · Headway\nSaturation flow · V85\nCSV · Vissim export", ACC_TEAL),
            ("⚡", "Performance",  "CPU & GPU modes\nReal-time HUD\nHomography speed", ACC_GREEN),
        ]
        for i, (icon, title, body, color) in enumerate(caps):
            f = ctk.CTkFrame(cap_f, corner_radius=14, border_width=1)
            f.grid(row=0, column=i, padx=(0 if i==0 else 10, 0), sticky="nsew")
            ctk.CTkFrame(f, height=4, fg_color=color, corner_radius=0).pack(fill="x")
            ctk.CTkLabel(f, text=icon, font=("Segoe UI", 28)).pack(pady=(16,4))
            ctk.CTkLabel(f, text=title, font=("Segoe UI", 12, "bold")).pack()
            ctk.CTkLabel(f, text=body, font=("Segoe UI", 10),
                         text_color="#64748b", justify="center").pack(pady=(4,16))

        # ── Developer + Research ──────────────────────────────
        dev = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        dev.pack(fill="x", padx=32, pady=(16, 0))
        dev.grid_columnconfigure((0,1), weight=1)

        # Left: developer
        dev_l = ctk.CTkFrame(dev, fg_color="transparent")
        dev_l.grid(row=0, column=0, padx=(24,12), pady=22, sticky="nsew")

        ctk.CTkLabel(dev_l, text="DEVELOPER",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#64748b").pack(anchor="w", pady=(0,10))

        dev_row = ctk.CTkFrame(dev_l, fg_color="transparent")
        dev_row.pack(fill="x")
        dev_row.grid_columnconfigure(1, weight=1)

        av = ctk.CTkFrame(dev_row, width=52, height=52, corner_radius=26,
                           fg_color=ACC_BLUE)
        av.grid(row=0, column=0, rowspan=2, padx=(0,14))
        av.grid_propagate(False)
        ctk.CTkLabel(av, text="N", font=("Segoe UI", 22, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(dev_row, text="Nishan",
                     font=("Segoe UI", 15, "bold")).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(dev_row,
                     text="B.Sc. Civil & Environmental Engineering\nSUST · 2026",
                     font=("Segoe UI", 11), text_color="#64748b",
                     justify="left").grid(row=1, column=1, sticky="w")

        # Divider
        ctk.CTkFrame(dev, width=1, fg_color="#1f2937"
                     ).grid(row=0, column=0, sticky="nse", padx=0, pady=16)

        # Right: research context
        dev_r = ctk.CTkFrame(dev, fg_color="transparent")
        dev_r.grid(row=0, column=1, padx=(12,24), pady=22, sticky="nsew")

        ctk.CTkLabel(dev_r, text="RESEARCH CONTEXT",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#64748b").pack(anchor="w", pady=(0,10))
        ctk.CTkLabel(dev_r,
                     text="VELOXIS is a research instrument for transportation\n"
                          "engineering in Bangladesh, focusing on non-motorized\n"
                          "transport (NMT), rickshaw dominance, and intersection\n"
                          "capacity analysis on mixed-traffic urban roads.",
                     font=("Segoe UI", 11), text_color="#94a3b8",
                     justify="left").pack(anchor="w")

        # ── NextCity Tessera ──────────────────────────────────
        nct = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        nct.pack(fill="x", padx=32, pady=(16, 0))

        nct_inner = ctk.CTkFrame(nct, fg_color="transparent")
        nct_inner.pack(fill="x", padx=24, pady=20)
        nct_inner.grid_columnconfigure(1, weight=1)

        nct_ic = ctk.CTkFrame(nct_inner, width=54, height=54,
                               corner_radius=14, fg_color=ACC_BLUE)
        nct_ic.grid(row=0, column=0, rowspan=2, padx=(0,18))
        nct_ic.grid_propagate(False)
        ctk.CTkLabel(nct_ic, text="🏙", font=("Segoe UI", 26)
                     ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(nct_inner, text="NextCity Tessera",
                     font=("Segoe UI", 14, "bold"),
                     text_color=ACC_BLUE).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(nct_inner,
                     text="VELOXIS is a product of NextCity Tessera — building\n"
                          "intelligent tools for urban mobility, traffic engineering,\n"
                          "and smart city research.",
                     font=("Segoe UI", 11), text_color="#64748b",
                     justify="left").grid(row=1, column=1, sticky="w")

        # ── Tech stack ────────────────────────────────────────
        tech = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        tech.pack(fill="x", padx=32, pady=(16, 0))
        ctk.CTkLabel(tech, text="TECHNOLOGY STACK",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#64748b").pack(anchor="w", padx=24, pady=(16,10))

        tech_grid = ctk.CTkFrame(tech, fg_color="transparent")
        tech_grid.pack(fill="x", padx=24, pady=(0,16))
        tech_grid.grid_columnconfigure((0,1,2,3), weight=1)

        tech_items = [
            ("YOLOv11",       "Object Detection",      ACC_BLUE),
            ("BoTSORT",       "Multi-Object Tracking", ACC_TEAL),
            ("OpenCV",        "Computer Vision",       ACC_GREEN),
            ("CustomTkinter", "Desktop UI",            ACC_PURPLE),
            ("Matplotlib",    "Analytics Charts",      ACC_AMBER),
            ("Pandas",        "Data Processing",       "#fb923c"),
            ("Homography",    "Speed Calibration",     ACC_RED),
            ("Flask",         "Web Dashboard",         "#60a5fa"),
        ]
        for i, (name_t, desc_t, col_t) in enumerate(tech_items):
            f = ctk.CTkFrame(tech_grid, corner_radius=10, border_width=1)
            f.grid(row=i//4, column=i%4,
                   padx=(0 if i%4==0 else 8, 0), pady=(0,8), sticky="ew")
            ctk.CTkFrame(f, height=3, fg_color=col_t, corner_radius=0).pack(fill="x")
            ctk.CTkLabel(f, text=name_t,
                         font=("Segoe UI", 11, "bold")).pack(pady=(10,2))
            ctk.CTkLabel(f, text=desc_t, font=("Segoe UI", 9),
                         text_color="#64748b").pack(pady=(0,10))

        # ── License ───────────────────────────────────────────
        lic = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        lic.pack(fill="x", padx=32, pady=(16, 32))

        lic_row = ctk.CTkFrame(lic, fg_color="transparent")
        lic_row.pack(fill="x", padx=24, pady=18)
        lic_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(lic_row, text="⚖️", font=("Segoe UI", 28)
                     ).grid(row=0, column=0, rowspan=2, padx=(0,16), sticky="n")
        ctk.CTkLabel(lic_row, text="MIT License",
                     font=("Segoe UI", 13, "bold")).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(lic_row,
                     text="Copyright © 2026 Nishan, NextCity Tessera. "
                          "Free to use, modify, and distribute under the standard MIT terms. "
                          "This software is provided as-is, without warranty of any kind.",
                     font=("Segoe UI", 11), text_color="#64748b",
                     justify="left", wraplength=680
                     ).grid(row=1, column=1, sticky="w", pady=(4,0))


# ================================================================
#  MAIN WINDOW
# ================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        p=load_prefs()
        name=p.get("author_name","Nishan")
        inst=p.get("institution","SUST")
        self.title(f"VELOXIS  ·  {name}, {inst}  ·  NextCity Tessera")
        self.geometry("1340x840"); self.minsize(1100,680)
        self.grid_columnconfigure(1,weight=1); self.grid_rowconfigure(0,weight=1)

        # ── Sidebar ───────────────────────────────────────────
        sb=ctk.CTkFrame(self,width=234,corner_radius=0)
        sb.grid(row=0,column=0,sticky="nsew"); sb.grid_propagate(False)
        sb.grid_rowconfigure(12,weight=1); sb.grid_columnconfigure(0,weight=1)

        # Logo block
        lf=ctk.CTkFrame(sb,corner_radius=14,border_width=1)
        lf.grid(row=0,column=0,padx=14,pady=(20,6),sticky="ew")
        li=ctk.CTkFrame(lf,fg_color="transparent"); li.pack(padx=14,pady=12,fill="x")
        ic=ctk.CTkFrame(li,width=42,height=42,corner_radius=10)
        ic.pack(side="left",padx=(0,10)); ic.pack_propagate(False)
        ctk.CTkLabel(ic,text="🚦",font=("Segoe UI",20)
                    ).place(relx=0.5,rely=0.5,anchor="center")
        ctk.CTkLabel(li,text="VELOXIS",font=("Segoe UI",15,"bold")
                    ).pack(anchor="w")
        ctk.CTkLabel(li,text="by NextCity Tessera · v1.0",font=("Segoe UI",9)
                    ).pack(anchor="w")

        ctk.CTkFrame(sb,height=1).grid(row=1,column=0,padx=14,pady=(4,6),sticky="ew")
        SLabel(sb,"Navigation").grid(row=2,column=0,padx=22,pady=(0,4),sticky="w")

        nav=[("🏠","Home"),("📹","Live Detection"),("🎬","File Detection"),
             ("🗺","Lane Drawing"),("📐","Calibrate Speed"),("📊","Analytics"),
             ("⚙️","Settings"),("ℹ️","About")]
        self.nav_btns=[]
        for i,(icon,lbl) in enumerate(nav):
            btn=NavBtn(sb,icon,lbl,lambda idx=i: self._switch(idx))
            btn.grid(row=3+i,column=0,padx=10,pady=2,sticky="ew")
            self.nav_btns.append(btn)

        ctk.CTkFrame(sb,height=1).grid(row=12,column=0,padx=14,pady=6,sticky="ew")

        # Refresh button
        ctk.CTkButton(sb,text="🔄  Refresh App",height=36,
            fg_color="transparent",border_width=1,
            font=("Segoe UI",12),corner_radius=10,
            command=self._refresh
        ).grid(row=13,column=0,padx=14,pady=(0,4),sticky="ew")

        # Theme toggle
        tr=ctk.CTkFrame(sb,fg_color="transparent")
        tr.grid(row=14,column=0,padx=14,pady=(0,6),sticky="ew")
        ctk.CTkLabel(tr,text="☀️",font=("Segoe UI",14)).pack(side="left",padx=(6,6))
        self.theme_sw=ctk.CTkSwitch(tr,text="Light mode",
            button_color=ACC_AMBER,progress_color=ACC_AMBER,
            width=44,height=22,command=self._toggle_theme,
            font=("Segoe UI",11))
        self.theme_sw.pack(side="left")
        if _THEME=="light": self.theme_sw.select()

        # Author block
        af=ctk.CTkFrame(sb,fg_color="transparent")
        af.grid(row=15,column=0,padx=14,pady=(0,18),sticky="ew")
        ctk.CTkLabel(af,text=f"👨‍💻  {name}",font=("Segoe UI",11,"bold"),
                     text_color=ACC_BLUE).pack(anchor="w")
        ctk.CTkLabel(af,text=p.get("institution","SUST · CEE Dept."),
                     font=("Segoe UI",10)).pack(anchor="w")
        ctk.CTkLabel(af,text="NextCity Tessera  ·  © 2026",font=("Segoe UI",9)
                    ).pack(anchor="w",pady=(2,0))

        # Status bar
        self.sb2=StatusBar(self)
        self.grid_rowconfigure(1,weight=0)
        self.sb2.grid(row=1,column=0,columnspan=2,sticky="ew")

        # Content
        content=ctk.CTkFrame(self,corner_radius=0)
        content.grid(row=0,column=1,sticky="nsew")
        content.grid_columnconfigure(0,weight=1); content.grid_rowconfigure(0,weight=1)

        self.hp=HomePage(content)
        self.lp=LivePage(content,status_bar=self.sb2,home_page=self.hp)
        self.fp=FilePage(content,status_bar=self.sb2,home_page=self.hp)
        self.la=LanePage(content)
        self.cal=CalibratePage(content,status_bar=self.sb2)
        self.dp=DashboardPage(content)
        self.sp=SettingsPage(content)
        self.ab=AboutPage(content)
        self._pages=[self.hp,self.lp,self.fp,self.la,self.cal,self.dp,self.sp,self.ab]
        for pg in self._pages: pg.grid(row=0,column=0,sticky="nsew")
        self._switch(0)

    def _refresh(self):
        """Reload home stats, dashboard and settings from disk."""
        try: self.hp._load_stats()
        except: pass
        try: self.dp._ever_shown=False; self.dp.refresh()
        except: pass
        try: self.sp._load()
        except: pass
        self.sb2.set("App refreshed ✓","idle")

    def _switch(self,idx):
        self._pages[idx].tkraise()
        for i,b in enumerate(self.nav_btns): b.set_active(i==idx)
        self.sb2.set(["Home","Live Detection","File Detection",
                      "Lane Drawing","Calibrate Speed","Analytics","Settings","About"][idx],"idle")
        if idx==5 and not self.dp._ever_shown: self.dp.refresh()

    def _toggle_theme(self):
        global _THEME
        _THEME="light" if _THEME=="dark" else "dark"
        ctk.set_appearance_mode(_THEME)   # instant apply
        save_prefs({"theme":_THEME})


# ── Entry point ────────────────────────────────────────────────
if __name__=="__main__":
    for d in ["videos","data","data/snapshots"]: os.makedirs(d,exist_ok=True)
    App().mainloop()
