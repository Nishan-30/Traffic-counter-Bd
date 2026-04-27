# TrafficCounter BD — v9.0 

Author: Nishan, SUST CEE, 2026 | NextCity TESSERA

## Quick Start

1. Copy bd\_vehicles\_best.pt into this folder
2. Double-click setup\_windows.bat  (first time only)
3. Run: python app\_windows.py

## Key Settings (config.py)

YOLO\_MODEL = "bd\_vehicles\_best.pt"   # your custom model
FRAME\_SKIP = 2                        # 1=accurate 2=faster 3=fastest
ENHANCE\_NIGHT = True                  # auto-brighten dark frames
CONFIDENCE = 0.35                     # detection threshold

## Camera Sources

Webcam: Auto-scan button
DroidCam: Install app on phone, enter IP
USB: Index 1 or 2
IP Camera: rtsp:// or http:// URL

## Troubleshooting

Cannot start: pip install customtkinter pillow matplotlib pandas
Slow detection: FRAME\_SKIP=3 in config.py
No camera: Click Auto-scan in Live Detection

