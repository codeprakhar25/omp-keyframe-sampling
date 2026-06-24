"""Render an OpenPose body+hand skeleton sequence from a donor swing clip.

This is the CONTROL signal for wan_vace.py. We use controlnet_aux's
OpenposeDetector (pure-torch, downloads from lllyasviel/Annotators) rather than
DWposeDetector — DWpose needs mmpose/mmcv (heavy, and mmcv has NO prebuilt wheel
for cu128/torch2.7 + no nvcc to source-build = dead end on Blackwell).

Donor clip: a clean side-on / 3-4 forehand. Watermark is FINE — only the pose
skeleton is used, never the donor pixels. Trim to the swing window first, e.g.:
   ffmpeg -ss 7.0 -t 3.0 -i donor.mp4 -an -c:v libx264 -crf 16 istock_swing.mp4

Run in lpenv (has torch cu128 + onnxruntime): pip install controlnet_aux
"""
import cv2, numpy as np, os
from controlnet_aux import OpenposeDetector
from PIL import Image

W, H, N = 832, 480, 49          # match wan_vace.py output bucket + 4k+1 frames
SRC = "/workspace/avatar/wan_in/istock_swing.mp4"
OUT = "/workspace/avatar/vace_pose"
os.makedirs(OUT, exist_ok=True)

op = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
cap = cv2.VideoCapture(SRC); frames = []
while True:
    ret, fr = cap.read()
    if not ret: break
    frames.append(fr)
print("read frames:", len(frames), flush=True)

idx = np.linspace(0, len(frames) - 1, N).astype(int)   # subsample to N
for i, j in enumerate(idx):
    rgb = cv2.cvtColor(frames[j], cv2.COLOR_BGR2RGB)
    pose = op(Image.fromarray(rgb), include_body=True, include_hand=True,
              include_face=False, output_type="pil")
    pose.resize((W, H)).save(f"{OUT}/{i:03d}.png")
print("POSE_DONE", N, flush=True)
