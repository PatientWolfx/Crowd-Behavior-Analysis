from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import cv2
import numpy as np

from backend.inference import detect_people
from backend.behavior import crowd_density, detect_behavior
from utils.motion_utils import compute_motion
from utils.heatmap_utils import generate_heatmap
from utils.video_utils import create_camera_grid
from utils.alert_utils import generate_alert
from utils.logger import log_event

app = FastAPI()

templates = Jinja2Templates(directory="templates")

video_paths = [
    "data/camera1.mp4",
    "data/camera2.mp4"
]

caps = [cv2.VideoCapture(v) for v in video_paths]
prev_frames = [None] * len(video_paths)


def generate_frames():

    global prev_frames

    while True:

        frames = []

        for i, cap in enumerate(caps):

            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame,(400,300))

            people = detect_people(frame)

            count, density_status = crowd_density(people)

            motion = 0

            if prev_frames[i] is not None:
                motion = compute_motion(prev_frames[i], frame)

            behavior = detect_behavior(count, motion)
            # generate alert
            alert = generate_alert(behavior, i+1)
            
            # log event
            log_event(i+1, behavior, count)

            cv2.putText(frame, f"Camera {i+1}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            cv2.putText(frame, f"People: {count}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            cv2.putText(frame, f"Behavior: {behavior}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            if alert:
                cv2.putText(frame,
                            alert,
                            (10,130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2)

            frame = generate_heatmap(frame, people)
            if behavior in ["Congestion", "Sudden Movement"]:
                filename = f"alerts/cam{i+1}_{behavior}.jpg"
            
                cv2.imwrite(filename, frame)

            frames.append(frame)

            prev_frames[i] = frame.copy()

        dashboard = create_camera_grid(frames)

        ret, buffer = cv2.imencode('.jpg', dashboard)

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/video")
def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )