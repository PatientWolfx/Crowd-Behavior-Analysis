import csv
from datetime import datetime

def log_event(camera, behavior, people):

    with open("logs/events_log.csv","a",newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            datetime.now(),
            camera,
            behavior,
            people
        ])