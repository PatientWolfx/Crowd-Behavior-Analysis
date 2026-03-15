import numpy as np
import cv2


def generate_heatmap(frame, person_boxes):

    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for box in person_boxes:

        x1, y1, x2, y2 = map(int, box[0])

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        cv2.circle(
            heatmap,
            (center_x, center_y),
            50,
            1,
            -1
        )

    heatmap = cv2.GaussianBlur(heatmap, (51,51), 0)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = heatmap.astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    return overlay