def generate_alert(behavior, camera_id):

    if behavior == "Congestion":
        return f"⚠ Congestion detected in Camera {camera_id}"

    if behavior == "Sudden Movement":
        return f"⚠ Abnormal movement in Camera {camera_id}"

    return None