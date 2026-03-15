def crowd_density(person_boxes):

    count = len(person_boxes)

    if count < 20:
        status = "Normal Crowd"

    elif count < 50:
        status = "Moderate Crowd"

    else:
        status = "High Density"

    return count, status

def detect_behavior(density, motion):

    if density > 50 and motion > 3:
        return "Panic Movement"

    if density > 60:
        return "Congestion"

    if motion > 4:
        return "Sudden Dispersion"

    return "Normal"