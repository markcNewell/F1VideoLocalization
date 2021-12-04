import cv2
import numpy as np


def load_video_to_frames(video, step=6, locator_color=[223, 131, 69]):
    frames = []
    locations = []

    cap = cv2.VideoCapture(video)

    count = 0
    max_count = 100000

    while cap.isOpened():
        ret, frame = cap.read()
        
        if (count < max_count):
            if ret:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                minimap = frame[-250:, :300]
                loc = localize(minimap, np.array(locator_color), varience=20)

                frames.append(frame)
                locations.append(loc)

                count += step
                cap.set(1, count)
            else:
                cap.release()
                break
        else:
            cap.release()
            break

    return np.array(frames), np.array(locations)



def localize(image, color, varience=10):
    x,y,z = image.shape
    
    locator_color_ub = color + varience
    locator_color_lb = color - varience

    mask = cv2.inRange(image, locator_color_lb, locator_color_ub)  
    
    coords = cv2.findNonZero(mask)
    
    if coords is not None:
        coords = coords.reshape(-1,2)
        return np.mean(coords, axis=0)
    else:
        return np.array([0,0])
