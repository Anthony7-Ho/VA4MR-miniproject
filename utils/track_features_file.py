import cv2
import numpy as np

def track_fearures(prev_image, curr_image, prev_keypoints):
    """
    """
    prev_keypoints = np.array([kp.pt for kp in prev_keypoints], dtype=np.float32).reshape(-1, 1, 2)
    curr_keypoints, mask, _ = cv2.calcOpticalFlowPyrLK(prev_image, curr_image, prev_keypoints, None)
    curr_keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=5) for pt in curr_keypoints]

    return curr_keypoints