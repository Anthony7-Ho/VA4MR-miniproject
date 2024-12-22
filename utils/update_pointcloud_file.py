import cv2
import numpy as np
import data_loader
from utils import get_features, get_2D_2D_correspondence, get_keyframe_distance, get_average_depth, track_fearures

def update_pointcloud(prev_keyframe, curr_frame_idx, data_params, C):
    """
    """
    K = data_params['K']
    prev_keypoints = get_features(prev_keyframe)

    for i in range(curr_frame_idx, data_params['last_frame'] + 1):
        image_i = data_loader.load_image(data_params, i, grayscale=True)
        keypoints_i = track_fearures(prev_keyframe, image_i, prev_keypoints)
        _3D_points, R , T, mask =  get_2D_2D_correspondence(prev_keypoints, keypoints_i, K)
        d = get_keyframe_distance(R , T)
        z = get_average_depth(C, _3D_points)

        if d/z >= 0.1:
            print("found next keyframe")
            return image_i, _3D_points

        elif i >= 20:
            print("failed to find next keyframe")
            return image_i, _3D_points