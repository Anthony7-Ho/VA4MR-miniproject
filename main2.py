import data_loader
#from pipeline import *
import pipeline
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import visualization

from utils import get_features, get_2D_2D_correspondence, get_3D_2D_correspondence, get_keyframe_distance, get_average_depth, get_position, track_fearures, update_pointcloud

def main():
    np.random.seed(42)
    # Dataset selector (0 for KITTI, 1 for Malaga, 2 for Parking)
    ds = 1
    paths = {
        "kitti_path": "./Data/kitti05",
        "malaga_path": "./Data/malaga-urban-dataset-extract-07",
        "parking_path": "./Data/parking"
    }
    try:
        data_params = data_loader.setup_dataset(ds, paths)
    except AssertionError as e:
        print("Dataloading failed with error: ", e)
        return
    
    # ---------- Initialize -----------

    print("Initializing 3D landmarks...")
    C = np.eye(4)
    K = data_params['K']
    _3D_point_cloud = []
    Keyframes = [data_loader.load_image(data_params,0, grayscale=True)]
    keypoints = get_features(Keyframes[0])
    #visualization.visualize_keypoints(Keyframes[0], keypoints, "Keypoints in Image 0")
    
    for i in range(1, data_params['last_frame'] + 1):
        image_i = data_loader.load_image(data_params, i, grayscale=True)
        keypoints_i = track_fearures(Keyframes[0], image_i, keypoints)
        #visualization.visualize_keypoints(image_i, keypoints_i, "Keypoints in Image i")
        _3D_points, R , T, mask =  get_2D_2D_correspondence(keypoints, keypoints_i, K)
        visualization.visualize_keypoints_with_inliers(Keyframes[0], keypoints, mask, "Inliers in Image 0")
        visualization.visualize_keypoints_with_inliers(image_i, keypoints_i, mask, "Inliers in Image i")
        visualization.visualize_3d_landmarks(_3D_points)
        d = get_keyframe_distance(R , T)
        z = get_average_depth(C, _3D_points)

        if d/z >= 0.1:
            Keyframes.append(image_i)
            _3D_point_cloud.append(_3D_points)
            print("found next keyframe")
            break

        elif i >= 20:
            print("failed to find next keyframe")
            Keyframes.append(image_i)
            _3D_point_cloud.append(_3D_points)
            break

    # ---------- Main Loop -----------

    for i in range(1, data_params['last_frame'] + 1):
        image_i = data_loader.load_image(data_params, i, grayscale=True)
        keypoints_i = get_features(img = image_i)
        R, T, percent_inliers = get_3D_2D_correspondence(point_cloud =_3D_point_cloud, keypoints= keypoints_i)
        if percent_inliers <= 0.2:
            continue
        C = get_position(C=C, R = R, T = T)
        cv2.imshow("Current Frame", image_i)
        cv2.waitKey(30)
    
    
    

if __name__ == '__main__':
    main()