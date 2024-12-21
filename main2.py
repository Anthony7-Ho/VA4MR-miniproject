import data_loader
#from pipeline import *
import pipeline
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import get_features, get_2D_2D_correspondence, get_3D_2D_correspondence, get_keyframe_distance, get_average_depth, get_position

def main():
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
    keypoints, descriptors = get_features(image= Keyframes)
    
    for i in range(1, data_params['last_frame'] + 1):
        image_i = data_loader.load_image(data_params, i, grayscale=True)
        keypoints_i, descriptors_i = get_features(img = image_i)
        _3D_points, R , T =  get_2D_2D_correspondence(K, keypoints, descriptors, keypoints_i, descriptors_i)
        d = get_keyframe_distance(R , T)
        z = get_average_depth(C, _3D_points)

        if d/z >= 0.1:
            Keyframes.append(image_i)
            _3D_point_cloud.append(_3D_points)
            print("found next keyframe")
            break

        elif i >= 20:
            print("failed to find next keyframe")
            Keyframes = np.append(Keyframes,image_i)
            _3D_point_cloud = np.append(_3D_point_cloud, _3D_points)
            break

    # ---------- Main Loop -----------

    for i in range(1, data_params['last_frame'] + 1):
        image_i = data_loader.load_image(data_params, i, grayscale=True)
        keypoints_i, descriptors_i = get_features(img = image_i)
        R, T, percent_inliers = get_3D_2D_correspondence(point_cloud =_3D_point_cloud, keypoints= keypoints_i, descriptors=descriptors_i )
        if percent_inliers <= 0.2:
            break
        C = get_position(C=C, R = R, T = T)
        cv2.imshow("Current Frame", image_i)
        cv2.waitKey(30)
    
    
    

if __name__ == '__main__':
    main()