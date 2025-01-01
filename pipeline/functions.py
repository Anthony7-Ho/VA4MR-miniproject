import cv2
import numpy as np
import matplotlib.pyplot as plt

from plotting import plot_3d_scene
import data_loader

def detect_features(image: np.ndarray, current_matches:np.ndarray ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Extract SIFT features and descriptors from an image.
    Args:
        image: Input image
    Returns:
        Tuple containing keypoints and descriptors
    """
    sift = cv2.SIFT_create()
    if current_matches is not None:
        upper_bound = min(6000, 300/len(current_matches)*2000)
        nfeatures_des = max(3000, int(upper_bound))
        # print(len(current_matches))
        # print(nfeatures_des)
    else: nfeatures_des = 3000
    
    sift = cv2.SIFT_create(nfeatures=nfeatures_des , nOctaveLayers=8, contrastThreshold=0.02, edgeThreshold=15, sigma=1.6)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # keypoints: List of cv2.KeyPoint objects
    # cv2.KeyPoint attributes:
    # - pt: (x, y) coordinates of the keypoint
    # - size: diameter of the meaningful keypoint neighborhood
    # - angle: computed orientation of the keypoint (-1 if not applicable)
    # - response: the response by which the most strong keypoints have been selected
    # - octave: pyramid octave in which the keypoint has been detected
    # - class_id: object id (if the keypoints need to be clustered by an object)
    # descriptors: numpy.ndarray of shape (N, 128) where N is the number of keypoints
    return keypoints, descriptors

def match_features(
    kp1: list[cv2.KeyPoint],
    desc1: np.ndarray,
    kp2: list[cv2.KeyPoint],
    desc2: np.ndarray,
    use_lowes: bool = False,
) -> tuple[list[cv2.DMatch], np.ndarray, np.ndarray]:
    """Match features between two images.
    Args:
        kp1, kp2: Keypoints from the two images
        des1, des2: Feature descriptors
        use_lowes: Whether to use Lowe's ratio test
    Returns:
        List of matches and point correspondences
    """
    if use_lowes:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(desc1, desc2, k=2)
        ratio_thresh = 0.75
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        good_matches = sorted(good_matches, key=lambda x: x.distance)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        good_matches = bf.match(desc1, desc2)
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:1500]
    # good_matches: List of cv2.DMatch objects
    # cv2.DMatch attributes:
    # - queryIdx: Index of the descriptor in the first set
    # - trainIdx: Index of the descriptor in the second set
    # - imgIdx: Index of the train image (only when multiple images are trained)
    # - distance: Distance between the descriptors

    # Convert matches to point correspondences
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    desc1 = np.float32([desc1[m.queryIdx] for m in good_matches])
    desc2 = np.float32([desc2[m.trainIdx] for m in good_matches])

    return good_matches, desc1, desc2, pts1, pts2

def estimate_pose_from_2d2d(
    pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pose from 2D-2D correspondences using the essential matrix.
    Returns:
        Tuple of (E, mask_essential, R, t, final_inliers)
    """
    E, mask_essential = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=0.4
    )
    # Check if the essential matrix E has rank 2
    rank = np.linalg.matrix_rank(E)
    if rank != 2:
        U, S, Vt = np.linalg.svd(E)
        S[2] = 0  # Keep original scale of first two values
        E = U @ np.diag(S) @ Vt

    # Verbose output
    if verbose:
        print("Essential matrix rank:", rank)
        if rank != 2:
            print("Essential matrix rank after SVD:", np.linalg.matrix_rank(E))

    # Use only inliers from essential matrix for pose recovery
    inlier_pts1 = pts1[mask_essential.ravel() == 1]
    inlier_pts2 = pts2[mask_essential.ravel() == 1]

    # Recover pose
    # Returns R_21, t_21: transformation from frame 2 to frame 1
    # R_21 rotates points from frame 2 to frame 1
    # t_21 is the position of camera 1 as seen from camera 2
    _, R_21, t_21, mask_pose = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)

    # Combine both masks to get final inliers
    final_inlier_mask = mask_essential.copy()
    final_inlier_mask[mask_essential.ravel() == 1] = (mask_pose.ravel() >= 1).reshape(-1, 1)

    return E, R_21, t_21, final_inlier_mask #mask_essential

def triangulate_points(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,  # First camera pose
    R2: np.ndarray,
    t2: np.ndarray,  # Second camera pose
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from two views.
    Args:
        K: Camera intrinsic matrix
        pts1, pts2: Corresponding points in the two views
        R1, t1: Pose of the first camera (world frame)
        R2, t2: Pose of the second camera (camera 1 frame)
    Returns:
        3D points array
    """
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])

    points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    # Convert from homogeneous to Euclidean coordinates
    points3D = (points4D[:3, :] / points4D[3:4, :]).T
    return points3D

def get_keyframe_distance(t_relative: np.ndarray) -> float:
    """Calculate the distance between two keyframes.
    Args:
        t_relative: Relative translation vector between two keyframes
    Returns:
        Distance between the two keyframes
    """
    return np.linalg.norm(t_relative)

def get_average_depth(points3D: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """Calculates the average depth of 3D points in the camera frame.
    Args:
        points3D: 3D points in the world frame
        R: Rotation matrix from world to camera frame
        t: Translation vector from world to camera frame
    Returns:
        Average depth of 3D points in the camera frame
    """
    # Inverse transform to get points in camera frame
    R_cam_to_world = R.T
    t_cam_to_world = -R.T @ t

    points3D_camera = (R_cam_to_world @ points3D.T) + t_cam_to_world.reshape(3,1)
    points3D_camera = points3D_camera.T

    # old code, likely wrong (not 100% sure yet)
    # points3D_camera = (R @ points3D.T) + t
    # points3D_camera = points3D_camera.T

    # Extract depths (z-coordinates)
    depths = points3D_camera[:, 2]

    # Check for and handle negative depths
    # depths = depths[depths > 0] # original code
    num_negative_depths = np.sum(depths < 0)
    if num_negative_depths > 0:
        print(f"WARNING: Removed {num_negative_depths} negative depth values.")
        depths = depths[depths >= 0] # include 0 values which can also be problematic

    if len(depths) == 0:
        # handle case outside function
        return 0

    return np.mean(depths)
