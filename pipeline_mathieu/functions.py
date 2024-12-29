import cv2
import numpy as np


def detect_features(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Extract SIFT features and descriptors from an image.
    Args:
        image: Input image
    Returns:
        Tuple containing keypoints and descriptors
    """
    sift = cv2.SIFT_create()
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
) -> tuple[list[cv2.DMatch], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match features between two images.
    
    Args:
        kp1: List of keypoints from the first image
        desc1: (Nx128) Feature descriptors from the first image
        kp2: List of keypoints from the second image
        desc2: (Mx128) Feature descriptors from the second image
        use_lowes: Whether to use Lowe's ratio test
    
    Returns:
        Tuple containing:
        - good_matches: List of cv2.DMatch objects
        - desc1: (Kx128) Matched feature descriptors from the first image
        - desc2: (Kx128) Matched feature descriptors from the second image
        - pts1: (Kx2) Matched keypoint coordinates from the first image
        - pts2: (Kx2) Matched keypoint coordinates from the second image
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
        good_matches = sorted(good_matches, key=lambda x: x.distance)
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
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pose from 2D-2D correspondences using the essential matrix.
    
    Args:
        pts1: (Nx2) Array of 2D points from the first image
        pts2: (Nx2) Array of 2D points from the second image
        K: (3x3) Camera intrinsic matrix
        verbose: Whether to print detailed information
    
    Returns:
        Tuple containing:
        - E: (3x3) Essential matrix
        - R_21: (3x3) Rotation matrix from frame 1 to frame 2
        - t_21: (3x1) Translation vector from frame 1 to frame 2
        - final_inlier_mask: (Nx1) Mask of inliers used for pose estimation
    """
    E, mask_essential = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0 # TODO maybe lower threshold
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
    # Returns R_21, t_21: transformation from frame 1 to frame 2
    # R_21 rotates points from frame 1 to frame 2
    # t_21 is the position of camera 2 in camera 1 coordinates
    _, R_21, t_21, mask_pose = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)

    # Combine both masks to get final inliers
    final_inlier_mask = mask_essential.copy()
    final_inlier_mask[mask_essential.ravel() == 1] = (mask_pose.ravel() >= 1).reshape(-1, 1)

    return E, R_21, t_21, final_inlier_mask

def triangulate_points(
    K: np.ndarray,
    R1: np.ndarray, # First camera pose
    t1: np.ndarray,
    R2: np.ndarray, # Second camera pose
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from two views and transform them to world coordinates.
    
    Args:
        K: (3x3) Camera intrinsic matrix
        R1: (3x3) Rotation matrix of first camera in world frame
        t1: (3x1) Translation vector of first camera in world frame
        R2: (3x3) Rotation matrix of second camera in world frame
        t2: (3x1) Translation vector of second camera in world frame
        pts1: (Nx2) Corresponding points in first image
        pts2: (Nx2) Corresponding points in second image
    
    Returns:
        points_3d_world: (Nx3) Triangulated 3D points in world coordinates
    """
    # Calculate relative pose (camera 2 relative to camera 1)
    R_relative = (R2 @ R1.T).T  # Matches cv2.recoverPose convention
    t_relative = -R2.T @ (t2 - t1)

    # setup projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])   # First camera at origin
    P2 = K @ np.hstack([R_relative, t_relative])        # Second camera relative to first

    # Triangulate points in camera 1's frame and transform to world frame
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3, :] / points_4d[3:4, :]
    points_3d_world = (R1 @ points_3d + t1.reshape(3, 1)).T

    # alternative
    # T_w_c1 = np.vstack([np.hstack([R1, t1]), [0, 0, 0, 1]])
    # points_4d_world = T_w_c1 @ points_4d
    # points_3d = (points_4d_world[:3, :] / points_4d_world[3:4, :]).T

    return points_3d_world


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
