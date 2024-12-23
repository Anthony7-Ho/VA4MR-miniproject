import cv2
import numpy as np
from matplotlib import pyplot as plt

from plotting import (
    plot_features,
    plot_matches,
    plot_optical_flow,
    plot_3d_scene
)
import data_loader

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
    des1: np.ndarray,
    kp2: list[cv2.KeyPoint],
    des2: np.ndarray,
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
        matches = bf.knnMatch(des1, des2, k=2)
        ratio_thresh = 0.75
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        good_matches = sorted(good_matches, key=lambda x: x.distance)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        good_matches = bf.match(des1, des2)
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

    return good_matches, pts1, pts2


def estimate_pose_from_2d2d(
    pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pose from 2D-2D correspondences using the essential matrix.
    Returns:
        Tuple of (E, mask_essential, R, t, final_inliers)
    """
    E, mask_essential = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
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
        R1, t1: Pose of the first camera
        R2, t2: Pose of the second camera
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


class State:
    """Class to hold the state of our VO pipeline"""

    def __init__(self):
        self.keypoints = []  # P_i
        self.landmarks = []  # X_i
        self.candidates = []  # C_i
        self.first_observations = []  # F_i
        self.candidate_poses = []  # T_i


class VisualOdometry:
    """Visual Odometry pipeline class"""
    def __init__(self, K: np.ndarray):
        """Initialize the VO pipeline
        Args:
            K: 3x3 camera intrinsic matrix

        """
        self.K = K
        self.kp_prev = None
        self.desc_prev = None
        # Fix the first frame as world reference frame_: [R|t] = [I|0]
        self.R_prev = np.eye(3)
        self.t_prev = np.zeros((3, 1))
        self.curr_keyframe = 0
        self.landmarks = None

        self.state = State()

    def initialization(
        self, data_params: dict, use_lowes: bool = False, plotting: bool = False
    ) -> bool:
        """Initialize the VO pipeline from two keyframes using rule of thumb
        for keyframe selection.
        Args:
            data_params: Dictionary containing dataset parameters
            use_lowes: Whether to use Lowe's ratio test for feature matching
        Returns:
            bool: True if initialization successful
        """
        img0 = data_loader.load_image(data_params, 0, grayscale=True)  # First frame
        self.kp_prev, self.desc_prev = detect_features(img0)
        # plot_features(img0, self.kp_prev) # TODO: remove

        points_3d = None
        R_curr = None
        t_curr = None
        img_i = None
        for i in range(1, data_params["last_frame"] + 1):
            img_i = data_loader.load_image(data_params, i, grayscale=True)
            kp_i, des_i = detect_features(img_i)
            _, pts1, pts2 = match_features(self.kp_prev, self.desc_prev, kp_i, des_i, use_lowes)

            # estimte_pose returns relative pose of the second camera wrt the first camera
            E, R_21, t_21, mask = estimate_pose_from_2d2d(pts1, pts2, self.K)
            inliers1 = pts1[mask.ravel() == 1]
            inliers2 = pts2[mask.ravel() == 1]

            # Project points into world frame
            points_3d = triangulate_points(
                self.K, self.R_prev, self.t_prev, R_21, t_21, inliers1, inliers2
            )

            # Recover absolute pose of the second camera (in world frame)
            R_12 = R_21.T
            t_12 = -R_21.T @ t_21
            R_curr = self.R_prev @ R_12
            t_curr = self.t_prev + self.R_prev @ t_12

            keyframe_distance = get_keyframe_distance(t_21)
            average_depth = get_average_depth(points_3d, R_curr, t_curr)
            print(f"Keyframe distance: {keyframe_distance}")
            print(f"Average depth: {average_depth}")

            if average_depth == 0: # handle invalid cases where average depth is 0
                print(f"Skipping frame {i} due to zero average depth")
                continue

            if keyframe_distance / average_depth >= 0.1:
                print(30*"=")
                print(f"Selected keyframe at frame {i}")
                print(f"Previous keyframe: {self.curr_keyframe}, Current keyframe: {i}")
                self.curr_keyframe = i

                if plotting:
                    print(f"Keyframe distance: {keyframe_distance}")
                    print(f"Average depth: {average_depth}")
                    print("IMPORTANT: Not to real scale but relative scale")

                    poses = [np.hstack([self.R_prev, self.t_prev]), np.hstack([R_curr, t_curr])]
                    coords_prev = self.t_prev.flatten()
                    coords_curr = t_curr.flatten()
                    print(f"Previous pose coordinates: {coords_prev}")
                    print(f"Current pose coordinates: {coords_curr}")
                    plot_3d_scene(points_3d=points_3d,
                        poses=poses,
                        img=img_i,
                        keypoints=None, #kp_i,
                        matched_points=pts2, # matched correspondences =/= inliers1!
                        inliers1=inliers1,
                        inliers2=inliers2,
                        title="3D Scene Reconstruction"
                    )

                # update prev_variables to new keyframe
                self.R_prev = R_curr.copy()
                self.t_prev = t_curr.copy()
                self.kp_prev = kp_i
                self.desc_prev = des_i
                self.landmarks = points_3d

                break

            if i >= 10:
                # Error handling: No new keyframe was found!
                raise ValueError("Failed to find next keyframe")

        return True


def main():
    # Dataset selector (0 for KITTI, 1 for Malaga, 2 for Parking)
    ds = 0

    paths = {
        "kitti_path": "./Data/kitti05",
        "malaga_path": "./Data/malaga-urban-dataset-extract-07",
        "parking_path": "./Data/parking",
    }

    try:
        data_params = data_loader.setup_dataset(ds, paths)
    except AssertionError as e:
        print("Dataloading failed with error: ", e)
        return

    # Camera intrinsics (you'll need to provide these)
    K = data_params["K"]

    # Initialize VO
    vo = VisualOdometry(K)
    vo.initialization(data_params, use_lowes=False, plotting=True)


if __name__ == "__main__":
    main()
