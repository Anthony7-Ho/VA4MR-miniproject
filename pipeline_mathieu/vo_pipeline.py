import cv2
import numpy as np

from plotting import plot_3d_scene
from functions import (
    detect_features,
    match_features,
    estimate_pose_from_2d2d,
    triangulate_points,
    get_keyframe_distance,
    get_average_depth,
)
import data_loader

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
