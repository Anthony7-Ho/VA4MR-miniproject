import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import cv2
from matplotlib import pyplot as plt

from plotting import plot_3d_scene, ScenePlotter
from functions import (
    detect_features,
    match_features,
    estimate_pose_from_2d2d,
    triangulate_points,
    get_keyframe_distance,
    get_average_depth,
)
import data_loader


def print_log(i, success, statistics, t_C_W):

    print()
    print(30*"-")
    print(f"Frame {i}:")
    print(f"Success: {success}")
    print(10*"-")
    print(f"Points 3D percentage: {statistics[0]:.2f}%")
    print(f"Inlier percentage: {statistics[1]:.2f}%")
    print(f"Final percentage: {statistics[2]:.2f}%")
    print(10*"-")
    print(f"Norm of translation vector: {np.linalg.norm(t_C_W)}")
    print(f"Current pose coordinates: {t_C_W.flatten()}")


@dataclass
class FrameData:
    """Data class to store frame-specific information"""
    keypoints: np.ndarray
    descriptors: np.ndarray
    correspondences: np.ndarray
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    frame_idx: int

@dataclass
class KeyframeData:
    """Data class to store keyframe selection results"""
    is_keyframe: bool
    points_3d: np.ndarray | None = None
    frame_data: FrameData | None = None
    inliers1: np.ndarray | None = None
    inlier_desc1: np.ndarray | None = None
    inliers2: np.ndarray | None = None
    inlier_desc2: np.ndarray | None = None

class VisualOdometry:
    """Visual Odometry pipeline"""

    def __init__(self, K: np.ndarray):
        """Initialize the VO pipeline
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
        self.current_frame = FrameData(
            keypoints=None,
            descriptors=None,
            correspondences=None,
            rotation=np.eye(3),
            translation=np.zeros((3, 1)),
            frame_idx=0
        )
        self.current_keyframe = KeyframeData(
            is_keyframe=True,
            points_3d=None,
            frame_data=self.current_frame,
            inliers1=None,
            inliers2=None
        )
        self.landmarks = None
        self.keyframe_update_ratio = 0.1  # Can be made configurable

    def select_keyframe(self,
                       curr_frame: np.ndarray,
                       frame_idx: int,
                       use_lowes: bool = False) -> KeyframeData:
        """Determine if the current frame should be a keyframe
        Args:
            curr_frame: Current image frame
            frame_idx: Current frame index
            use_lowes: Whether to use Lowe's ratio test
        Returns:
            KeyframeData object containing keyframe decision and data
        """
        # Detect features in current frame
        curr_kp, curr_desc = detect_features(curr_frame)

        # Match features between current and previous frame
        _, desc1, desc2, pts1, pts2 = match_features(
            self.current_keyframe.frame_data.keypoints,
            self.current_keyframe.frame_data.descriptors,
            curr_kp,
            curr_desc,
            use_lowes
        )

        # Estimate relative pose
        _, R_21, t_21, mask = estimate_pose_from_2d2d(pts1, pts2, self.K)
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        inlier_desc1 = desc1[mask.ravel() == 1]
        inlier_desc2 = desc2[mask.ravel() == 1]

        # Calculate absolute pose
        # First invert the camera1-to-camera2 transformation
        R_12 = R_21.T
        t_12 = -R_21.T @ t_21
        # Then compose with camera 1's world pose (Transform to world frame)
        R_curr = self.current_keyframe.frame_data.rotation @ R_12
        t_curr = (self.current_keyframe.frame_data.translation +
              self.current_keyframe.frame_data.rotation @ t_12)

        # Triangulate points using absolute camera poses
        points_3d = triangulate_points(
            self.K,
            self.current_keyframe.frame_data.rotation,
            self.current_keyframe.frame_data.translation,
            R_curr,
            t_curr,
            inliers1,
            inliers2
        )

        # Check keyframe criteria
        keyframe_distance = get_keyframe_distance(t_21)
        average_depth = get_average_depth(points_3d, R_curr, t_curr)

        if average_depth == 0:
            return KeyframeData(is_keyframe=False)

        is_keyframe = (keyframe_distance / average_depth) >= self.keyframe_update_ratio
        if is_keyframe:

            frame_data = FrameData(
                keypoints=curr_kp,
                descriptors=curr_desc,
                correspondences=pts2,
                rotation=R_curr,
                translation=t_curr,
                frame_idx=frame_idx
            )

            return KeyframeData(
                is_keyframe=True,
                points_3d=points_3d,
                frame_data=frame_data,
                inliers1=inliers1,
                inlier_desc1=inlier_desc1,
                inliers2=inliers2,
                inlier_desc2=inlier_desc2
            )

        return KeyframeData(is_keyframe=False)

    def initialization(self,
                      data_params: dict[str, Any],
                      use_lowes: bool = False,
                      plotting: bool = False,
                      verbose: bool = False) -> bool:
        """Initialize the VO pipeline using two keyframes
        Args:
            data_params: Dictionary containing dataset parameters
            use_lowes: Whether to use Lowe's ratio test
            plotting: Whether to plot results
        Returns:
            bool: True if initialization successful
        """
        if verbose:
            start_time = time.time()
        # Process first frame
        img0 = data_loader.load_image(data_params, 0, grayscale=True)
        kp0, desc0 = detect_features(img0)
        self.current_frame.keypoints = kp0
        self.current_frame.descriptors = desc0
        self.current_keyframe.frame_data.keypoints = kp0
        self.current_keyframe.frame_data.descriptors = desc0

        # Search for second keyframe
        for i in range(1, min(data_params["last_frame"] + 1, 11)):
            img_i = data_loader.load_image(data_params, i, grayscale=True)

            result = self.select_keyframe(img_i, i, use_lowes)

            if result.is_keyframe:
                if verbose:
                    print(30*"=")
                    print(f"Previous keyframe: {self.current_keyframe.frame_data.frame_idx}\n"
                          f"Selected keyframe at frame {i}")
                    end_time = time.time()
                    print(f"Initialization took: {end_time - start_time:.3f} seconds")

                if plotting:
                    self._plot_initialization_results(
                        img_i, result, self.current_keyframe.frame_data, verbose
                    )

                # Update state with new keyframe
                self.current_keyframe = result
                self.current_keyframe.frame_data = result.frame_data
                self.landmarks = result.points_3d
                return True

        raise ValueError("Failed to find next keyframe during initialization")

    def _plot_initialization_results(self,
                                   img: np.ndarray,
                                   result: KeyframeData,
                                   prev_keyframe: FrameData,
                                   verbose: bool = False) -> None:
        """Plot initialization results for visualization
        Args:
            img: Current keyframe image
            result: KeyframeData
            prev_frame: Previous KeyframeData
        """
        poses = [
            np.hstack([prev_keyframe.rotation, prev_keyframe.translation]),
            np.hstack([result.frame_data.rotation, result.frame_data.translation])
        ]

        if verbose:
            # Print pose coordinates for debugging
            coords_prev = prev_keyframe.translation.flatten()
            coords_curr = result.frame_data.translation.flatten()
            print(f"Previous pose coordinates: {coords_prev}")
            print(f"Current pose coordinates: {coords_curr}")
            print("IMPORTANT: Not to real scale but relative scale")

        plot_3d_scene(
            points_3d=result.points_3d,
            poses=poses,
            img=img,
            keypoints=None,
            matched_points=result.frame_data.correspondences,
            inliers1=result.inliers1,
            inliers2=result.inliers2,
            title="3D Scene Reconstruction"
        )

    def run_pnp(self, matches: list[cv2.DMatch], matched_pts_current_frame: np.ndarray) -> None:
        points_3d = self.landmarks

        # Get 3D points from the current frame
        matched_3d_indices = [m.queryIdx for m in matches]
        matched_3d_points = points_3d[matched_3d_indices]

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            matched_3d_points,
            matched_pts_current_frame,
            self.K,
            dist_coeffs,
            # flags=cv2.SOLVEPNP_P3P # TODO: Check this
        )

        points_3d_percentage = len(matched_3d_points) /len(points_3d) * 100
        inlier_percent = len(inliers) / len(matched_3d_points) * 100
        final_percentage = len(inliers) / len(points_3d) * 100
        statistics = (points_3d_percentage, inlier_percent, final_percentage)

        if success:
            R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix
            t = tvec
            return success, R, t, statistics
        return success, None, None, None

    def main_loop(self, data_params):
        """Main loop for the VO pipeline"""

        starting_frame = self.current_keyframe.frame_data.frame_idx

        scene_plotter = ScenePlotter()
        # scene_plotter.fig.show()  # Show the window once at the beginning
        # scene_plotter.initialize_plot()

        for i in range(starting_frame+1, data_params["last_frame"] + 1):
            img_i = data_loader.load_image(data_params, i, grayscale=True)
            kp_i, desc_i = detect_features(img_i)
            desc_keyframe = self.current_keyframe.inlier_desc2

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(desc_keyframe, desc_i)
            matches = sorted(matches, key=lambda x: x.distance)
            pts2 = np.float32([kp_i[m.trainIdx].pt for m in matches])

            success, R_W_C, t_W_C, statistics = self.run_pnp(matches, pts2)

            if not success:
                raise ValueError("PnP failed to find camera pose")

            # Retrieve camera pose in world frame
            R_C_W = R_W_C.T
            t_C_W = -R_W_C.T @ t_W_C

            # Print debug log
            print_log(i, success, statistics, t_C_W)

            poses = [np.hstack([R_C_W, t_C_W])]
            scene_plotter.update_plot(self.landmarks, poses, self.K,
                                    title=f"Frame {i}")
            plt.pause(0.05)  # Add small pause to allow for visualization

            # --- Main Loop Keyframe recompute ---
            if statistics[2] <= 50: #30:
                print()
                print(30*"=")
                print("Recomputing keyframe...")

                # plt.ioff()
                self.next_keyframe(R_C_W, t_C_W, kp_i, desc_i, i, img_i)
                # plt.ion()


    def next_keyframe(self, R_C_W, t_C_W, kp2, desc2, i, img_i):
        """This function is called when a new keyframe is necessary.
        Goal is to triangulate new 3d points and update the current keyframe
        and point cloud with the new values."""

        # debug print

        print(f"Shape of R_C_W: {R_C_W.shape}")
        print(f"Shape of t_C_W: {t_C_W.shape}")
        print(f"Shape of kp2: {len(kp2)}")
        print(f"Shape of desc2: {desc2.shape}")
        print(f"Shape of current keyframe keypoints: {len(self.current_keyframe.frame_data.keypoints)}")
        print(f"Shape of current keyframe descriptors: {self.current_keyframe.frame_data.descriptors.shape}")

        # match features between last keyframe and current frame
        _, matched_desc1, matched_desc2, pts1, pts2 = match_features(
            self.current_keyframe.frame_data.keypoints,
            self.current_keyframe.frame_data.descriptors,
            kp2,
            desc2,
            use_lowes=False
        )

        # # Estimate relative pose (scale ambiguous for t!)
        # _, R_21, t_21, mask = estimate_pose_from_2d2d(pts1, pts2, self.K)
        # use estimate pose to filter correspondences with Ransac and chirality check
        _, R_21, t_21, mask = estimate_pose_from_2d2d(pts1, pts2, self.K)
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        inlier_desc1 = matched_desc1[mask.ravel() == 1]
        inlier_desc2 = matched_desc2[mask.ravel() == 1]

        # fix scale:
        prev_pose = self.current_keyframe.frame_data.translation
        curr_pose = t_C_W
        relative_trans_world = curr_pose - prev_pose
        scale = np.linalg.norm(relative_trans_world)
        t_21 = t_21 * scale

        # Calculate absolute pose
        # First invert the camera1-to-camera2 transformation
        R_12 = R_21.T
        t_12 = -R_21.T @ t_21
        # Then compose with camera 1's world pose (Transform to world frame)
        R_curr = self.current_keyframe.frame_data.rotation @ R_12
        t_curr = (self.current_keyframe.frame_data.translation +
              self.current_keyframe.frame_data.rotation @ t_12)


        # Triangulate new 3d points using the poses of last keyframe and current frame
        points_3d = triangulate_points(
            self.K,
            self.current_keyframe.frame_data.rotation,
            self.current_keyframe.frame_data.translation,
            R_C_W,
            t_C_W,
            inliers1, #pts1,
            inliers2 #pts2
        )

        # comparison between estimate pose and triangulation
        print(30*"-")
        print("R_curr is from cv2.recoverPose transformed int world frame")
        print(f"R_curr: \n{R_curr}")
        print(f"R_C_W: \n{R_C_W}")
        print()
        print(f"t_curr: \n{t_curr}")
        print(f"t_C_W: \n{t_C_W}")
        print(30*"-")


        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # from plotting import _plot_3d_reconstruction
        # # new plot where i plot old landmarks and new 3dpoints
        # poses = [np.hstack([self.current_keyframe.frame_data.rotation, self.current_keyframe.frame_data.translation]),
        #          np.hstack([R_C_W, t_C_W]),
        #          np.hstack([R_curr, t_curr])]
        
        # title = "3D Scene Reconstruction"

        # fig = plt.figure(figsize=(15, 8))
        # # Create 3D subplot
        # ax_3d = plt.axes(projection="3d")
        # # ax_3d = fig.add_subplot(121, projection="3d")

        # # Filter points based on distance
        # initial_point_count = points_3d.shape[0]
        # points_3d = points_3d[np.linalg.norm(points_3d, axis=1) <= 100]
        # filtered_point_count = points_3d.shape[0]
        # print(f"Filtered out {initial_point_count - filtered_point_count} points")


        # ax_3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
        #           c="red", marker=".", s=5)
        # _plot_3d_reconstruction(ax_3d, self.landmarks, poses, title)
        # plt.show()



        # Update the current keyframe
        frame_data = FrameData(
            keypoints=kp2,
            descriptors=desc2,
            correspondences=pts2,
            rotation=R_C_W,
            translation=t_C_W,
            frame_idx=i
        )

        result= KeyframeData(
            is_keyframe=True,
            points_3d=points_3d,
            frame_data=frame_data,
            inliers1=inliers1,
            inlier_desc1=inlier_desc1,
            inliers2=inliers2,
            inlier_desc2=inlier_desc2
        )


        # # call self._plot_initialization_results
        # self._plot_initialization_results(
        #     img_i, result, self.current_keyframe.frame_data, verbose=True
        # )

        # Update state with new keyframe and landmarks
        self.current_keyframe = result
        self.landmarks = points_3d  # replace old landmarks with new ones


def main():
    # Dataset selector (0 for KITTI, 1 for Malaga, 2 for Parking)
    ds = 1
    # MIT flag für P3P lauft KITTI besser somehow

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

    # Camera intrinsics
    K = data_params["K"]

    # Initialize VO
    vo = VisualOdometry(K)
    vo.initialization(data_params, use_lowes=False, plotting=True, verbose=True)

    # return # TODO remove this later (mathieu)
    vo.main_loop(data_params)

if __name__ == "__main__":
    main()
