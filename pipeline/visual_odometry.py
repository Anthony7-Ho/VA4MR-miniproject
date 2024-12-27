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
        self.keyframe_update_ratio = 0.05  # Can be made configurable

        # Add frame buffer to store previous frames
        self.frame_buffer = []
        self.max_buffer_size = 15

        # Add pose history tracking
        self.pose_history = {
            'rotations': [np.eye(3)],  # Initialize with first pose
            'translations': [np.zeros((3, 1))],
            'frame_indices': [0]
        }
    def update_pose_history(self, R: np.ndarray, t: np.ndarray, frame_idx: int):
        """Update pose history with new pose
        Args:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            frame_idx: Current frame index
        """
        self.pose_history['rotations'].append(R)
        self.pose_history['translations'].append(t)
        self.pose_history['frame_indices'].append(frame_idx)

    def get_trajectory_points(self):
        """Get trajectory points for plotting
        Returns:
            numpy.ndarray: Nx3 array of trajectory points
        """
        points = np.array([t.flatten() for t in self.pose_history['translations']])
        return points

    def update_frame_buffer(self, frame_data: FrameData):
        """Update the frame buffer with new frame data"""
        self.frame_buffer.append(frame_data)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)

    def find_best_frame_match(self, current_kp, current_desc):
        """Find the best matching frame from the buffer based on feature matches"""
        best_matches_count = 0
        best_frame = None
        best_matches = None
        
        for frame in self.frame_buffer:
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(frame.descriptors, current_desc)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Count good matches (you can adjust the distance threshold)
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > best_matches_count:
                best_matches_count = len(good_matches)
                best_frame = frame
                best_matches = good_matches
                
        return best_frame, best_matches

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
        # Convert relative pose from camera 2 to camera 1 perspective
        R_12 = R_21.T
        t_12 = -R_21.T @ t_21
        # Transform to world frame
        R_curr = self.current_keyframe.frame_data.rotation @ R_12
        t_curr = (self.current_keyframe.frame_data.translation +
              self.current_keyframe.frame_data.rotation @ t_12)

        # Triangulate points using relative pose
        points_3d = triangulate_points(
            self.K,
            #self.current_keyframe.frame_data.rotation,
            #self.current_keyframe.frame_data.translation,
            np.eye(3),
            np.zeros((3,1)),
            R_21,   # Use relative pose for triangulation
            t_21,
            inliers1,
            inliers2
        )
        # Check keyframe criteria
        keyframe_distance = get_keyframe_distance(t_21)
        average_depth = get_average_depth(points_3d, np.eye(3), np.zeros((3,1)))
        
        # put 3d points into world frame

        points_3d = (self.current_keyframe.frame_data.rotation @ points_3d.T + 
             self.current_keyframe.frame_data.translation.reshape(3,1)).T

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

    def select_keyframe_boot(self,
                        scene_plotter,
                        img_i: np.ndarray,
                       frame: KeyframeData,
                       use_lowes: bool = False) -> KeyframeData:
        """Determine if the current frame should be a keyframe
        Args:
            curr_frame: Current image frame
            frame_idx: Current frame index
            use_lowes: Whether to use Lowe's ratio test
        Returns:
            KeyframeData object containing keyframe decision and data
        """
        # Detect features in previous frame
        curr_kp, curr_desc = frame.keypoints, frame.descriptors

        # Match features between current and previous frame
        _, desc1, desc2, pts1, pts2 = match_features(
            curr_kp,
            curr_desc,
            self.current_keyframe.frame_data.keypoints,
            self.current_keyframe.frame_data.descriptors,
            use_lowes
        )
        _, R_21, t_21, mask = estimate_pose_from_2d2d(pts1, pts2, self.K)
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]
        inlier_desc1 = desc1[mask.ravel() == 1]
        inlier_desc2 = desc2[mask.ravel() == 1]

        R_12 = R_21.T
        t_12 = -R_21.T @ t_21

        # Triangulate points using relative pose
        points_3d = triangulate_points(
            self.K,
            np.eye(3),
            np.zeros((3,1)),
            R_21,   # Use relative pose for triangulation
            t_21,
            inliers1, 
            inliers2
        )

        # Check keyframe criteria
        keyframe_distance = get_keyframe_distance(t_21)
        average_depth = get_average_depth(points_3d, np.eye(3), np.zeros((3,1)))
        
        # put 3d points into world frame

        points_3d = (self.current_keyframe.frame_data.rotation @ points_3d.T + 
             self.current_keyframe.frame_data.translation.reshape(3,1)).T

        poses = [np.hstack([self.current_keyframe.frame_data.rotation, self.current_keyframe.frame_data.translation])]
        
        #scene_plotter.update_plot_boot(img_i, curr_kp, points_3d, pts2, inliers1, inliers2,poses, self.K,
        #                            title=f"Frame {frame.frame_idx}")

        if average_depth == 0:
            return KeyframeData(is_keyframe=False)

        is_keyframe = (keyframe_distance / average_depth) >= self.keyframe_update_ratio
        if is_keyframe:
#
            frame_data = FrameData(
                keypoints=self.current_keyframe.frame_data.keypoints,
                descriptors=self.current_keyframe.frame_data.descriptors,
                correspondences=pts2,
                rotation=self.current_keyframe.frame_data.rotation,
                translation=self.current_keyframe.frame_data.translation,
                frame_idx=self.current_keyframe.frame_data.frame_idx
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
        else:
#
            frame_data = FrameData(
                keypoints=self.current_keyframe.frame_data.keypoints,
                descriptors=self.current_keyframe.frame_data.descriptors,
                correspondences=pts2,
                rotation=self.current_keyframe.frame_data.rotation,
                translation=self.current_keyframe.frame_data.translation,
                frame_idx=self.current_keyframe.frame_data.frame_idx
            )

            return KeyframeData(
                is_keyframe=False,
                points_3d=points_3d,
                frame_data=frame_data,
                inliers1=inliers1,
                inlier_desc1=inlier_desc1,
                inliers2=inliers2,
                inlier_desc2=inlier_desc2
            )

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

    def boot(self,
                      data_params: dict[str, Any],
                      scene_plotter,
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
        self.current_frame = FrameData(
            keypoints=self.current_keyframe.frame_data.keypoints,
            descriptors=self.current_keyframe.frame_data.descriptors,
            correspondences=None,
            rotation= self.current_keyframe.frame_data.rotation,
            translation= self.current_keyframe.frame_data.translation,
            frame_idx= self.current_keyframe.frame_data.frame_idx
        )

        self.current_keyframe = KeyframeData(
            is_keyframe=True,
            points_3d=None,
            frame_data=self.current_frame,
            inliers1=None,
            inliers2=None
        )
        
        for frame in reversed(self.frame_buffer[:-1]):
            img_i = data_loader.load_image(data_params, frame.frame_idx, grayscale=True)
            result = self.select_keyframe_boot(scene_plotter,img_i,frame, use_lowes)
            if result.is_keyframe:
                if verbose:
                    print(30*"=")
                    print(f"Previous keyframe: {self.current_keyframe.frame_data.frame_idx}\n"
                          f"Selected keyframe at frame {frame.frame_idx}")
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
            flags=cv2.SOLVEPNP_P3P # TODO: Check this
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

    def main_loop(self, data_params, scene_plotter):
        """Main loop for the VO pipeline"""

        starting_frame = self.current_keyframe.frame_data.frame_idx

        for i in range(starting_frame+1, data_params["last_frame"] + 1):
            img_i = data_loader.load_image(data_params, i, grayscale=True)
            kp_i, desc_i = detect_features(img_i)
            kp_keyframe = self.current_keyframe.inliers2
            desc_keyframe = self.current_keyframe.inlier_desc2

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(desc_keyframe, desc_i)
            matches = sorted(matches, key=lambda x: x.distance)
            pts2 = np.float32([kp_i[m.trainIdx].pt for m in matches])

            success, R_W_C, t_W_C, statistics = self.run_pnp(matches, pts2)
            # Retrieve camera pose in world frame
            R_C_W = R_W_C.T
            t_C_W = -R_W_C.T @ t_W_C

            # Update frame buffer
            current_frame = FrameData(
                    keypoints=kp_i,
                    descriptors=desc_i,
                    correspondences=pts2,
                    rotation=R_C_W,
                    translation=t_C_W,
                    frame_idx=i
                )
            # Update pose history
            self.update_pose_history(R_C_W, t_C_W, i)
            self.update_frame_buffer(current_frame)

            print()
            print(f"Frame {i}:")
            print(f"Success: {success}")
            print(30*"-")
            print(f"Points 3D percentage: {statistics[0]:.2f}%")
            print(f"Inlier percentage: {statistics[1]:.2f}%")
            print(f"Final percentage: {statistics[2]:.2f}%")
            print(30*"-")
            print(f"Norm of translation vector: {np.linalg.norm(t_C_W)}")
            print(f"Current pose coordinates: {t_C_W.flatten()}")

            poses = [np.hstack([R_C_W, t_C_W])]

            trajectory_points = self.get_trajectory_points()
            scene_plotter.update_plot_loop(img_i, kp_i, self.landmarks, pts2, poses, self.K,
                                        title=f"Frame {i}", trajectory=trajectory_points)

            
            #plt.pause(1.0)  # Add small pause to allow for visualization

            # # --- Main Loop Keyframe recompute ---
            if statistics[2] <= 40:
                self.current_keyframe.frame_data.frame_idx = i
                self.current_keyframe.frame_data.rotation = R_C_W
                self.current_keyframe.frame_data.translation = t_C_W
                self.current_keyframe.frame_data.keypoints = kp_i
                self.current_keyframe.frame_data.descriptors = desc_i
                return i , R_C_W, t_C_W


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

    # Camera intrinsics
    K = data_params["K"]

    # Initialize VO
    vo = VisualOdometry(K)
    scene_plotter = ScenePlotter()
    scene_plotter.initialize_plot()
    vo.initialization(data_params,use_lowes=False, plotting=False, verbose=False)
    while vo.current_keyframe.frame_data.frame_idx < data_params["last_frame"] + 1:
        vo.main_loop(data_params, scene_plotter)
        vo.boot(data_params,scene_plotter, use_lowes=False, plotting=False, verbose=False)

if __name__ == "__main__":
    main()
