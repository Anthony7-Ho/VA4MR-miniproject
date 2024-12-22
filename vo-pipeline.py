from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import data_loader


def plot_features(image, keypoints, title="Detected Features"):
    """Plot detected features on image"""
    img_with_kp = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(img_with_kp)
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_matches(img1, img2, keypoints1, keypoints2, matches, mask, title="Feature Matches"):
    """Plot matching features between two images"""
    img_matches = cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        None,
        matchesMask=mask.ravel().tolist(),
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(15, 5))
    plt.imshow(img_matches)
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_optical_flow(img1, img2, pts1, pts2, title="Optical Flow"):
    """Visualize optical flow between two frames"""
    # Create a mask for drawing
    mask = np.zeros_like(img1)

    # Draw the tracks
    for new, old in zip(pts1, pts2):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2)
        mask = cv2.circle(mask, (int(a), int(b)), 5, (255, 255, 255), -1)

    # Combine image and flow visualization
    output = cv2.addWeighted(img1, 0.8, mask, 1, 0)

    plt.figure(figsize=(10, 6))
    plt.imshow(output, "gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_3d_points_and_poses(points3D, poses, title="3D Scene"):
    """Plot 3D points and camera poses"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot 3D points
    ax.scatter(
        points3D[:, 0], points3D[:, 1], points3D[:, 2], c="blue", marker=".", s=1
    )

    # Plot camera poses
    for pose in poses:
        # Camera position
        pos = pose[:3, 3]
        # Camera orientation (plot coordinate axes)
        for i, color in enumerate(["r", "g", "b"]):
            direction = pose[:3, i] * 0.5  # Scale for visualization
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                length=1,
            )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


class State:
    """Class to hold the state of our VO pipeline"""

    def __init__(self):
        self.keypoints = []  # P_i
        self.landmarks = []  # X_i
        self.candidates = []  # C_i
        self.first_observations = []  # F_i
        self.candidate_poses = []  # T_i


class VisualOdometry:
    def __init__(self, K: np.ndarray):
        """Initialize the VO pipeline
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
        self.state = State()

    def initialize_from_frames(
        self, img0: np.ndarray, img1: np.ndarray, use_lowes: bool = False
    ) -> bool:
        """Initialize the VO pipeline from two frames
        Args:
            img0: First image
            img1: Second image
        Returns:
            bool: True if initialization successful
        """
        # 1. Extract keypoints and compute descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img0, None)
        kp2, des2 = sift.detectAndCompute(img1, None)
        # kp1, kp2: List of cv2.KeyPoint objects
        # cv2.KeyPoint attributes:
        # - pt: (x, y) coordinates of the keypoint
        # - size: diameter of the meaningful keypoint neighborhood
        # - angle: computed orientation of the keypoint (-1 if not applicable)
        # - response: the response by which the most strong keypoints have been selected
        # - octave: pyramid octave in which the keypoint has been detected
        # - class_id: object id (if the keypoints need to be clustered by an object)
        # des1, des2: numpy.ndarray of shape (N, 128) where N is the number of keypoints
        plot_features(img0, kp1, "Features in First Frame")
        plot_features(img1, kp2, "Features in Second Frame")


        # 2. Match keypoints
        if use_lowes:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des1, des2, k=2)
            ratio_thresh = 0.75
            matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
        # matches: List of cv2.DMatch objects
        # cv2.DMatch attributes:
        # - distance: distance between descriptors (lower is better)
        # - trainIdx: index of the descriptor in train descriptors
        # - queryIdx: index of the descriptor in query descriptors
        # - imgIdx: index of the train image
        matches = sorted(matches, key=lambda x: x.distance)

        # Convert matches to point correspondences
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # 3. Estimate fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if F is None:
            return False

        # Get inlier points
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]

        plot_matches(img0, img1, kp1, kp2, matches, mask, "Matches after RANSAC")

        # 4. Estimate essential matrix
        E = self.K.T @ F @ self.K

        # 5. Recover pose (R,t)
        _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, self.K)

        # 6. Triangulate points
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, :3] = R
        T2[:3, 3] = t.ravel()

        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        pts1_hom = cv2.convertPointsToHomogeneous(inliers1)[:, 0, :]
        pts2_hom = cv2.convertPointsToHomogeneous(inliers2)[:, 0, :]

        points4D = cv2.triangulatePoints(P1, P2, inliers1.T, inliers2.T)
        points3D = points4D[:3, :] / points4D[3:4, :]

        # Initialize state
        self.state.keypoints = inliers2  # Current frame keypoints
        self.state.landmarks = points3D.T  # 3D landmarks
        self.last_frame = img1
        self.last_pose = T2

        plot_3d_points_and_poses(points3D.T, [T1, T2], "Initial 3D reconstruction")

        return True

    def process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Tuple[np.ndarray, bool]:
        """Process a new frame
        Args:
            frame: New frame to process
        Returns:
            Tuple[np.ndarray, bool]: (Estimated pose, Success flag)
        """
        # 1. Track existing keypoints using KLT
        if len(self.state.keypoints) < 10:  # Need minimum points
            return None, False

        # old_pts = self.state.keypoints.astype(np.float32)
        old_pts = self.state.keypoints.reshape(-1, 1, 2).astype(np.float32)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_frame, frame, old_pts, None
        )
        plot_optical_flow(
            self.last_frame,
            frame,
            new_pts,
            old_pts,
            f"Optical Flow (Frame {frame_idx})",
        )

        # Filter only valid points
        status = status.ravel()
        old_pts = old_pts.reshape(-1, 2)
        new_pts = new_pts.reshape(-1, 2)

        good_new = new_pts[status == 1]
        good_old = old_pts[status == 1]
        good_landmarks = self.state.landmarks[status == 1]

        if len(good_new) < 8:  # Need minimum points for PnP
            return None, False

        # 2. Estimate pose using PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            good_landmarks, good_new, self.K, None
        )

        if not success:
            return None, False

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()

        # Update state with inliers only
        if inliers is not None:
            self.state.keypoints = good_new[inliers.ravel()]
            self.state.landmarks = good_landmarks[inliers.ravel()]
        else:
            self.state.keypoints = good_new
            self.state.landmarks = good_landmarks

        # 3. Detect and track new keypoints if needed
        if len(self.state.keypoints) < 50:  # Threshold for new keypoints
            sift = cv2.SIFT_create()
            new_kps = sift.detect(frame, None)
            # new_kps: List of cv2.KeyPoint objects

            # Convert keypoints to numpy array
            new_pts = np.float32([kp.pt for kp in new_kps])

            # Filter out points too close to existing ones
            if len(self.state.keypoints) > 0:
                min_dist = 10  # Minimum distance between keypoints
                valid_pts = []
                for pt in new_pts:
                    distances = np.linalg.norm(self.state.keypoints - pt, axis=1)
                    if np.min(distances) > min_dist:
                        valid_pts.append(pt)

                if valid_pts:
                    self.state.candidates.extend(valid_pts)
                    self.state.first_observations.extend(valid_pts)
                    self.state.candidate_poses.extend([self.last_pose] * len(valid_pts))

        # Update state
        self.last_frame = frame
        self.last_pose = T

        return T, True


def main():
    # Dataset selector (0 for KITTI, 1 for Malaga, 2 for Parking)
    ds = 0

    paths = {
        "kitti_path": "./Data/kitti05",
        "malaga_path": "./Data/malaga-urban-dataset-extract-07",
        "parking_path": "./Data/parking",
    }

    bootstrap_frames = [0, 2]  # Using frames 0 and 2 for initialization

    try:
        data_params = data_loader.setup_dataset(ds, paths)
    except AssertionError as e:
        print("Dataloading failed with error: ", e)
        return

    # Load initial frames
    img0 = data_loader.load_image(data_params, bootstrap_frames[0], grayscale=True)
    img1 = data_loader.load_image(data_params, bootstrap_frames[1], grayscale=True)

    # Camera intrinsics (you'll need to provide these)
    K = data_params["K"]

    # Initialize VO
    vo = VisualOdometry(K)
    if not vo.initialize_from_frames(img0, img1, use_lowes=False):
        print("Failed to initialize")
        return

    # Initialize visualization
    trajectory = []
    plt.figure(figsize=(15, 7))

    # Create two subplots
    plt.subplot(121)  # Left plot for full trajectory
    plt.subplot(122)  # Right plot for recent frames

    # Process remaining frames
    frame_idx = bootstrap_frames[1] + 1
    while True:
        try:
            frame = data_loader.load_image(data_params, frame_idx, grayscale=True)
        except:
            break

        pose, success = vo.process_frame(frame, frame_idx)
        if success:
            # Extract position from pose
            position = pose[:3, 3]
            trajectory.append(position[:2])  # Keep only x,y for 2D plot

            # Convert to numpy array for easier slicing
            trajectory_arr = np.array(trajectory)

            # Clear and update plots
            plt.clf()

            # Full trajectory plot
            plt.subplot(121)
            plt.plot(trajectory_arr[:, 0], trajectory_arr[:, 1], "b-")
            plt.scatter(
                trajectory_arr[-1, 0],
                trajectory_arr[-1, 1],
                c="red",
                marker="x",
                s=100,
                label="Current Position",
            )
            plt.title("Full Trajectory")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()

            # Recent frames plot (last 20 frames)
            plt.subplot(122)
            recent_trajectory = (
                trajectory_arr[-20:] if len(trajectory_arr) > 20 else trajectory_arr
            )
            plt.plot(recent_trajectory[:, 0], recent_trajectory[:, 1], "g-")
            plt.scatter(
                recent_trajectory[-1, 0],
                recent_trajectory[-1, 1],
                c="red",
                marker="x",
                s=100,
                label="Current Position",
            )
            plt.title(f"Last 20 Frames (Current: {frame_idx})")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.pause(0.01)

        frame_idx += 1

    plt.show()


if __name__ == "__main__":
    main()
