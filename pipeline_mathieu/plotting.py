import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def plot_matches(
    img1, img2, keypoints1, keypoints2, matches, mask, title="Feature Matches"
):
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


def plot_features_and_matches(img1, img2, pts1, pts2, matched_pts1, matched_pts2):
    """Plot correspondences between two images and the matched points on the second image"""
    # Plot images on top of each other (alpha blending)
    plt.figure(figsize=(10, 5))
    # plt.imshow(img1, cmap='gray', alpha=0.6)
    plt.imshow(img2, cmap="gray", alpha=0.7)
    # plt.scatter(pts1[:,0], pts1[:,1], c='red', s=5, label='All features img1')
    plt.scatter(pts2[:, 0], pts2[:, 1], c="red", s=5, label="All features img2")
    # plt.scatter(matched_pts1[:,0], matched_pts1[:,1], c='green', s=8, label='Matched img1')
    plt.scatter(
        matched_pts2[:, 0], matched_pts2[:, 1], c="green", s=8, label="Matched img2"
    )
    plt.legend()
    plt.show()


def plot_3d_scene(
    points_3d,
    poses,
    img,
    keypoints,
    matched_points,
    inliers1,
    inliers2,
    title="3D Scene",
):
    """
    Creates a figure with two subplots: 3D scene reconstruction and 2D feature matches.

    Args:
        points_3d (np.ndarray): 3D points of shape (N, 3)
        poses (list): List of camera pose matrices
        img (np.ndarray): Input image
        keypoints (list): List of cv2.KeyPoint objects
        matched_points (np.ndarray): Array of matched point coordinates
        inliers1 (np.ndarray): Inlier points from first image
        inliers2 (np.ndarray): Inlier points from second image
        title (str): Title for the 3D plot
    """
    fig = plt.figure(figsize=(15, 8))

    # Create 3D subplot
    ax_3d = fig.add_subplot(121, projection="3d")
    _plot_3d_reconstruction(ax_3d, points_3d, poses, title)

    # Create 2D subplot
    ax_2d = fig.add_subplot(122)
    plot_feature_matches(img, keypoints, matched_points, inliers1, inliers2, ax_2d)

    plt.tight_layout()
    plt.show()


def plot_feature_matches(img, keypoints, matched_points, inliers1, inliers2, ax=None):
    """
    Visualizes keypoints and feature matches on an image.

    Args:
        img (np.ndarray): Input image (the second image)
        keypoints (list): List of cv2.KeyPoint objects
        matched_points (np.ndarray): Array of matched point coordinates
        inliers1 (np.ndarray): Inlier points from first image
        inliers2 (np.ndarray): Inlier points from second image
        ax (matplotlib.axes.Axes, optional): Axes to plot on
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.gca()

    # Display image
    ax.imshow(img, cmap="gray")

    # Plot matched points
    if matched_points.size > 0:
        ax.scatter(
            matched_points[:, 0],
            matched_points[:, 1],
            marker="x",
            color="red",
            label=f"Matched Keypoints: {len(matched_points)}",
        )

    # Plot all keypoints
    if keypoints:
        kp_coords = np.array([kp.pt for kp in keypoints])
        ax.scatter(
            kp_coords[:, 0],
            kp_coords[:, 1],
            marker="o",
            color="blue",
            s=8,
            label="Keypoints",
        )

    # Draw inlier connections
    for pt1, pt2 in zip(inliers1, inliers2):
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "g-", linewidth=2)

    # # Add inlier count and matched keypoints count
    # ax.text(0.05, 0.90, f'Matched Keypoints: {len(matched_points)}\nInliers: {len(inliers1)}',
    #         transform=ax.transAxes, fontsize=12,
    #         bbox={"facecolor": 'white', "alpha": 0.5})

    # Add legend
    ax.plot([], [], "g-", linewidth=2, label=f"Inliers: {len(inliers1)}")
    ax.legend()
    ax.set_title("Feature Matches")
    ax.axis("off")


def _plot_3d_reconstruction(ax, points_3d, poses, title):
    """
    Helper function to plot 3D reconstruction.

    Args:
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If None, creates new figure
        points_3d (np.ndarray): 3D points to plot
        poses (list): List of camera poses
        title (str): Title for the plot
    """
    # Create figure and 3D axis if none provided
    plot_figure = False
    if ax is None:
        plot_figure = True
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

    # Plot 3D points
    ax.scatter(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c="blue", marker=".", s=5
    )

    # Plot camera poses
    colors = ["r", "g", "b"]
    for idx, pose in enumerate(poses):
        position = pose[:3, 3]
        for i, color in enumerate(colors):
            direction = pose[:3, i] * 0.5
            ax.quiver(
                position[0],
                position[1],
                position[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                length=1,
            )
        # Add label for the camera pose
        ax.text(
            position[0], position[1], position[2], str(idx), color="black", fontsize=7
        )

    # Set plot properties
    # ax.set_title(title)
    ax.text2D(
        0.5, 0.85, title, transform=ax.transAxes, ha="center", fontsize=12
    )  # manual title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_box_aspect([1, 1, 1])
    ax.set_aspect("equal")
    ax.view_init(elev=0, azim=270)

    # Show plot if we created our own figure
    if plot_figure:
        plt.show()


# ---------------------------------------------

# class ScenePlotter:
#     def __init__(self):
#         self.fig = plt.figure(figsize=(10, 6))
#         self.ax = self.fig.add_subplot(111, projection="3d")
#         self.scatter = None
#         self.quiver_objects = []
#         self.fov_lines = []

#         # Set window properties
#         self.fig.canvas.manager.set_window_title("3D Scene Viewer")
#         self._initialize_plot()
#         plt.show(block=False)

#     def _initialize_plot(self):
#         self.ax.set_xlabel("X")
#         self.ax.set_ylabel("Y")
#         self.ax.set_zlabel("Z")
#         self.ax.set_aspect("equal")
#         self.ax.view_init(elev=0, azim=270)

#     def update_plot(self, points_3d, poses, K, title="3D Scene"):
#         self._clear_previous_objects()

#         self.scatter = self.ax.scatter(
#             points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c="blue", marker=".", s=5
#         )

#         self._plot_camera_poses(poses)

#         for pose in poses:
#             self.fov_lines.extend(self._plot_camera_fov(pose, K))

#         self.ax.set_aspect("equal")
#         self.fig.canvas.draw()

#     def _clear_previous_objects(self):
#         if self.scatter is not None:
#             self.scatter.remove()
#         for quiver in self.quiver_objects:
#             quiver.remove()
#         for line in self.fov_lines:
#             line.remove()
#         self.quiver_objects = []
#         self.fov_lines = []

#     def _plot_camera_poses(self, poses):
#         colors = ["r", "g", "b"]
#         for pose in poses:
#             position = pose[:3, 3]
#             for i, color in enumerate(colors):
#                 direction = pose[:3, i] * 0.5
#                 quiver = self.ax.quiver(
#                     position[0],
#                     position[1],
#                     position[2],
#                     direction[0],
#                     direction[1],
#                     direction[2],
#                     color=color,
#                     length=1,
#                 )
#                 self.quiver_objects.append(quiver)

#     def _plot_camera_fov(self, pose, K, near=0.1, far=1.0):
#         R = pose[:3, :3]
#         t = pose[:3, 3]

#         # Calculate FOV from K matrix
#         fx = K[0, 0]
#         fy = K[1, 1]
#         width = int(2 * K[0, 2])
#         height = int(2 * K[1, 2])

#         fov_x = 2 * np.arctan(width / (2 * fx))
#         fov_y = 2 * np.arctan(height / (2 * fy))

#         # Calculate frustum corners
#         x_near = near * np.tan(fov_x/2)
#         y_near = near * np.tan(fov_y/2)
#         x_far = far * np.tan(fov_x/2)
#         y_far = far * np.tan(fov_y/2)

#         points_cam = np.array([
#             [-x_near, -y_near, near],
#             [x_near, -y_near, near],
#             [x_near, y_near, near],
#             [-x_near, y_near, near],
#             [-x_far, -y_far, far],
#             [x_far, -y_far, far],
#             [x_far, y_far, far],
#             [-x_far, y_far, far]
#         ])

#         points_world = (R @ points_cam.T + t.reshape(3, 1)).T

#         lines = [
#             [0, 1], [1, 2], [2, 3], [3, 0],
#             [4, 5], [5, 6], [6, 7], [7, 4],
#             [0, 4], [1, 5], [2, 6], [3, 7]
#         ]

#         line_objects = []
#         for line in lines:
#             p1 = points_world[line[0]]
#             p2 = points_world[line[1]]
#             line_obj, = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
#                                    [p1[2], p2[2]], 'k--', alpha=0.5)
#             line_objects.append(line_obj)

#         return line_objects


import matplotlib.pyplot as plt
import numpy as np

# class ScenePlotter:
#     def __init__(self):
#         # Create a figure with two subplots: 3D and 2D
#         self.fig = plt.figure(figsize=(15, 6))
#         self.ax_3d = self.fig.add_subplot(121, projection="3d")  # 3D plot
#         self.ax_2d = self.fig.add_subplot(122)  # 2D trajectory plot

#         # Initialize variables for objects in the plots
#         self.scatter = None
#         self.quiver_objects = []
#         self.fov_lines = []
#         self.trajectory_points = []

#         # Set window properties and initialize plots
#         self.fig.canvas.manager.set_window_title("3D Scene Viewer with Trajectory")
#         self._initialize_plot()
#         plt.show(block=False)

#     def _initialize_plot(self):
#         # Initialize the 3D plot
#         self.ax_3d.set_xlabel("X")
#         self.ax_3d.set_ylabel("Y")
#         self.ax_3d.set_zlabel("Z")
#         self.ax_3d.set_aspect("auto")
#         self.ax_3d.view_init(elev=0, azim=270)

#         # Initialize the 2D trajectory plot
#         self.ax_2d.set_xlabel("X")
#         self.ax_2d.set_ylabel("Y")
#         self.ax_2d.set_title("Camera Trajectory")

#     def update_plot(self, points_3d, poses, K, title="3D Scene"):
#         # Clear previous objects from both plots
#         self._clear_previous_objects()

#         # Update the 3D scatter plot with new points
#         if points_3d is not None:
#             self.scatter = self.ax_3d.scatter(
#                 points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
#                 c="blue", marker=".", s=5
#             )

#         # Plot camera poses and FOV in the 3D plot
#         if poses:
#             self._plot_camera_poses(poses)
#             for pose in poses:
#                 self.fov_lines.extend(self._plot_camera_fov(pose, K))

#             # Update the 2D trajectory plot with camera positions
#             self._update_trajectory_plot(poses)

#         # Refresh both plots
#         self.fig.canvas.draw()

#     def _clear_previous_objects(self):
#         # Clear objects from the 3D plot
#         if self.scatter is not None:
#             self.scatter.remove()
#         for quiver in self.quiver_objects:
#             quiver.remove()
#         for line in self.fov_lines:
#             line.remove()
        
#         # Clear objects from the 2D trajectory plot
#         self.ax_2d.clear()
        
#         # Reset lists for new objects
#         self.quiver_objects = []
#         self.fov_lines = []

#     def _plot_camera_poses(self, poses):
#         colors = ["r", "g", "b"]
        
#         for pose in poses:
#             position = pose[:3, 3]  # Camera position (translation vector)
            
#             for i, color in enumerate(colors):  # Plot orientation axes (x, y, z)
#                 direction = pose[:3, i] * 0.5
#                 quiver = self.ax_3d.quiver(
#                     position[0], position[1], position[2],
#                     direction[0], direction[1], direction[2],
#                     color=color, length=1
#                 )
#                 self.quiver_objects.append(quiver)

#     def _plot_camera_fov(self, pose, K, near=0.1, far=1.0):
#         R = pose[:3, :3]
#         t = pose[:3, 3]

#         fx = K[0, 0]
#         fy = K[1, 1]
        
#         width = int(2 * K[0, 2])
#         height = int(2 * K[1, 2])
        
#         fov_x = 2 * np.arctan(width / (2 * fx))
#         fov_y = 2 * np.arctan(height / (2 * fy))

#         x_near = near * np.tan(fov_x / 2)
#         y_near = near * np.tan(fov_y / 2)
        
#         x_far = far * np.tan(fov_x / 2)
#         y_far = far * np.tan(fov_y / 2)

#         points_cam = np.array([
#             [-x_near, -y_near, near],
#             [x_near, -y_near, near],
#             [x_near, y_near, near],
#             [-x_near, y_near, near],
#             [-x_far, -y_far, far],
#             [x_far, -y_far, far],
#             [x_far, y_far, far],
#             [-x_far, y_far, far]
#         ])

#         points_world = (R @ points_cam.T + t.reshape(3, 1)).T

#         lines = [
#             [0, 1], [1, 2], [2, 3], [3, 0],
#             [4, 5], [5, 6], [6, 7], [7, 4],
#             [0, 4], [1, 5], [2, 6], [3, 7]
#         ]

#         line_objects = []
        
#         for line in lines:
#             p1 = points_world[line[0]]
#             p2 = points_world[line[1]]
            
#             line_obj, = self.ax_3d.plot(
#                 [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
#                 'k--', alpha=0.5
#             )
            
#             line_objects.append(line_obj)

#         return line_objects

#     def _update_trajectory_plot(self, poses):
#         for pose in poses:
#             position = pose[:3, 3]  
#             self.trajectory_points.append(position[:2])  

#         trajectory_array = np.array(self.trajectory_points)
        
#         # Plot the trajectory in the updated subplot
#         if len(trajectory_array) > 0:
#             self.ax_2d.plot(
#                 trajectory_array[:, 0], trajectory_array[:, 1],
#                 'bo-', markersize=5
#             )
        

import matplotlib.pyplot as plt
import numpy as np

class ScenePlotter:
    def __init__(self):
        # Create a figure with two subplots: 3D and 2D
        self.fig = plt.figure(figsize=(15, 6))
        self.ax_3d = self.fig.add_subplot(121, projection="3d")  # 3D plot
        self.ax_2d = self.fig.add_subplot(122)  # 2D trajectory plot

        # Initialize variables for objects in the plots
        self.scatter = None
        self.quiver_objects = []
        self.fov_lines = []
        self.trajectory_points = []

        # Set window properties and initialize plots
        self.fig.canvas.manager.set_window_title("3D Scene Viewer with Trajectory")
        self._initialize_plot()
        plt.show(block=False)

    def _initialize_plot(self):
        # Initialize the 3D plot
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.set_aspect("auto")
        self.ax_3d.view_init(elev=0, azim=270)

        # Initialize the 2D trajectory plot
        self.ax_2d.set_xlabel("X (Ground Plane)")
        self.ax_2d.set_ylabel("Z (Ground Plane)")
        self.ax_2d.set_title("Camera Trajectory (Top View)")
        # self.ax_2d.set_aspect("equal")  # Set axis equal for 2D plot
        self.ax_2d.axis('equal')

    def update_plot(self, points_3d, poses, K, title="3D Scene"):
        # Clear previous objects from both plots
        self._clear_previous_objects()

        # Update the 3D scatter plot with new points
        if points_3d is not None:
            self.scatter = self.ax_3d.scatter(
                points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c="blue", marker=".", s=5
            )

        # Plot camera poses and FOV in the 3D plot
        if poses:
            self._plot_camera_poses(poses)
            for pose in poses:
                self.fov_lines.extend(self._plot_camera_fov(pose, K))

            # Update the 2D trajectory plot with camera positions (X-Z plane)
            self._update_trajectory_plot(poses)

        # Refresh both plots
        self.fig.canvas.draw()

    def _clear_previous_objects(self):
        # Clear objects from the 3D plot
        if self.scatter is not None:
            self.scatter.remove()
        for quiver in self.quiver_objects:
            quiver.remove()
        for line in self.fov_lines:
            line.remove()

        # Clear objects from the 2D trajectory plot
        self.ax_2d.clear()

        # Reset lists for new objects
        self.quiver_objects = []
        self.fov_lines = []

    def _plot_camera_poses(self, poses):
        colors = ["r", "g", "b"]

        for pose in poses:
            position = pose[:3, 3]  # Camera position (translation vector)

            for i, color in enumerate(colors):  # Plot orientation axes (x, y, z)
                direction = pose[:3, i] * 0.5
                quiver = self.ax_3d.quiver(
                    position[0], position[1], position[2],
                    direction[0], direction[1], direction[2],
                    color=color, length=1
                )
                self.quiver_objects.append(quiver)

    def _plot_camera_fov(self, pose, K, near=0.1, far=1.0):
        R = pose[:3, :3]
        t = pose[:3, 3]

        fx = K[0, 0]
        fy = K[1, 1]

        width = int(2 * K[0, 2])
        height = int(2 * K[1, 2])

        fov_x = 2 * np.arctan(width / (2 * fx))
        fov_y = 2 * np.arctan(height / (2 * fy))

        x_near = near * np.tan(fov_x / 2)
        y_near = near * np.tan(fov_y / 2)

        x_far = far * np.tan(fov_x / 2)
        y_far = far * np.tan(fov_y / 2)

        points_cam = np.array([
            [-x_near, -y_near, near],
            [x_near, -y_near, near],
            [x_near, y_near, near],
            [-x_near, y_near, near],
            [-x_far, -y_far, far],
            [x_far, -y_far, far],
            [x_far, y_far, far],
            [-x_far, y_far, far]
        ])

        points_world = (R @ points_cam.T + t.reshape(3, 1)).T

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        line_objects = []

        for line in lines:
            p1 = points_world[line[0]]
            p2 = points_world[line[1]]

            line_obj, = self.ax_3d.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                'k--', alpha=0.5
            )

            line_objects.append(line_obj)

        return line_objects

    def _update_trajectory_plot(self, poses):
        for pose in poses:
            position = pose[:3, 3]

            # Append only X and Z coordinates to trajectory_points
            xz_position = position[[0, 2]]
            self.trajectory_points.append(xz_position)

        trajectory_array = np.array(self.trajectory_points)

        # Plot the trajectory in the updated subplot (X-Z plane)
        if len(trajectory_array) > 0:
            self.ax_2d.plot(
                trajectory_array[:, 0], trajectory_array[:, 1],
                'bo-', markersize=5
            )
