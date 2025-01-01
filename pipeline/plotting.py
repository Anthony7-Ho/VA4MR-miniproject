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

def plot_features_and_matches(img1, img2, pts1, pts2, matched_pts1, matched_pts2):
    """Plot correspondences between two images and the matched points on the second image"""
    # Plot images on top of each other (alpha blending)
    plt.figure(figsize=(10, 5))
    plt.imshow(img2, cmap='gray', alpha=0.7)
    # plt.scatter(pts1[:,0], pts1[:,1], c='red', s=5, label='All features img1')
    plt.scatter(pts2[:,0], pts2[:,1], c='red', s=5, label='All features img2')
    # plt.scatter(matched_pts1[:,0], matched_pts1[:,1], c='green', s=8, label='Matched img1')
    plt.scatter(matched_pts2[:,0], matched_pts2[:,1], c='green', s=8, label='Matched img2')
    plt.legend()
    plt.show()

def plot_3d_scene(points_3d, poses, img, keypoints, matched_points, inliers1, inliers2,
                  title="3D Scene"):
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
    fig.suptitle('Keyframe Selection', fontsize=16)

    # Create 3D subplot
    ax_3d = fig.add_subplot(121, projection="3d")
    _plot_3d_reconstruction(ax_3d, points_3d, poses, title)

    # Create 2D subplot
    ax_2d = fig.add_subplot(122)
    plot_feature_matches(img, keypoints, matched_points, inliers1, inliers2, title, ax_2d)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def plot_feature_matches(img, keypoints, matched_points, inliers1, inliers2, title, ax=None):
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
    if ax is not None:
        ax.clear()

    # Display image
    ax.imshow(img, cmap='gray')

    # Plot matched points
    if matched_points.size > 0:
        ax.scatter(matched_points[:, 0], matched_points[:, 1],
                  marker='x', color='red',s=8, label=f'Matched Keypoints (3D-2D): {len(matched_points)}')

    # Plot all keypoints
    if keypoints:
        kp_coords = np.array([kp.pt for kp in keypoints])
        ax.scatter(kp_coords[:, 0], kp_coords[:, 1],
                  marker='o', color='blue', s=2, label='Keypoints')
        
    # Draw inlier connections
    if inliers1 is not None:
        for pt1, pt2 in zip(inliers1, inliers2):
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                    'g-', linewidth=1)
    # Add legend
    #ax.plot([], [], 'g-', linewidth=2, label=f'Bootstraping Inliers: {len(inliers1)}')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_title(title)
    #ax.axis("off")

def _plot_3d_reconstruction(ax, points_3d, poses, title):
    """
    Helper function to plot 3D reconstruction.
    """
    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
              c="blue", marker=".", s=5)

    # Plot camera poses
    colors = ["r", "g", "b"]
    for pose in poses:
        position = pose[:3, 3]
        for i, color in enumerate(colors):
            direction = pose[:3, i] * 0.5
            ax.quiver(position[0], position[1], position[2],
                     direction[0], direction[1], direction[2],
                     color=color, length=1)

    # Set plot properties
    # ax.set_title(title)
    ax.text2D(0.5, 0.85, title, transform=ax.transAxes, ha='center', fontsize=12) # manual title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=0, azim=270)


# ---------------------------------------------
class ScenePlotter:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 12))
        # Update subplot sizes and positions
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
   
        # Create smaller plots
        self.ax = self.fig.add_subplot(gs[0, 0])  # Top left
        self.ax_screen = self.fig.add_subplot(gs[1, :])  # Bottom full width
        self.ax_global = self.fig.add_subplot(gs[0, 1])  # Top right

        self.scatter1 = None
        self.scatter2 = None
        self.scatter3 = None

        self.quiver_objects = []
        self.fov_lines = []

    def initialize_plot(self):
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_aspect('equal')
        self.ax_global.set_xlabel("X")
        self.ax_global.set_ylabel("Y")
        self.ax_global.set_aspect('equal')

    def update_plot(self, img, keypoints, points_3d, matched_points, poses, K, title,trajectory=None):
        
        # Local trajectory
        limit = 20
        self.ax.set_xlim(poses[0][0, 3]-limit, poses[0][0, 3]+limit)
        self.ax.set_ylim(poses[0][2, 3]-limit, poses[0][2, 3]+limit)

        self.ax.set_title('Local Trajectory')

        for quiver in self.quiver_objects:
            quiver.remove()
        self.quiver_objects = []

        for line in self.fov_lines:
            line.remove()
        self.fov_lines = []

        # Update scatter plot
        if self.scatter1 is not None:
            self.scatter1.remove()
        if self.scatter2 is not None:
            self.scatter2.remove()


        self.scatter1 = self.ax.scatter(points_3d[:, 0], 
                                    points_3d[:, 2], 
                                    c="blue", marker=".", s=5,label='3D Landmarks')

        self.scatter2 = self.ax.scatter(trajectory[-30:, 0], trajectory[-30:, 2], 
                         c='g', linewidth=1, label='Last 30 Camera Poses')

        # Plot camera poses
        for pose in poses:
            position = pose[:3, 3]
            direction = pose[:3, 0] * 0.5
            quiver = self.ax.quiver(position[0], position[2],
                                    direction[0], direction[2],
                                    color="red", scale=5, label=f'Camera e_x')
            self.quiver_objects.append(quiver)
            position = pose[:3, 3]
            direction = pose[:3, 2] * 0.5
            quiver = self.ax.quiver(position[0], position[2],
                                    direction[0], direction[2],
                                    color="blue", scale=5, label=f'Camera e_z')
            self.quiver_objects.append(quiver)

            # Add FOV visualization
            self.fov_lines.extend(self.plot_camera_fov(pose, K, near=0.1, far=6.0))

        self.ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1),facecolor='white', framealpha=1.0)

        # Gobal trajectory 

        self.ax_global.set_title('Global Trajectory')
        # Get the min and max values for padding
        max_distx = 1.25*np.max(trajectory[:,0]) + 10  
        max_disty = 1.25*np.max(trajectory[:,2]) + 10
        min_distx = 1.25*np.min(trajectory[:,0]) - 10
        min_disty = 1.25*np.min(trajectory[:,2]) - 10

        # Find the overall range needed to keep axes equal
        total_range_x = max_distx - min_distx
        total_range_y = max_disty - min_disty
        max_range = max(total_range_x, total_range_y)

        # Calculate the center points
        center_x = (max_distx + min_distx) / 2
        center_y = (max_disty + min_disty) / 2

        # Set new limits maintaining equal scale
        self.ax_global.set_xlim(center_x - max_range/2, center_x + max_range/2)
        self.ax_global.set_ylim(center_y - max_range/2, center_y + max_range/2)

        if self.scatter3 is None:
            self.scatter3 = self.ax_global.scatter(trajectory[-1, 0], trajectory[-1, 2], 
                        c='orange', linewidth=0.5, label='Camera Poses')
            
        self.scatter3.set_offsets(np.c_[trajectory[:, 0], trajectory[:, 2]])
        
            
        self.ax_global.legend(loc='upper right', bbox_to_anchor=(1.2, 1),facecolor='white', framealpha=1.0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Screen Plotting

        plot_feature_matches(img, keypoints, matched_points, None, None, title, self.ax_screen)
        
    def plot_camera_fov(self, pose, K, near=0.1, far=1.0):
        # Get camera position and orientation
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Use x and z components for 2D
        forward = np.array([R[0, 2], R[2, 2]])  # Forward direction (z-axis)
        right = np.array([R[0, 0], R[2, 0]])    # Right direction (x-axis)
        pos = np.array([t[0], t[2]])            # Camera position
        
        # Calculate FOV
        fx = K[0, 0]
        width = int(2 * K[0, 2])
        fov_x = 2 * np.arctan(width / (2 * fx))
        
        # Calculate FOV corners
        x_near = near * np.tan(fov_x/2)
        x_far = far * np.tan(fov_x/2)
        
        # Corner points
        left_near = pos + near * forward - x_near * right
        right_near = pos + near * forward + x_near * right
        left_far = pos + far * forward - x_far * right
        right_far = pos + far * forward + x_far * right
        
        # Draw FOV lines
        line_objects = []
        lines = [(left_near, right_near), (left_far, right_far),
                (left_near, left_far), (right_near, right_far)]
                
        for p1, p2 in lines:
            line, = self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.5)
            line_objects.append(line)
        
        return line_objects
        