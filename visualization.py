import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_keypoints(image, keypoints, window_name="Keypoints"):
    """
    Visualize keypoints detected in an image.

    image: Input image
    keypoints: Detected keypoints
    window_name: Name of the display window
    """
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color = (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(window_name, img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_matches(img1, keypoints1, img2, keypoints2, matches, window_name="Matches"):
    """
    Visualize matches between two sets of keypoints in two images.
    img1: First image
    keypoints1: Keypoints in the first image
    img2: Second image
    keypoints2: Keypoints in the second image
    matches: Matched keypoints
    window_name: Name of the display window
    """
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(window_name, img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_3d_landmarks(points_3d):
    """
    Visualize triangulated 3D points in a 3D scatter plot.
    
    points_3d: Nx3 array of 3D points
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1)
    ax.set_title("Triangulated 3D Landmarks")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def visualize_keypoints_with_inliers(image, keypoints, inliers_mask, window_name="Keypoints and Inliers"):
    """
    """
    # Convert grayscale image to BGR for colored drawing
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Iterate through keypoints and draw them
    for i, kp in enumerate(keypoints):
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        if inliers_mask is not None and inliers_mask[i]: 
            cv2.circle(output_image, pt, radius=2, color=(0, 255, 0), thickness=-1)  # Green circle: inlier
        else:  
            cv2.circle(output_image, pt, radius=2, color=(0, 0, 255), thickness=-1)  # Red circle: outlier

    # Display the image with keypoints and inliers
    cv2.imshow(window_name, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


