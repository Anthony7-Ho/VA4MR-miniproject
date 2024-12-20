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
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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


