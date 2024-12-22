import cv2
import numpy as np

def get_2D_2D_correspondence(keypoints1, tracked_points, K):
    """
    inputs: - 

    outputs:
    """
    keypoints1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32)
    tracked_points = np.array([tp.pt for tp in tracked_points], dtype=np.float32)

    E, mask = cv2.findEssentialMat(keypoints1, tracked_points, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers1 = keypoints1[mask.ravel() == 1]
    inliers2 = tracked_points[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, K)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    points1 = inliers1.T.astype(np.float32)
    points2 = inliers2.T.astype(np.float32)
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]
    points_3d = points_3d.T

    return points_3d, R, t, mask