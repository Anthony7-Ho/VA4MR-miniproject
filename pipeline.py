import cv2
import numpy as np
import visualization

def initialize_3d_landmarks(img0, img1, K):
    """
    Initialize 3D landmarks using the bootstrap images.

    img0: first view
    img1: second view
    K: Camera intrinsic matrix

    return: 3D landmarks, rotation matrix, translation vector, and inliers
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img0, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img1, None)

    print("Visualizing keypoints in bootstrap images...")
    visualization.visualize_keypoints(img0, keypoints1, "Keypoints in Image 0")
    visualization.visualize_keypoints(img1, keypoints2, "Keypoints in Image 1")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    print("Visualizing matches between bootstrap images...")
    visualization.visualize_matches(img0, keypoints1, img1, keypoints2, matches)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # E: Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    _, R, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, K)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for frame 1
    P2 = K @ np.hstack((R, t))  # Projection matrix for frame 2

    points1 = inliers1.T.astype(np.float32)  # Transpose to (2, N)
    points2 = inliers2.T.astype(np.float32)  # Transpose to (2, N)

    P1 = P1.astype(np.float32)
    P2 = P2.astype(np.float32)

    #print("P1 shape:", P1.shape, "type:", P1.dtype)
    #print("P2 shape:", P2.shape, "type:", P2.dtype)
    #print("points1 shape:", points1.shape, "type:", points1.dtype)
    #print("points2 shape:", points2.shape, "type:", points2.dtype)

    points_4d_homogeneous = cv2.triangulatePoints(
        P1, P2, points1, points2
    )
    points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]
    points_3d = points_3d.T  # Transpose to Nx3 format
    #TODO: check dimensionality

    print("Visualizing triangulated 3D landmarks...")
    visualization.visualize_3d_landmarks(points_3d)

    return points_3d, R, t, inliers1, inliers2
