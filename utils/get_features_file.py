import cv2

def get_features(image):
    """
    input: 
        - image : list[ndarray]
    output:
        -
    """
    print("Type:", type(image))
    window_size = 3
    # Compute Harris corner response
    harris_response = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.08)

    # Apply threshold
    threshold = 0.01 * harris_response.max()
    thresholded_response = (harris_response > threshold) * harris_response

    # non-maximum suppression
    keypoints = []
    for y in range(harris_response.shape[0]):
        for x in range(harris_response.shape[1]):
            if thresholded_response[y, x] == 0:
                continue
            # Compare to neighbors in a window_size x window_size window
            local_max = True
            for dy in range(-window_size, window_size + 1):
                for dx in range(-window_size, window_size + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < harris_response.shape[0] and 0 <= nx < harris_response.shape[1]:
                        if harris_response[ny, nx] > harris_response[y, x]:
                            local_max = False
                            break
                if not local_max:
                    break
            if local_max:
                keypoints.append(cv2.KeyPoint(x, y, size=5, response=harris_response[y, x]))

    return keypoints