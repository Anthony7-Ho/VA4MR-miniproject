import cv2
import data_loader
import pipeline

from utils import randomfunction

randomfunction(number=0.5)

def main():
    # Dataset selector (0 for KITTI, 1 for Malaga, 2 for Parking)
    ds = 2

    paths = {
        "kitti_path": "./Data/kitti05",
        "malaga_path": "./Data/malaga-urban-dataset-extract-07",
        "parking_path": "./Data/parking"
    }

    bootstrap_frames = [0, 1]

    try:
        data_params = data_loader.setup_dataset(ds, paths)
    except AssertionError as e:
        print("Dataloading failed with error: ", e)
        return

    img0 = data_loader.load_image(data_params, bootstrap_frames[0], grayscale=True)
    img1 = data_loader.load_image(data_params, bootstrap_frames[1], grayscale=True)

    if img0 is None or img1 is None:
        print("Bootstrap images could not be loaded.")
        return

    print("Bootstrap images loaded successfully.")

    K = data_params['K']

    print("Initializing 3D landmarks...")
    points_3d, R, t, inliers1, inliers2 = pipeline.initialize_3d_landmarks(img0, img1, K)
    print(f"Initialized {len(points_3d)} 3D landmarks.")

    cv2.imshow("Bootstrap Image 0", img0)
    cv2.imshow("Bootstrap Image 1", img1)
    cv2.waitKey(0)  # Wait for a key press to close windows
    cv2.destroyAllWindows()

    for i in range(bootstrap_frames[1] + 1, data_params['last_frame'] + 1):
        print(f"\n\nProcessing frame {i}\n{'=' * 20}")
        image = data_loader.load_image(data_params, i, grayscale=True)
        if image is None:
            print(f"Frame {i} could not be loaded. Skipping.")
            continue

        # Show the current frame
        cv2.imshow("Current Frame", image)
        cv2.waitKey(30)  # Wait for 30ms before showing the next frame




if __name__ == '__main__':
    main()
