import os
import cv2
import numpy as np


def setup_dataset(ds, paths):
    """
    Configure dataset-specific parameters.

    ds: Dataset selector (0: KITTI, 1: Malaga, 2: Parking)
    paths: Dictionary containing dataset paths

    return: Dictionary with dataset configuration
    """
    data_params = {}

    if ds == 0:  # KITTI
        assert "kitti_path" in paths
        kitti_path = paths["kitti_path"]
        data_params["path"] = kitti_path
        data_params["ground_truth"] = np.loadtxt(
            os.path.join(kitti_path, "poses/05.txt")
        )[:, [-9, -8]]
        data_params["last_frame"] = 2760 
        data_params["K"] = np.array(
            [[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]]
        )
    elif ds == 1:  # Malaga
        assert "malaga_path" in paths
        malaga_path = paths["malaga_path"]
        images_dir = os.path.join(
            malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images"
        )
        data_params["images"] = sorted(
            [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        )[2::2]
        data_params["path"] = images_dir
        data_params["last_frame"] = len(data_params["images"]) - 1
        data_params["K"] = np.array(
            [[621.18428, 0, 404.0076], [0, 621.18428, 309.05989], [0, 0, 1]]
        )
    elif ds == 2:  # Parking
        assert "parking_path" in paths
        parking_path = paths["parking_path"]
        data_params["path"] = parking_path
        data_params["last_frame"] = 598
        data_params["K"] = np.loadtxt(os.path.join(parking_path, "K.txt"))
        data_params["ground_truth"] = np.loadtxt(
            os.path.join(parking_path, "poses.txt")
        )[:, [-9, -8]]
    else:
        raise ValueError("Dataset selection 'ds' must be 0, 1, or 2.")

    return data_params


def load_image(data_params, frame_idx, grayscale=False):
    """
    Load an image for a given frame index based on dataset parameters.

    data_params: Dictionary containing dataset configuration
    frame_idx: Frame index to load
    grayscale: Whether to load the image in grayscale

    return: Loaded image
    """
    if "path" not in data_params:
        raise ValueError("Dataset path is not set in data_params.")

    if "images" in data_params:
        # Malaga dataset
        image_path = os.path.join(data_params["path"], data_params["images"][frame_idx])
    else:
        # KITTI or Parking datasets
        if frame_idx < 0 or frame_idx > data_params["last_frame"]:
            return None
        if "kitti" in data_params["path"]:
            image_path = os.path.join(
                data_params["path"], f"05/image_0/{frame_idx:06d}.png"
            )
        elif "parking" in data_params["path"]:
            image_path = os.path.join(
                data_params["path"], f"images/img_{frame_idx:05d}.png"
            )
        else:
            raise ValueError("Unsupported dataset path structure.")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if grayscale else image
