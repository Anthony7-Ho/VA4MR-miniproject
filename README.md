
# VA4MR-miniproject
Miniproject of the VA4MR course in 2024

## Directory Structure
```bash
VA4MR-miniproject/
├── Data/
│   ├── kitti05
│   ├── malaga-urban-dataset-extract-07
│   └── parking
├── src/
│   ├── data_loader.py
│   ├── functions.py
│   ├── main.py
│   └── plotting.py
├── .gitignore
├── README.md
├── requirements.txt
└── environment.yml
```

## Data Setup

1. Place the following datasets into `./Data` folder:
   * kitti05
   * malaga-urban-dataset-extract-07
   * parking

## Running The Code

1. Ensure Python 3.8+ is installed (Tested on Python 3.12)

2. Install required dependencies (`pip` or `conda`):
   ```bash
   pip install -r requirements.txt
   # OR
   conda env create -f environment.yml
   ```

3. Select dataset in `./src/main.py` using the `ds` variable (line 534):
   ```python
   ds = 0  # kitti05 (Default)
   ds = 1  # malaga-urban-dataset-extract-07
   ds = 2  # parking
   ```

4. Run Main script:
   ```bash
   python src/main.py
   ```

## System Specifications
* MacBook Pro with M2 Max
* 32 GB RAM
* 12 CPU cores (10 performance, 2 efficiency)

## Demo Videos
* [KITTI 05 Dataset Demo](https://youtu.be/Rv31rvKm_hI)
* [Malaga Dataset Demo](https://youtu.be/nQ58voV8Q14)
* [Parking Dataset Demo](https://youtu.be/wTDQpprqmb8)
