# Step-by-step Instructions on How to Use CMU Multimodal Data SDK

## Installation

1. Clone the github project:

```bash
git clone git@github.com:A2Zadeh/CMU-MultimodalSDK.git
```

2. Install required python packages:

```bash
pip install h5py validators tqdm numpy argparse
```

3. Download datasets: Go to the project directory, run examples/download_dataset.py.

```bash
cd CMU-MultimodalSDK
python examples/download_dataset.py dataset_name
```
Here the dataset_name option can be 'cmu_mosei' for downloading CMU_MOSEI, 'cmu_mosi' for downloading CMU_MOSI, 'pom' for downloading POM/
