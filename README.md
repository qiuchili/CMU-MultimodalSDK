# Step-by-step Instructions on How to Use CMU Multimodal Data SDK

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
Here the dataset_name option can be "cmu_mosei" for downloading CMU_MOSEI, "cmu_mosi" for downloading CMU_MOSI, "pom" for downloading POM.
After this step, the dataset will be downloaded to the folder /downloaded_dataset.

4. Access the downloaded dataset: 
```python
import mmsdk
from mmsdk import mmdatasdk
import numpy
cmumosei_highlevel=mmdatasdk.mmdataset('dataset_folder_path/')
```
Now the variable "cmumosei_highlevel" contains all the features and labels of the dataset. A dataset contains language, visual and acoustics modalities of features, as well as sentiment labels. All features and labels are in the form of computational sequences, stored in the dictionary object "cmumosei_highlevel.computational_sequences".

For example, the features and labels for CMU-MOSEI are: 
```python
cmumosei_highlevel.computational_sequences.keys()
<<<
'''
