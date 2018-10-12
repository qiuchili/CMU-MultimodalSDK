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
>>>import mmsdk
>>>from mmsdk import mmdatasdk
>>>import numpy
>>>cmumosei_highlevel=mmdatasdk.mmdataset('dataset_folder_path/')
```
Now the variable "cmumosei_highlevel" contains all the features and labels of the dataset. A dataset contains language, visual and acoustics modalities of features, as well as sentiment labels. All features and labels are in the form of computational sequences, stored in the dictionary object "cmumosei_highlevel.computational_sequences".

For example, the features and labels for CMU-MOSEI are: 
```python
>>>cmumosei_highlevel.computational_sequences.keys()
dict_keys(['COAVAREP', 'Emotion Labels', 'Sentiment Label', 'glove_vectors', 'FACET 4.2'])
```

Which means that the features are "COAVAREP" for acoustic features, "glove_vectors" for language features, and "FACET 4.2" for visual features, and the labels include a binary sentiment label and a finer-grained label for a video.

To access the features and labels:
```python
>>> features_or_labels = cmumosei_highlevel.computational_sequences['key_name'].data
```
The returned "features_or_labels" is a dictionary object, where the keys are the unique Youtube IDs for videos and the values are features or labels of the videos. One video has a single label and a sequence of feature vectors. Each element of the sequence contains a time interval of the video and the corresponding feature. The features and intervals are stored in different arrays:

```python
>>> features = features_or_labels[video_id]['features'].value
>>> intervals = features_or_labels[video_id]['intervals'].value
```

For example, for the video "--qXJuDtHPw", to access its sentiment label:
```python
>>> sentiment_labels = cmumosei_highlevel.computational_sequences['Sentiment Label'].data
>>> label = sentiment_labels['--qXJuDtHPw']['features'].value
>>> label
array([[1.]], dtype=float32)
```

To access its glove vectors features:
```python
>>> language_features = cmumosei_highlevel.computational_sequences['glove_vectors'].data
>>> features = language_features['--qXJuDtHPw']['features'].value
>>> intervals = language_features['--qXJuDtHPw']['intervals'].value
>>> features.shape
(183, 300)
>>> intervals.shape
(183, 2)
```
Which means there are 183 words in the video. Each word is represented as a 300-dim glove vector, and its beginning and ending time in the video is stored as a 2-dim array.

5. Multimodal data alignment:





