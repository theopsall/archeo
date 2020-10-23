# archeo
 ARCHEO: a python lib for sound event detection in areas of touristic Interest.

Archeo dataset contains small audio files from 2 main data sources. The first source comes from recordings in the urban area of Athens while the second from corresponding videos, of a walk in the urban area of Athens, from YouTube. The data set consists of 13 classes in total as described in the publication.


**Link**: https://drive.google.com/file/d/1rxLlUlqj72oU2Uz9VWQREcBB3Am6YBHr/view?usp=sharing
- - -
# Experiments
Experiments, as described in the paper.

1 Train Multi Label SVC Classifier based on audio features
```bash
    python3 audio_features_classifier.py -a /home/SOURCE_DATA/ -g /home/SOURCE_LABELS/
```
2 Train Multi Label SVC Classifier with smote based on audio features
```bash
    python3 smote_audio_feature_classifier.py -a /home/SOURCE_DATA/ -g /home/SOURCE_LABELS/ -res 2000
```
3 Train Multi Label SVC Classifier based on Bag of Visual Words descriptors extracted from spectograms
```bash
    python3 bag_visual_word_classifier.py -i /home/SPECT/ -o /home/bovw
```

4 Train Multi Label Convolutional Neural Networks based on Spectograms images
```bash
    python3 cnn.py -i /home/SPECT/
```
- - -
In order to run the last two experiments, you first have to seperate the wavs files into subdirectories and then create the spectrograms.

Seperate wavs to subdirectoies base on the label
```bash
python3 source/split_wavs.py  -a home/DATA_SOURCE -g home/DATA_LABELS -o WAVS
```
Create spectrograms
```bash
python3 source/make_spectros.py  -a home/WAVS -o SPECT
