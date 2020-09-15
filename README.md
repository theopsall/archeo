# archeo
 ARCHEO: a python lib for sound event detection in areas of touristic Interest.

# Experiments
Experiments can be found at experiment directory, as described in the paper.

1 Train Multi Label SVC Classifier based on audio features
```bash
    python3 SVC_audio_features.py -a /home/SOURCE_DATA/ -g /home/SOURCE_LABELS/
```
2 Train Multi Label SVC Classifier based on audio features
```bash
    python3 SVC_smote_audio_features.py -a /home/SOURCE_DATA/ -g /home/SOURCE_LABELS/ -res 2000
```
3 Train Multi Label SVC Classifier based on Bag of Visual Words descriptors extracted from spectograms
```bash
    python3 BoVW.py -i /home/SPECT/ -o /home/bovw
```

4 Train Multi Label Convolutional Neural Networks based on Spectograms images
```bash
    python3 cnn.py -i /home/SPECT/
```

