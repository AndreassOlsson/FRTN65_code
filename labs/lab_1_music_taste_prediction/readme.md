# Lab 1 Music Taste Prediction

## Structure

- `data` holds the actual trainingdata, ignored by git. I placed a sample in `instructions` which is the shape that both training data `labs/lab_1_music_taste_prediction/data/training_data.csv` and test data `labs/lab_1_music_taste_prediction/data/songs_to_classify.csv` has. The sample is `labs/lab_1_music_taste_prediction/instructions/sample_data.csv`
- `instructions` holds the the files given to us, with some code examples and the pdf instructions, and the sample data
- our solution is implemented here in the root lab 1 directory.
- the final handin will be bundled as a zip file once we are done.

## About the training data

This is the readme file to the files songs_to_classify.csv and
training_data.csv.

The files contain the (unlabeled) test data and the (labeled)
training data, respectively.

The columns represent features, as specified by the header and
documented in the instructions. The column "label"
(training_data.csv. only) is encoded as 1 = like, 0 = dislike.

The files can be loaded into Python using, e.g., panda as

import pandas as pd
training=pd.read_csv('training_data.csv', sep=',')

The files were created by Andreas Svensson in August 2018 using
Spotipy (https://spotipy.readthedocs.io/)
