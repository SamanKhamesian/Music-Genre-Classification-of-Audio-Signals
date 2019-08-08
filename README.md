# Music-Genre-Classification-of-Audio-Signals

### Abstract

Musical genres are categorical labels created by humans to characterize pieces of music.A musical genre is characterized by the common characteristics shared by its members. These characteristics typically are related to the instrumentation, rhythmic structure, and harmonic content of the music. Genre hierarchies are commonly used to structure the large collections of music available on the Web. Currently musical genre annotation is performed manually. Automatic musical genre classification can assist or replace the human user in this process and would be a valuable addition to music information retrieval systems. In addition, automatic musical genre classification provides a frame- work for developing and evaluating features for any type of con- tent-based analysis of musical signals.
This project is an implementation of music genre classification of audio signals based on machine learning techniques with python language. The main idea and theoretical approaches with their results are published in [this paper](https://pdfs.semanticscholar.org/4ccb/0d37c69200dc63d1f757eafb36ef4853c178.pdf).
##

#### To use this work on your researches or projects you need:
* Python 3.7.0
* Python packages:
	* pandas
	* librosa
	* numpy
	* joblib
	* scikit_learn

##

#### To install Python:
_First, check if you already have it installed or not_.
~~~~
python3 --version
~~~~
_If you don't have python 3 in your computer you can use the code below_:
~~~~
sudo apt-get update
sudo apt-get install python3
~~~~
##

#### To install packages via pip install:
~~~~
sudo pip3 install pandas librosa numpy joblib scikit_learn
~~~~
_If you haven't installed pip, you can use the codes below in your terminal_:
~~~~
sudo apt-get update
sudo apt install python3-pip
~~~~
_You should check and update your pip_:
~~~~
pip3 install --upgrade pip
~~~~
##

#### Dataset:
* The files that used were stored as 22 050 Hz, 16-bit, mono audio files. An effort was made to ensure that the training sets are representative of the corresponding musical genres. The Genres dataset has the following classes: classical, country, disco, hiphop, jazz, rock, blues, reggae, pop, metal. I use [GTZAN Genre Collection](http://opihi.cs.uvic.ca/sound/genres.tar.gz) for the dataset.

* NOTE : PLEASE DOWNLOAD THE DATASET AND STORE THEM IN THE CORRECT PATH BEFORE RUNNING THE PROJECT
##

#### Information about the files:
* config.py file includes some properties like dataset directory, test directory and some properties for signal processing and feature extraction.
* Utilities.py file includes useful variables and functions shared between multiple files.
* CreateDataset.py file is used for feature extraction and creating dataset.
* TrainModel.py file is used for creating and training a model.
* Classification.py file is for predicting the genres of test music files.
* Main.py file runs CreateDataset.py and TrainModel.py and Classification.py sequentially.
