import os
import argparse 
import pickle
import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
import librosa
import argparse 
import tqdm as tqdm

from librosa.feature import mfcc
import tensorflow as tf

hyperams ={
	
	'input_folder': './hmm-speech-recognition-0.1/audio',

}

class HMMTrainer(object):
    def __init__(self, model_name='GMMHMM', n_components=4, n_mix = 3, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.n_mix = n_mix
        self.models = []

        if self.model_name == 'GMMHMM':
            self.model = hmm.GMMHMM(n_components=self.n_components, n_mix=self.n_mix,
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X, length=None):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X,length))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

def padding(mfcc, max):
    step_pad = max - mfcc.shape[1]
    if step_pad >0:
        return np.pad(mfcc, pad_width=((0,0),(0,step_pad)))
    else:
        return mfcc[:,:max]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
    parser.add_argument('--size_max', help ='size of each feauture in MFCC')
    args = parser.parse_args()
    
    input_folder = args.input_folder
    size_max = int(args.size_max)

    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder 
      subfolder = os.path.join(input_folder, dirname)
      #print(subfolder)
      label = subfolder[subfolder.rfind('/') + 1:]
      print(label)
    filepaths = []
    hmm_models = []
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder): 
            continue
        label = subfolder[subfolder.rfind('/') + 1:]
        #X = np.array([])
        y_words = []
        length = []
        X=[]
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = librosa.load(filepath)            
            mfcc_features = mfcc(sampling_freq, audio)
            #mfcc_features = padding(mfcc_features,max = size_max)
            mfcc_features = mfcc_features.flatten()
            mfcc_features = mfcc_features.reshape(len(mfcc_features),-1)[:size_max]
            
            length.append(len(mfcc_features))
            #mfcc_features = mfcc_features.reshape(len(mfcc_features),-1)
            if len(X)== 0:
                X = mfcc_features
            else:
                X=np.concatenate([X,mfcc_features])
            
        

            y_words.append(label)
            filepaths.append(filepath)
        #X=tf.keras.preprocessing.sequence.pad_sequences(X,maxlen=200, padding = 'post',truncating='post' )

        print('X.shape =', X.shape)
        print('length.shape =', len(length))
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X,length)
        hmm_models.append((hmm_trainer, label))
        

	# save model
    for item in hmm_models:
        hmm_model, label = item
        with open(os.path.join('HMMs',label + ".pkl"), "wb") as file: 
            pickle.dump(hmm_model, file)

    # save list of train file
    with open('audio_train_file.txt', 'a') as f:
        for filepath in filepaths:
            f.write('%s\n' % filepath)