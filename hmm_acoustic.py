import os
import argparse 
import pickle
import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
import librosa

from librosa.feature import mfcc


hyperams ={
	
	'input_folder': './hmm-speech-recognition-0.1/audio',

}

class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X,length):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X,length))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)


if __name__ == "__main__":

    input_folder = hyperams['input_folder'] 

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
        X = np.array([])
        y_words = []
        
        length = []
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = librosa.load(filepath)            
            mfcc_features = mfcc(sampling_freq, audio)
            mfcc_features = mfcc_features[:,:15]
            #mfcc_features = mfcc_features.reshape(len(mfcc_features),-1)
            length.append(len(mfcc_features))
            if len(X) == 0:
                X=mfcc_features
            else:
                X= np.concatenate([X,mfcc_features])            
            y_words.append(label)
            filepaths.append(filepath)
        print('X.shape =', X.shape)
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