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

hyperams ={
	
	'input_folder': './hmm-speech-recognition-0.1/audio',

}
FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 0.74  # in seconds
SAMPLE_RATE = 22050
MONO = True

def recreate_dir(folder):
	"""
	   input :
	        folder: path to create
	  purpose : create folder or recreate if exist
	"""
	if os.path.exists(folder):
		shutil.rmtree(folder, ignore_errors=True)
	
	os.makedirs(folder)
        
class HMMTrainer(object):
    def __init__(self, model_name='GMMHMM', n_components=5, n_mix = 1, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.n_mix = n_mix
        self.models = []

        if self.model_name == 'GMMHMM':
            tmp_p = 1.0/(self.n_components-2)
            transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], 
                                    [0, tmp_p, tmp_p, tmp_p , 0], 
                                      [0, 0, tmp_p, tmp_p,tmp_p], 
                                       [0, 0, 0, 0.5, 0.5], 
                                       [0, 0, 0, 0, 1]],dtype=float)
            startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=float)
            self.model = hmm.GMMHMM(n_components=self.n_components, n_mix=self.n_mix, transmat_prior=transmatPrior, 
                                    startprob_prior=startprobPrior,
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

def padding(signal, num_expected_samples ):
    num_missing_samples = num_expected_samples - len(signal)
    if num_missing_samples>0:
        return np.pad(signal,
                             (num_missing_samples, 0),
                              mode='constant')
     
    else:
        return signal[:np.abs(num_expected_samples)]

def process(filepath):
    size_max = int(SAMPLE_RATE*DURATION)
    audio, sampling_freq = librosa.load(filepath,sr=SAMPLE_RATE,
                              duration=DURATION) 
    audio = padding(signal =audio, num_expected_samples=size_max)           
    mfcc_features = mfcc(y=audio, sr=sampling_freq, n_fft=FRAME_SIZE,
                            hop_length=HOP_LENGTH)

    return mfcc_features

if __name__ == "__main__":
   

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
   
    args = parser.parse_args()
    
    input_folder = args.input_folder
    
    
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
                       
            mfcc_features = process(filepath)
            mfcc_features = mfcc_features.reshape((len(mfcc_features)*mfcc_features.shape[1]),1)
            length.append(len(mfcc_features))
            
            if len(X)== 0:
                X = mfcc_features
            else:
                X=np.concatenate([X,mfcc_features])
            
        

            y_words.append(label)
            filepaths.append(filepath)
        

        print('X.shape =', X.shape)
        print('length.shape =', len(length))
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X,length)
        hmm_models.append((hmm_trainer, label))
        

	# save model
    recreate_dir('HMMs')
    for item in hmm_models:
        hmm_model, label = item
        with open(os.path.join('HMMs',label + ".pkl"), "wb") as file: 
            pickle.dump(hmm_model, file)

    # save list of train file
    with open('audio_train_file.txt', 'a') as f:
        for filepath in filepaths:
            f.write('%s\n' % filepath)