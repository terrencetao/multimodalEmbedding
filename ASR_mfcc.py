import os
import argparse 
import pickle
import numpy as np
from scipy.io import wavfile 
import librosa
from sklearn import svm
import json
from sklearn import preprocessing


from librosa.feature import mfcc


hyperams ={
	
	'input_folder': './hmm-speech-recognition-0.1/audio',

}



if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', help ='file containing a list of path to audio test')
    args = parser.parse_args()

    input_file = args.test_file

    input_folder = hyperams['input_folder'] 

    # for dirname in os.listdir(input_folder):
    #     # Get the name of the subfolder 
    #   subfolder = os.path.join(input_folder, dirname)
    #   #print(subfolder)
    #   label = subfolder[subfolder.rfind('/') + 1:]
    #   print(label)
    # filepaths = []
    
    y_words = []
    X_train=[]
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder): 
            continue
        label = subfolder[subfolder.rfind('/') + 1:]
        X = np.array([])
        
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = librosa.load(filepath)            
            mfcc_features = mfcc(sampling_freq, audio)
            if len(X) == 0:
                X = mfcc_features[:,:15]
            else:
                X = np.append(X, mfcc_features[:,:15], axis=0) 
            X_train.append(mfcc_features[:,:15].flatten())          
            y_words.append(label)
            
            
        
    print('X.shape =', X.shape)  
    print('y_words.shape =', len(y_words))  
    clf =  svm.SVC()
    clf.fit(X_train, y_words)
    

    

    with open(input_file, 'r') as f:         # load items
        files = f.readlines()
    input_files = []
    for f in files:
        input_files.append(f.strip('\n'))
    
    X_test = []
    for input_file in input_files:
        sampling_freq, audio = librosa.load(input_file)

        mfcc_features = mfcc(sampling_freq, audio)

        mfcc_features=mfcc_features[:,:15]
        X_test.append(mfcc_features.flatten())

    y_pred = clf.predict(X_test)
    print(y_pred)
    

    