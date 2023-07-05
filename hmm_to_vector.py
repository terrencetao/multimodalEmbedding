import os
import argparse 
import pickle
import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
import librosa
import json

from librosa.feature import mfcc
from hmm_acoustic import HMMTrainer
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from hmm_acoustic import process
hyperparam = {
	'items_file' : 'items.txt' 
}

def gmf(means,covars, weights , X):
	"""
	   inputs:
	     means: vector (n_features)
	     covars: matrix (n_feaures)
	     weights: vecteur (n_mix)
	        X : feature
	   output: vraissamblance de X
	"""
	p=0
	for i in range(len(weights)):
		mvn = multivariate_normal(mean= means[i], cov=covars[i], allow_singular=True)
		p = p + weights[i]*mvn.pdf(X) 

	return p

def coef_state(phi, B, feature):
	"""
	  phi: Stationnary distribution of an HMM λ
	  B : Means and covariance for each state
	  feautre : observed symbol
	output: estimates the overall proportion of time spent by λ  at observing symbol vk over a long time span
	"""
	w = 0
	means, covars, weights = B
	for i in range(len(phi)):
		
		w = w+ phi[i]*gmf(means[i],covars[i],weights[i],feature) # gmm.score need 2D(sample,feature) dimension as parameter and return an array
	
	return w


def hmm_to_vector(hmm, features):
	"""
	     inputs:
	         hmm_file : path to the  pickle file of HMM model
	     outputs : embedding of hmm

	""" 

	

	phi = hmm.get_stationary_distribution()

	nb_symbols = features.shape[0] # number of symbols
	#B = hmm.predict_proba(features)
    #compute multivariate normal distribution over features
	B = hmm.means_ , hmm.covars_, hmm.weights_
	

	w = [] # vector representation of HMM
	for k in range(nb_symbols):
		w.append(coef_state(phi, B, features[k])) 
	
	return w

def acoutic_vectors(input_file, models):
	"""
	   inputs:
	        input_files : audio file
	        models : list of HMM models
	        
	   purpose : convert audio file to vector through HMM

	   Outputs: word, vector
	"""
	
	          
	mfcc_features = process(input_file)
            # transform input to vector
	labels = input_file.split('/')[4].strip('\n')
	scores = []
	index =0
	print(labels)
	for hmm_model,label in models:
		try:
			score = hmm_model.get_score(mfcc_features)
		except:
			continue
		scores.append(score)
		# if labels == label.strip('\n'):
		# 	print('trouver')
		# 	break
	index=np.array(scores).argmax()
	
	vector = hmm_to_vector(models[index][0].model,mfcc_features)
	
	return vector

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', help ='file containing a list of path to audio')
	parser.add_argument('--output_file', help = 'file to write vectors')
	#parser.add_argument('--size_max', help ='size of each feauture in MFCC')
	args = parser.parse_args()

	input_file = args.input_file
	output_file = args.output_file
	

	with open(input_file, 'r') as f:         # load items
		files = f.readlines()
	input_files = []
	for f in files:
		input_files.append(f.strip('\n'))
	# input_files = [
    #         './hmm-speech-recognition-0.1/audio/pineapple/pineapple15.wav',
    #         './hmm-speech-recognition-0.1/audio/orange/orange15.wav',
    #         './hmm-speech-recognition-0.1/audio/apple/apple15.wav',
    #         './hmm-speech-recognition-0.1/audio/kiwi/kiwi15.wav'
    #         ]
	
	with open(hyperparam['items_file'], 'r') as f:         # load items
		items = f.readlines()

    #load hmm models
	hmm_models = []
	for item in items:	
		with open(os.path.join('HMMs',item.strip('\n') + ".pkl"), "rb") as file: 
			hmm = pickle.load(file)       
		hmm_models.append((hmm,item))

	mat_vectors={}

	for input_file in input_files:
		
		# Get acoustic vector
		vector=acoutic_vectors(input_file, hmm_models)
		
	       
	
		#  the name for the vector is item/item.wav this is for easier dataset contruction for ASR model svm
		ifi =input_file.split('/')
		mat_vectors[os.path.join(ifi[4],ifi[5])]= vector.tolist()
	
	# Serialization
	with open(output_file, "w") as outfile:
		json.dump(mat_vectors, outfile,indent=4)
	      
