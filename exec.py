import os 
import subprocess
import shutil
def recreate_dir(folder):
	"""
	   input :
	        folder: path to create
	  purpose : create folder or recreate if exist
	"""
	if os.path.exists(folder):
		shutil.rmtree(folder, ignore_errors=True)
	
	os.makedirs(folder)
	
speakers = ["george","jackson","nicolas","theo","yweweler","lucas"]
recreate_dir("resultats")
for speaker in speakers:
    train_folder = os.path.join("dataset/recording_speakers" ,speaker ,"train")
    test_folder = os.path.join("dataset/recording_speakers" ,speaker , "test")
    train_txt = "audio_train_file.txt"
    test_txt = os.path.join(test_folder,"audio_test_file.txt")
    train_json = "acoustic_vetors_train.json"
    test_json = "acoustic_vetors_test.json"
    recreate_dir(os.path.join("resultats", speaker))
    hmm_res = os.path.join("resultats",speaker+"_hmm_res.txt")
    vec_res = os.path.join("resultats",speaker+"_vec_res.txt")
    asrh_res = os.path.join("resultats",speaker+"_asr_h_res.txt")
    asrm_res = os.path.join('resultats',speaker+"_asr_m_res.txt")

    print(speaker)
    hmm_train =subprocess.Popen(["python3", "hmm_acoustic.py"],stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
			      text=True,stdin = train_folder)
    #hmm_train.stdin.write()

    vec_train =subprocess.Popen(["python3", "hmm_to_vector.py"], text=True,stdout=subprocess.PIPE, 
		   stderr=subprocess.PIPE, input=train_folder+' '+train_json)
    vec_train.stdin.write(train_folder+" "+train_json)
    print('vecteurs')  
      
    vec_test =subprocess.Popen(["python3", "hmm_to_vector.py "], text=True, 
			       stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
    
    vec_test.stdin.write(test_folder+" "+test_folder)

    asrh = subprocess.Popen(["python3", "ASR_hmm.py"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    asrh.stdin.write(train_json+" "+test_json)
    
    asrm = subprocess.Popen(["python3", "ASR_mfcc.py"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    asrh.stdin.write(train_folder+" "+test_folder)
    
    
    