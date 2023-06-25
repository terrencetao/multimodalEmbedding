#multimodalEmbedding

1- Build HMM for each word

   `python hmm_acoustic.py`
   
   NB:
2- Transform HMM to vectors

    input_file :  fichier contenant la liste des fichiers audio a convertir
    output_file :  fichier contenant les vecteurs de representation pour chaque fichier audio
    Exemple:
   `python hmm_to_vector.py --input_file audio_train_file.txt --output_file acoustic_vetors_train.json`  
   
   this command generates a json where is the vectors for each recording
   
3- Use these vectors to train svm

   `python ASR.py --train_file acoustic_vetors_train.json --test_file  acoustic_vetors_test.json`
