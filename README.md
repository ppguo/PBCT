Requirements:
python==3.9
numpy==1.23.3
pandas==1.4.4
scikit-learn==1.1.2
scipy==1.9.1
gurobi==9.5.2  


--- Project
    - utils/
	PBCT.py
	SFS.py
	solve_loss.py
    - Data/
	train_20_feature.csv (dataset1)
	pri_20_feature.csv (dataset2)
	sec_20_feature.csv (dataset3)
    - Offline/
	PBCT_offline.py
    - Online/
	PBCT_online.py
    - Unlabeled_Offline/
	PBCT_offline_unlabeled.py
    - Unlabeled_Online/
	PBCT_online_unlabeled.py


The core function of the PBCT algorithm is included in the file utils/PBCT.py. Given the labeled and unlabeled training data as well as the test data, it triggers the training of the complete-view model and parital-view models, save the model parameters in the desired paths, and return the test error measured using RMSE. An example for utilizing the PBCT algorithm is provided in the __main__ section of this file.

As four scenarios have been considered in this work, we provide four scripts accordingly as follows, to obtain the corresponding experimental results. 
    - Offline/PBCT_offline.py
    - Online/PBCT_online.py
    - Unlabeled_Offline/PBCT_offline_unlabeled.py
    - Unlabeled_Online/PBCT_online_unlabeled.py
The impact of labeled training data size in the offline scenario can be evaluated through executing Offline/PBCT_offline.py. The impact of labeled training data size in the online scenario can be evaluated through executing Online/PBCT_online.py. The impact of unlabeled training data size in the offline scenario can be evaluated through executing Unlabeled_Offline/PBCT_offline_unlabeled.py. The impact of unlabeled training data size in the online scenario can be evaluated through executing Unlabeled_Online/PBCT_online_unlabeled.py. The results of the baseline methods considered in these scenarios can also be obtained through executing the corresponding scripts.
