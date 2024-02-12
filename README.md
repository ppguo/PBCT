Requirements:
python==3.9
numpy==1.23.3
pandas==1.4.4
scikit-learn==1.1.2
scipy==1.9.1
gurobi==9.5.2  

The core function of the PBCT algorithm is included in the file utils/PBCT.py. Given the labeled and unlabeled training data as well as the test data, it triggers the training of the complete-view model and parital-view models, save the model parameters in the desired paths, and return the test error measured using RMSE. An example for utilizing the PBCT algorithm is provided in the __main__ section of this file.
