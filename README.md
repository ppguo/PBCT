The core function of the PBCT algorithm is included in the file utils/PBCT.py. Given the labeled and unlabeled training data as well as the test data, it triggers the training of the complete-view model and parital-view models, save the model parameters in the desired paths, and return the test error measured using RMSE. An example for utilizing the PBCT algorithm is provided in the __main__ section of this file.

The source dataset under /data comes from

Severson, K.A., Attia, P.M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M.H., Aykol, M., Herring, P.K., Fraggedakis, D., et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nat. Energy 4, 383–391

The original data can be found in [link](https://data.matr.io/1/) under the license of [CC-BY](https://creativecommons.org/licenses/by/4.0/).
We extract the features according to the instruction from paper.
