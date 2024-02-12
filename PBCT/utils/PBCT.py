from tkinter import FALSE
import pandas as pd
from SFS import Sequential_Forward_Selection_corr_test
from sklearn.linear_model import LinearRegression
from solve_loss import solve_loss
from sklearn.model_selection import LeaveOneOut,cross_val_score
import numpy as np
import math
import sklearn
import random
import matplotlib.pyplot as plt


def PBCT(L,random_index,repeated_num,coef_file,data_labled,data_unlabled,data_test,data_columnslable_x,data_columnslable_y):
    """
    L:number of labeled data
    repeat_num: random experiment index (for repeated experiment)
    random_index: list of current index sample
    coef_file: file to store the coef of learned model parameter
    data_labeled: labeled data (feature and cyclelife)
    data_unlabeled: unlabeled data (feature and cyclelife(ignored))
    data_test: test data (feature and cyclelife)
    data_columnslable_x: column names of feature
    data_columnslable_y: column names of cyclelife

    """

    data_labled_x = data_labled[data_columnslable_x]
    data_labled_y = data_labled[data_columnslable_y]
    data_unlabled_x = data_unlabled[data_columnslable_x]
    data_unlabled_y = data_unlabled[data_columnslable_y]
    data_test_x = data_test[data_columnslable_x]
    data_test_y = data_test[data_columnslable_y]
    ####Nomalize####
    mean_labled_x = data_labled_x.mean()
    std_labled_x = data_labled_x.std()

    mean_labled_y = data_labled_y.mean()
    std_labled_y = data_labled_y.std()

    X_train_labled = (data_labled_x-mean_labled_x)/std_labled_x
    X_train_unlabled = (data_unlabled_x-mean_labled_x)/std_labled_x
    y_train = (data_labled_y-mean_labled_y)/std_labled_y
    X_test = (data_test_x-mean_labled_x)/std_labled_x
    y_test = (data_test_y-mean_labled_y)/std_labled_y
    #################

    ####build Partial model####
    Partial_feature_var2 = Sequential_Forward_Selection_corr_test(data_columnslable_x,data_labled_x,data_labled_y,random_index,L,L-2)
    Partial_feature = Partial_feature_var2[0]
    print(Partial_feature)


    ZL = data_labled_x[Partial_feature]
    ZU = data_unlabled_x[Partial_feature]
    Z_test = X_test[Partial_feature]
    #var2 = max(Partial_feature_var2[1],1e-3)
    var2 = Partial_feature_var2[1]
    print('var2 is ',var2)



    ####find the var1####
    V = [0.5,1,2,5,10]
    print('current V',V)
    var1_candidate_set = [i*var2 for i in V]

    l5_candidate = [10,100]

    LOO_list = []

    for i in range(len(V)):
        for m in range(len(l5_candidate)):
            tmp_var1 = var1_candidate_set[i]
            l1 = 1/2/tmp_var1
            l2 = 1/2/var2
            l3 = 0
            l4 = 1/2/(tmp_var1+var2)
            #l5 = 0
            l5 = l1*l5_candidate[m]
            error_list = []
            for j in range(L):
                predict_x = data_labled_x.iloc[j]
                tmp_X = data_labled_x.drop(random_index[j])
                tmp_ZL = ZL.drop(random_index[j])
                predict_y = data_labled_y.iloc[j]
                tmp_y = data_labled_y.drop(random_index[j])

                ##Normalize##
                mean_labled_x = tmp_X.mean()
                std_labled_x = tmp_X.std()
                mean_labled_y = tmp_y.mean()
                std_labled_y = tmp_y.std()
                mean_labled_z = tmp_ZL.mean()
                std_labled_z = tmp_ZL.std()

                X_train_tmp = (tmp_X - mean_labled_x)/std_labled_x
                y_train_tmp = (tmp_y-mean_labled_y)/std_labled_y
                X_train_unlabled_tmp = (data_unlabled_x-mean_labled_x)/std_labled_x

                tmp_ZL = (tmp_ZL-mean_labled_z)/std_labled_z
                tmp_ZU = (ZU-mean_labled_z)/std_labled_z


                predict_x = (predict_x-mean_labled_x)/std_labled_x


                alpha,beta = solve_loss(y_train_tmp.to_numpy().ravel(),X_train_tmp.to_numpy(),tmp_ZL.to_numpy(),
                        X_train_unlabled_tmp.to_numpy(),tmp_ZU.to_numpy(),l1,l2,l3,l4,l5)

                real_predict_y = predict_y.to_numpy()
                alpha_y = np.matmul(alpha.T,predict_x.to_numpy())
                real_alpha_y = (alpha_y*std_labled_y+mean_labled_y).to_numpy()

                tmp_error = (real_predict_y[0] - real_alpha_y[0])
                tmp_error_square = tmp_error * tmp_error
                error_list.append(tmp_error_square)
            LOO_list.append(np.mean(error_list))
    var1_index = np.argmin(np.array(LOO_list))
    l1_index = var1_index // len(l5_candidate)
    l5_index = var1_index % len(l5_candidate)
    #######################
    print('var1_index',var1_index)
    ####get the optimal alpha and beta####
    l1 = 1/2/var1_candidate_set[l1_index]
    l2 = 1/2/var2
    l3 = 0
    l4 = 1/2/(var1_candidate_set[l1_index]+var2)
    l5 = l1*l5_candidate[l5_index]

    ZL_all = X_train_labled[Partial_feature]
    ZU_all = X_train_unlabled[Partial_feature]



    alpha,beta = solve_loss(y_train.to_numpy().ravel(),X_train_labled.to_numpy(),ZL_all.to_numpy(),
                        X_train_unlabled.to_numpy(),ZU_all.to_numpy(),l1,l2,l3,l4,l5)
    print('alpha is ',alpha)
    print('beta is',beta)
    Loss = (l1*math.pow(np.linalg.norm(y_train.to_numpy()-np.matmul(X_train_labled.to_numpy(),alpha)),2)+
            l2*math.pow(np.linalg.norm(y_train.to_numpy()-np.matmul(ZL.to_numpy(),beta)),2)+
            l3*math.pow(np.linalg.norm(np.matmul(X_train_labled.to_numpy(),alpha)-np.matmul(ZL.to_numpy(),beta)),2)+
            l4*math.pow(np.linalg.norm(np.matmul(X_train_unlabled.to_numpy(),alpha)-np.matmul(ZU.to_numpy(),beta)),2))
    #check_loss(y_train,X_train_labled,ZL,
    #                    X_train_unlabled,ZU,l1,l2,l3,l4,alpha,beta,Loss)
    print('Loss',math.pow(np.linalg.norm(np.matmul(X_train_unlabled.to_numpy(),alpha)-np.matmul(ZU.to_numpy(),beta)),2))
    #####################################


    ####start to test######
    test_err_list = []
    err_percent_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = np.matmul(alpha.T,X_test.iloc[i].to_numpy())*std_labled_y+mean_labled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test_err_list.append(tmp_error_square)
        err_percent_list.append(tmp_percent_err)

    print('PBCT_RMSE',np.sqrt(np.mean(test_err_list)))
    print('PBCT_ERR',np.mean(err_percent_list))
    
    PBCT_RMSE = np.sqrt(np.mean(test_err_list))
    PBCT_ERR = np.mean(err_percent_list)
    #print(LOO_list)
    test0_err_list = []
    err0_percent_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = np.matmul(beta.T,Z_test.iloc[i].to_numpy())*std_labled_y+mean_labled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test0_err_list.append(tmp_error_square)
        err0_percent_list.append(tmp_percent_err)

    print('PBCT_beta_RMSE',np.sqrt(np.mean(test0_err_list)))
    print('PBCT_beta_ERR',np.mean(err0_percent_list))
    PBCT_beta_RMSE = np.sqrt(np.mean(test0_err_list))
    PBCT_beta_ERR = np.mean(err0_percent_list)


    coef_list = []
    coef_list.append(np.array([v for v in alpha.ravel()]))
    coef_list.append(np.array(beta.ravel()))
    tmp_df = pd.DataFrame(data = coef_list)
    tmp_df.to_csv(coef_file+str(L)+'_'+str(repeated_num)+'.csv',index = False, header = False)
    return PBCT_RMSE, PBCT_beta_RMSE


if __name__=="__main__":
    coef_file = './tmp_coef/'
    repeated_num = 50
    all_average = {}
    all_median = {}
    csv_name = '../Data/pri_20_feature'
    file_name =  'tmp_file'
    Upper_var1 = 1
    data_num = 43
    labeled_num = 10
    unlabeled_num = 10 
    test_num = 10
    random_index = [random.sample(range(43),43) for i in range(repeated_num)]
    repeated_num = 0

    data_samples = pd.read_csv(csv_name+'.csv',index_col=FALSE)
    data_columnslable_x = data_samples.columns[:-2]
    data_columnslable_y = data_samples.columns[-1:]
    train_index = random_index[repeated_num][:data_num-test_num]
    test_index = random_index[repeated_num][data_num-test_num:]
    data_labled = data_samples.loc[train_index[:labeled_num]]  
    data_unlabled = data_samples.loc[train_index[-unlabeled_num:]]
    data_test = data_samples.loc[test_index]


    PBCT_RMSE, PBCT_beta_RMSE= PBCT(labeled_num, random_index[repeated_num],repeated_num,coef_file,data_labled,data_unlabled,data_test,data_columnslable_x,data_columnslable_y)
    











