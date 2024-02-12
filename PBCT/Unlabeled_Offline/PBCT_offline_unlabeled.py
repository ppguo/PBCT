from tkinter import FALSE
import pandas as pd
from SFS import Sequential_Forward_Selection_corr_test
from sklearn.linear_model import LinearRegression
from solve_loss import solve_loss
#from solve_loss import solve_loss
from sklearn.model_selection import LeaveOneOut,cross_val_score
import numpy as np
import math
import sklearn
import random
import matplotlib.pyplot as plt


def PBCT_log(csv_name,L,U,test_num,Upper_var1,random_index,repeated_num,coef_file):
    data_samples = pd.read_csv(csv_name+'.csv',index_col=FALSE)
    data_shape = data_samples.shape
    data_columnslable_x = data_samples.columns[:-2]
    data_columnslable_y = data_samples.columns[-1:]
    #print(data_columnslable_x)
    #print(data_columnslable_y)
    ##split the labeled and unlabeled data
    #labeled_ratio = 0.10
    #L = int(data_shape[0]*labeled_ratio)
    #L = 12
    #U = 20 
    #random_index = random.sample(range(120),L+U)
    print('labeled_num',L)
    data_labled = data_samples.loc[random_index[:L]]  ##check why loc is diff from normal slice checked
    #print(data_labled)
    data_unlabled = data_samples.loc[random_index[-test_num:-test_num+U]]
    data_test = data_samples.loc[random_index[-test_num:]]
    #print(data_unlabled)
    print('unlabled_num',U)
    #print(data_labled)
    #print(data_unlabled)

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
    #print("std_y",std_labled_y)

    X_train_labled = (data_labled_x-mean_labled_x)/std_labled_x
    #print(X_train_labled)
    X_train_unlabled = (data_unlabled_x-mean_labled_x)/std_labled_x

    X_test= (data_test_x-mean_labled_x)/std_labled_x
    y_train = (data_labled_y-mean_labled_y)/std_labled_y
    y_test = (data_test_y-mean_labled_y)/std_labled_y
    #################

    ####build Partial model####
    Partial_feature_var2 = Sequential_Forward_Selection_corr_test(data_columnslable_x,data_labled_x,data_labled_y,random_index,L,L-2)
    Partial_feature = Partial_feature_var2[0]
    print('var features for'+str(L)+' '+'repeated_num '+str(repeated_num)+'is', Partial_feature)
 

    ZL = data_labled_x[Partial_feature]
    ZU = data_unlabled_x[Partial_feature]
    var2 = Partial_feature_var2[1]
    print('var2 is ',var2)
    #var2 = max(1e-8,var2)
    #print('var2 is ',var2)
    ########################
    cv = LeaveOneOut()

    #######this part should be checked#####
    lasso_model = sklearn.linear_model.LassoCV(fit_intercept=False,cv=cv,alphas=[0.01,0.1,1,10])
    Reg_lasso = lasso_model.fit(X_train_labled.to_numpy(), y_train.to_numpy().ravel())
    print('alpha of lasso is',Reg_lasso.alpha_)

    ####find the var1####
    V = [0.5,1,2,5,10]
    print('current V',V)
    var1_candidate_set = [Upper_var1*i*var2 for i in V] ##
    l5_candidate = [10,100]
    #V = 5
    #var1_candidate_set = [math.pow(0.1,V)*0.05*var2 for i in range(V)] 
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
                #print('real_alpha_y',real_alpha_y[0])

                tmp_error = (real_predict_y[0] - real_alpha_y[0])
                tmp_error_square = tmp_error * tmp_error
                error_list.append(tmp_error_square)
            LOO_list.append(np.mean(error_list))
    #print(LOO_list)
    var1_index = np.argmin(np.array(LOO_list))
    print(var1_index)
    l1_index = var1_index // len(l5_candidate)
    print(l1_index)
    l5_index = var1_index % len(l5_candidate)
    print(l5_index) 
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
    #print('Loss',math.pow(np.linalg.norm(np.matmul(X_train_unlabled.to_numpy(),alpha)-np.matmul(ZU.to_numpy(),beta)),2))
    #####################################
    z_test_all = X_test[Partial_feature]

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
        tmp_pre = np.matmul(beta.T,z_test_all.iloc[i].to_numpy())*std_labled_y+mean_labled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test0_err_list.append(tmp_error_square)
        err0_percent_list.append(tmp_percent_err)

    print('PBCT_beta_RMSE',np.sqrt(np.mean(test0_err_list)))
    print('PBCT_beta_ERR',np.mean(err0_percent_list))
    PBCT_beta_RMSE = np.sqrt(np.mean(test0_err_list))
    PBCT_beta_ERR = np.mean(err0_percent_list)

    ###LinearRegreesion normalized manually####
    model = LinearRegression(fit_intercept=False)
    
    #print('Z_train_shape',ZL_all.to_numpy().shape)
    Reg = model.fit(ZL_all.to_numpy(), y_train.to_numpy())
    test1_err_list = []
    err1_percent_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = Reg.predict([z_test_all.iloc[i].to_numpy()])[0]*std_labled_y+mean_labled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test1_err_list.append(tmp_error_square)
        err1_percent_list.append(tmp_percent_err)
    print('LS_beta_RMSE',np.sqrt(np.mean(test1_err_list)))
    print('LS_beta_ERR',np.mean(err1_percent_list))

    LS_beta_RMSE = np.sqrt(np.mean(test1_err_list))
    LS_beta_ERR = np.mean(err1_percent_list)





    #lasso_model = sklearn.linear_model.LassoCV(fit_intercept=False,cv=cv,tol=1e-2)

    #Reg_lasso = lasso_model.fit(X_train_labled.to_numpy(), y_train.to_numpy().ravel())
    #print('alpha of lasso is',Reg_lasso.alpha_)
    test2_err_list = []
    err2_percent_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = Reg_lasso.predict([X_test.iloc[i].to_numpy()])[0]*std_labled_y+mean_labled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test2_err_list.append(tmp_error_square)
        err2_percent_list.append(tmp_percent_err)
    print('LS_lasso_RMSE',np.sqrt(np.mean(test2_err_list)))
    print('LS_lasso_ERR',np.mean(err2_percent_list))
    LS_lasso_RMSE = np.sqrt(np.mean(test2_err_list))
    LS_lasso_ERR = np.mean(err2_percent_list)
    l1_ratio_list =[0.05,0.2,0.4,0.6,0.8,0.95]
    elasticNet_model = sklearn.linear_model.ElasticNetCV(fit_intercept=False,cv=cv,alphas=[0.01,0.1,1,10],l1_ratio = l1_ratio_list)
    Reg_elasticnet = elasticNet_model.fit(X_train_labled.to_numpy(), y_train.to_numpy().ravel())
    print('alpha and l1_ratio of elasticNet is',Reg_elasticnet.alpha_,Reg_elasticnet.l1_ratio_)

    test3_err_list = []
    err3_percent_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = Reg_elasticnet.predict([X_test.iloc[i].to_numpy()])[0]*std_labled_y+mean_labled_y
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test3_err_list.append(tmp_error_square)
        err3_percent_list.append(tmp_percent_err)
    print('LS_elasnet_RMSE',np.sqrt(np.mean(test3_err_list)))
    print('LS_elasnet_ERR',np.mean(err3_percent_list))
    LS_elasnet_RMSE = np.sqrt(np.mean(test3_err_list))
    LS_elasnet_ERR = np.mean(err3_percent_list)


    coef_list = []
    coef_list.append(Reg.coef_[0])
    coef_list.append(np.array([v for v in Reg_lasso.coef_]))
    coef_list.append(np.array([v for v in Reg_elasticnet.coef_]))
    coef_list.append(np.array([v for v in alpha.ravel()]))
    coef_list.append(np.array(beta.ravel()))
    tmp_df = pd.DataFrame(data = coef_list)
    tmp_df.to_csv(coef_file+str(U)+'_'+str(repeated_num)+'.csv',index = False, header = False)
    return PBCT_RMSE,  PBCT_beta_RMSE,LS_beta_RMSE,LS_lasso_RMSE,LS_elasnet_RMSE

if __name__=="__main__":
    repeated_num = 100
    coef_file = './tmp_coef/'
    all_average = {}
    all_median = {}
    csv_name = '../Data/pri_20_feature'
    file_name =  'tmp_file'
    test_num = 25
    Upper_var1 = 1
    random_index = [random.sample(range(43),43) for i in range(repeated_num)]
    np.save('pri_r1_10.npy',random_index)
    sweep_num = 10
    print(random_index)
    for unlabeled_num in [0,5,10,15,20,25]:
        tmp_dict ={}
        all_list  = []
        for i in range(repeated_num):
            print('the ',i+1,'-th result')
            PBCT_RMSE, PBCT_beta_RMSE,LS_beta_RMSE,LS_lasso_RMSE,LS_elasnet_RMSE = PBCT_log(csv_name,sweep_num,unlabeled_num,test_num,Upper_var1,random_index[i],i,coef_file)
            tmp_dict[i] = [PBCT_RMSE, PBCT_beta_RMSE,LS_beta_RMSE,LS_lasso_RMSE,LS_elasnet_RMSE]
            all_list.append(tmp_dict[i])
        average_array = np.mean(np.array(all_list),axis = 0)
        median_array = np.median(np.array(all_list),axis = 0)
        tmp_dict['average'] = average_array
        tmp_dict['median'] = median_array
        all_average[unlabeled_num] = average_array
        all_median[unlabeled_num] = median_array
        df = pd.DataFrame(tmp_dict)
        df.to_csv('./'+file_name+'/'+csv_name+'_result_PBCT_'+str(sweep_num)+'_'+str(unlabeled_num)+'_offline.csv',index=False)
    avg_all = pd.DataFrame(all_average)
    median_all = pd.DataFrame(all_median)
    avg_all.to_csv('./'+file_name+'/offline_avg_'+str(Upper_var1)+'.csv',index = False)
    median_all.to_csv('./'+file_name+'/offline_median_'+str(Upper_var1)+'.csv',index = False)







