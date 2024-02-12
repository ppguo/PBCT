from tkinter import FALSE
from sklearn.model_selection import LeaveOneOut,cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy import mean,absolute,sqrt
import copy
import math
import pandas as pd
import random

def Sequential_Forward_Selection(Features,labeled_X,labled_Y,R):
    Feature_set = []
    Feature_candidates = copy.deepcopy(Features)

    cv = LeaveOneOut()
    model = LinearRegression(fit_intercept=False)
    for i in range(R):
        error_list = []
        for tmp_Feature in Feature_candidates:
            tmp_Feature_set = copy.deepcopy(Feature_set)
            tmp_Feature_set.append(tmp_Feature)
            tmp_X = labeled_X[tmp_Feature_set]
            #print(tmp_X)
            #use LOOCV to evaluate model
            scores = cross_val_score(model, tmp_X, labled_Y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)
            #print(scores)
            error_list.append(mean(absolute(scores)))

        min_index = np.argmin(np.array(error_list))
        #print(min_index)
        if i>0:
            if error_list[min_index]>curr_error:
                return [Feature_set,curr_error]
        curr_error = error_list[min_index]
        Feature_set.append(Feature_candidates[min_index])
        Feature_candidates = Feature_candidates.delete(min_index)
        #print(Feature_candidates)
    return [Feature_set,curr_error]


def Sequential_Forward_Selection_dc(Features,labeled_X,labled_Y,R):
    Feature_set = ['dc']
    Feature_candidates = copy.deepcopy(Features)

    cv = LeaveOneOut()
    model = LinearRegression(fit_intercept=False)
    for i in range(R):
        error_list = []
        for tmp_Feature in Feature_candidates:
            tmp_Feature_set = copy.deepcopy(Feature_set)
            tmp_Feature_set.append(tmp_Feature)
            tmp_X = labeled_X[tmp_Feature_set]
            #print(tmp_X)
            #use LOOCV to evaluate model
            scores = cross_val_score(model, tmp_X, labled_Y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)
            #print(scores)
            error_list.append(mean(absolute(scores)))

        min_index = np.argmin(np.array(error_list))
        #print(min_index)
        if i>0:
            if error_list[min_index]>curr_error:
                return [Feature_set,curr_error]
        curr_error = error_list[min_index]
        Feature_set.append(Feature_candidates[min_index])
        Feature_candidates = Feature_candidates.delete(min_index)
        #print(Feature_candidates)
    return [Feature_set,curr_error]




def Sequential_Forward_Selection_corr(Features,labeled_X,labled_Y,random_index,L,R):
    """
    The input should be original Labeled X, y
    """
    
    
    Feature_set = []
    Feature_candidates = copy.deepcopy(Features)
    
    for i in range(R):
        error_list = []
        for tmp_Feature in Feature_candidates:
            tmp_Feature_set = copy.deepcopy(Feature_set)
            tmp_Feature_set.append(tmp_Feature)
            tmp_X_all = labeled_X[tmp_Feature_set]
            #print(tmp_X)
            #use LOOCV to evaluate model
            LOO_list = []
            for j in range(L):
                model = LinearRegression(fit_intercept=False)
                predict_x = tmp_X_all.iloc[j]
                tmp_X = tmp_X_all.drop(random_index[j])
                predict_y = labled_Y.iloc[j]
                tmp_Y = labled_Y.drop(random_index[j])
                print(tmp_X)
                mean_labled_x = tmp_X.mean()
                print(mean_labled_x)
                std_labled_x = tmp_X.std()
                mean_labled_y = tmp_Y.mean()
                std_labled_y = tmp_Y.std()
                ##Normalize##
                X_train_tmp = (tmp_X - mean_labled_x)/std_labled_x
                print(X_train_tmp)
                y_train_tmp = (tmp_Y-mean_labled_y)/std_labled_y
                predict_x = (predict_x-mean_labled_x)/std_labled_x


                #print(tmp_X.to_numpy())
                #print(tmp_Y.to_numpy())
                model.fit(X_train_tmp.to_numpy(),y_train_tmp.to_numpy())
                #print('alpha',alpha)
                #print('beta',beta)
                #print(predict_x)
                alpha_y = model.predict([predict_x])
                real_alpha_y = (alpha_y[0]*std_labled_y+mean_labled_y).to_numpy()
                print('alpha_y',real_alpha_y[0])

                tmp_error = (predict_y.to_numpy()[0] - real_alpha_y[0])
                #print(tmp_error)
                tmp_error_square = tmp_error * tmp_error
                LOO_list.append(tmp_error_square)
            print(LOO_list)
            error_list.append(np.mean(LOO_list))


        min_index = np.argmin(np.array(error_list))
        #print(min_index)
        if i>0:
            if error_list[min_index]>curr_error:
                return [Feature_set,curr_error]
        curr_error = error_list[min_index]
        Feature_set.append(Feature_candidates[min_index])
        Feature_candidates = Feature_candidates.delete(min_index)
        #print(Feature_candidates)
    return [Feature_set,curr_error]

    
def Sequential_Forward_Selection_corr_test(Features,labeled_X,labled_Y,random_index,L,R):
    """
    The input should be original Labeled X, y
    """
    
    
    Feature_set = []
    Feature_candidates = copy.deepcopy(Features)
    
    for i in range(R):
        error_list = []
        err_nor_list = []
        for tmp_Feature in Feature_candidates:
            tmp_Feature_set = copy.deepcopy(Feature_set)
            tmp_Feature_set.append(tmp_Feature)
            tmp_X_all = labeled_X[tmp_Feature_set]
            #print(tmp_X)
            #use LOOCV to evaluate model
            LOO_list = []
            LOO_nor_list = []
            for j in range(L):
                model = LinearRegression(fit_intercept=False)
                predict_x = tmp_X_all.iloc[j]
                tmp_X = tmp_X_all.drop(random_index[j])
                predict_y = labled_Y.iloc[j]
                tmp_Y = labled_Y.drop(random_index[j])
                print(tmp_X)
                mean_labled_x = tmp_X.mean()
                print(mean_labled_x)
                std_labled_x = tmp_X.std()
                mean_labled_y = tmp_Y.mean()
                std_labled_y = tmp_Y.std()
                ##Normalize##
                X_train_tmp = (tmp_X - mean_labled_x)/std_labled_x
                print(X_train_tmp)
                y_train_tmp = (tmp_Y-mean_labled_y)/std_labled_y
                predict_x = (predict_x-mean_labled_x)/std_labled_x
                predict_y_nor = (predict_y-mean_labled_y)/std_labled_y


                #print(tmp_X.to_numpy())
                #print(tmp_Y.to_numpy())
                model.fit(X_train_tmp.to_numpy(),y_train_tmp.to_numpy())
                #print('alpha',alpha)
                #print('beta',beta)
                #print(predict_x)
                alpha_y = model.predict([predict_x])
                real_alpha_y = (alpha_y[0]*std_labled_y+mean_labled_y).to_numpy()
                print('alpha_y',real_alpha_y[0])

                tmp_error = (predict_y.to_numpy()[0] - real_alpha_y[0])
                tmp_error_nor = alpha_y[0]-predict_y_nor.to_numpy()[0]
                #print(tmp_error)
                tmp_error_square = tmp_error * tmp_error
                tmp_err_nor_square = tmp_error_nor * tmp_error_nor
                LOO_list.append(tmp_error_square)
                LOO_nor_list.append(tmp_err_nor_square)
            print(LOO_list)
            error_list.append(np.mean(LOO_list))
            err_nor_list.append(np.mean(LOO_nor_list))


        min_index = np.argmin(np.array(error_list))
        #print(min_index)
        if i>0:
            if error_list[min_index]>curr_error:
                return [Feature_set,curr_error_nor]
        curr_error = error_list[min_index]
        curr_error_nor = err_nor_list[min_index]
        Feature_set.append(Feature_candidates[min_index])
        Feature_candidates = Feature_candidates.delete(min_index)
        #print(Feature_candidates)
    return [Feature_set,curr_error_nor]

if __name__=="__main__":
    data_samples = pd.read_csv('train_20_feature.csv',index_col=FALSE)
    data_shape = data_samples.shape
    data_columnslable_x = data_samples.columns[:-2]
    data_columnslable_y = data_samples.columns[-1:]
    print(data_columnslable_x)
    print(data_columnslable_y)
    ##split the labeled and unlabeled data
    L = 5
    random_index = [32, 27, 29, 7, 20, 39, 16, 18, 24, 23, 11, 33, 10, 40, 5, 37, 2, 25, 34, 6, 36, 1, 21, 14, 9, 19, 13, 0, 12, 22, 35, 17, 3, 31, 4, 38, 28, 26, 30, 8, 15]
    data_labled = data_samples.loc[random_index[:L]]  ##check why loc is diff from normal slice checked


    data_labled_x = data_labled[data_columnslable_x]
    data_labled_y = data_labled[data_columnslable_y]
    ####
    Partial_Feature = Sequential_Forward_Selection_corr(data_columnslable_x,data_labled_x,data_labled_y,random_index,L,L-2)

    print(Partial_Feature[0],Partial_Feature[1])
