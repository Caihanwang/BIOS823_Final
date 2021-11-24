"""This code is for Tuning the parameters for XGB models for daily cases prediction"""


import pandas as pd
import datetime
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


"""Create Final Data List"""

final_data = pd.read_csv("final_data.csv")

# change the date format:
format = '%Y-%m-%d'
Da = []
for index, row in final_data.iterrows():
    #print(row['c1'], row['c2'])
    Da.append(datetime.datetime.strptime(row["Date"], format).date())
final_data["Date"] = Da

final_data_list = {}

# Create lag varaibles
num = range(7)
for i in num:
    final_data_test = final_data.copy(deep=True)
    j = i + 1
    final_data_test["lag"+str(j+0)+"c"] = final_data_test.groupby("State").Daily_Case.shift(j+0)
    final_data_test["lag"+str(j+1)+"c"] = final_data_test.groupby("State").Daily_Case.shift(j+1)
    final_data_test["lag"+str(j+2)+"c"] = final_data_test.groupby("State").Daily_Case.shift(j+2)
    final_data_test["lag"+str(j+3)+"c"] = final_data_test.groupby("State").Daily_Case.shift(j+3)
    final_data_test["lag"+str(j+4)+"c"] = final_data_test.groupby("State").Daily_Case.shift(j+4)
    final_data_test["lag"+str(j+5)+"c"] = final_data_test.groupby("State").Daily_Case.shift(j+5)
    final_data_test["lag"+str(j+6)+"c"] = final_data_test.groupby("State").Daily_Case.shift(j+6)
    
    final_data_test["lag"+str(j+0)+"d"] = final_data_test.groupby("State").Daily_Death.shift(j+0)
    final_data_test["lag"+str(j+1)+"d"] = final_data_test.groupby("State").Daily_Death.shift(j+1)
    final_data_test["lag"+str(j+2)+"d"] = final_data_test.groupby("State").Daily_Death.shift(j+2)
    final_data_test["lag"+str(j+3)+"d"] = final_data_test.groupby("State").Daily_Death.shift(j+3)
    final_data_test["lag"+str(j+4)+"d"] = final_data_test.groupby("State").Daily_Death.shift(j+4)
    final_data_test["lag"+str(j+5)+"d"] = final_data_test.groupby("State").Daily_Death.shift(j+5)
    final_data_test["lag"+str(j+6)+"d"] = final_data_test.groupby("State").Daily_Death.shift(j+6)
    
    final_data_test["lag"+str(j+0)+"t"] = final_data_test.groupby("State").tests_combined_total.shift(j+0)
    final_data_test["lag"+str(j+1)+"t"] = final_data_test.groupby("State").tests_combined_total.shift(j+1)
    final_data_test["lag"+str(j+2)+"t"] = final_data_test.groupby("State").tests_combined_total.shift(j+2)
    final_data_test["lag"+str(j+3)+"t"] = final_data_test.groupby("State").tests_combined_total.shift(j+3)
    final_data_test["lag"+str(j+4)+"t"] = final_data_test.groupby("State").tests_combined_total.shift(j+4)
    final_data_test["lag"+str(j+5)+"t"] = final_data_test.groupby("State").tests_combined_total.shift(j+5)
    final_data_test["lag"+str(j+6)+"t"] = final_data_test.groupby("State").tests_combined_total.shift(j+6)
    
    final_data_test["lag"+str(j+0)+"v"] = final_data_test.groupby("State").People_Fully_Vaccinated.shift(j+0)
    final_data_test["lag"+str(j+1)+"v"] = final_data_test.groupby("State").People_Fully_Vaccinated.shift(j+1)
    final_data_test["lag"+str(j+2)+"v"] = final_data_test.groupby("State").People_Fully_Vaccinated.shift(j+2)
    final_data_test["lag"+str(j+3)+"v"] = final_data_test.groupby("State").People_Fully_Vaccinated.shift(j+3)
    final_data_test["lag"+str(j+4)+"v"] = final_data_test.groupby("State").People_Fully_Vaccinated.shift(j+4)
    final_data_test["lag"+str(j+5)+"v"] = final_data_test.groupby("State").People_Fully_Vaccinated.shift(j+5)
    final_data_test["lag"+str(j+6)+"v"] = final_data_test.groupby("State").People_Fully_Vaccinated.shift(j+6)
    
    
    final_data_test = final_data_test.dropna().copy(deep = True)
    
    final_data_test['Fully_reopen']        = final_data_test['Fully_reopen'].astype(object)
    final_data_test['Mask_Mandate']        = final_data_test['Mask_Mandate'].astype(object)
    final_data_test['Vaccination_or_test'] = final_data_test['Vaccination_or_test'].astype(object)
    final_data_test['State']               = final_data_test['State'].astype(object)
    final_data_test['Region']              = final_data_test['Region'].astype(object)
    final_data_test['Division']            = final_data_test['Division'].astype(object)
    
    final_data_test = final_data_test.drop(columns=['People_Fully_Vaccinated', 'tests_combined_total']).copy(deep = True)

    final_data_list["final_data_"+str(j)] = final_data_test


# Hold out data after 2021-11-15 for validation
sep = [i for i in final_data_list]
thresh = datetime.date(2021, 11, 15)
predict_data_list = {}
for i in sep:
    predict_data_list[i] = final_data_list[i][final_data_list[i]['Date'] >= thresh]
    final_data_list[i] = final_data_list[i][final_data_list[i]['Date'] < thresh]   
    

"""Parameters Tuning For XGBoost of Daily Case"""

params_list = []

num_boost_round = 999

for i in final_data_list:
    
    # Set default parameters
    params = {
        # Parameters that we are going to tune.
        'max_depth':6,
        'min_child_weight': 1,
        'eta':.3,
        'subsample': 1,
        'colsample_bytree': 1,
        # Other parameters
        'objective':'reg:linear',
        'eval_metric': 'mae'
    }    
    
    # read in data and preprocessing
    data = final_data_list[i].drop(["Date","State","state"], 1)
    data = pd.get_dummies(data)
    
    # train test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    y_train = train["Daily_Case"]
    X_train = train.drop(["Daily_Case","Daily_Death"],1)
    y_test = test["Daily_Case"]
    X_test = test.drop(["Daily_Case","Daily_Death"],1)
    
    # xgb.DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    """Tune max_depth & min_child_weight"""
    gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(0,10)
    for min_child_weight in range(0,8)]

    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
    #     print("CV with max_depth={}, min_child_weight={}".format(
    #                              max_depth,
    #                              min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        #print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth,min_child_weight)
    #print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

    params['max_depth'] = best_params[0]
    params['min_child_weight'] = best_params[1]    

    """Tune subsample & colsample"""
    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(3,7)]
        for colsample in [i/10. for i in range(3,7)]
    ]    
    
    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
#         print("CV with subsample={}, colsample={}".format(
#                                  subsample,
#                                  colsample))
        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        #print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample,colsample)
    params['subsample'] = best_params[0]
    params['colsample_bytree'] = best_params[1]     
    #print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))    
    
    """Tune ETA"""
    min_mae = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        #print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=5,
                metrics=['mae'],
                early_stopping_rounds=10)
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        #print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta
    #print("Best params: {}, MAE: {}".format(best_params, min_mae))
    params['eta'] = best_params
    
    params_list.append(params)
    
    
print(params_list)    



"""
Parameters Tuning Result

[
    {'max_depth': 4, 
    'min_child_weight': 2, 
    'eta': 0.05, 
    'subsample': 0.6, 
    'colsample_bytree': 0.6, 
    'objective': 'reg:linear', 
    'eval_metric': 'mae'}, 
    
    {'max_depth': 5, 
    'min_child_weight': 2, 
    'eta': 0.05, 
    'subsample': 0.6, 
    'colsample_bytree': 0.4, 
    'objective': 'reg:linear', 
    'eval_metric': 'mae'}, 

    {'max_depth': 6, 
    'min_child_weight': 1, 
    'eta': 0.05, 
    'subsample': 0.6, 
    'colsample_bytree': 0.6, 
    'objective': 'reg:linear', 
    'eval_metric': 'mae'}, 
    
    {'max_depth': 6, 
    'min_child_weight': 3, 
    'eta': 0.1, 
    'subsample': 0.6, 
    'colsample_bytree': 0.6, 
    'objective': 'reg:linear', 
    'eval_metric': 'mae'}, 
    
    {'max_depth': 5, 
    'min_child_weight': 1, 
    'eta': 0.05, 
    'subsample': 0.6, 
    'colsample_bytree': 0.6, 
    'objective': 'reg:linear', 
    'eval_metric': 'mae'}, 
    
    {'max_depth': 5, 
    'min_child_weight': 2, 
    'eta': 0.05, 
    'subsample': 0.6, 
    'colsample_bytree': 0.4, 
    'objective': 'reg:linear', 
    'eval_metric': 'mae'}, 
    
    {'max_depth': 6, 
    'min_child_weight': 1, 
    'eta': 0.05, 
    'subsample': 0.6, 
    'colsample_bytree': 0.6, 
    'objective': 'reg:linear', 
    'eval_metric': 'mae'}
    
    ]




"""
