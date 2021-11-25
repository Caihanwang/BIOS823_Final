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

num = range(7)

#please refere our final data here:
final_data = pd.read_csv("final_data.csv")

# change the date format:
format = '%Y-%m-%d'
Da = []
for index, row in final_data.iterrows():
    #print(row['c1'], row['c2'])
    Da.append(datetime.datetime.strptime(row["Date"], format).date())
final_data["Date"] = Da

final_data_list = {}
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
    
    # seperate the data:
sep = ["final_data_1", "final_data_2", "final_data_3", "final_data_4", "final_data_5", "final_data_6", "final_data_7"]
thresh = datetime.date(2021, 11, 15)
predict_data = {}
for i in sep:
    predict_data[i] = final_data_list[i][final_data_list[i]['Date'] >= thresh]
    final_data_list[i] = final_data_list[i][final_data_list[i]['Date'] < thresh]

data_list = []
ran = list(final_data_list.keys())
for i in ran:
    data = final_data_list[i]
    data = data.drop(["Date", "State", "state"], 1)
    data = pd.get_dummies(data)
    data_list.append(data)
len(data_list)

train_and_test = []
for i in data_list:
    train, test = train_test_split(i, test_size=0.2, random_state=42)
    item = [train, test]
    train_and_test.append(item)


#seperate train and test
y_train = []
X_train = []
y_test = []
X_test = []
for i in train_and_test:
    y_train_item = i[0]["Daily_Death"]
    y_train.append(y_train_item)
    
    X_train_item = i[0].drop(["Daily_Case","Daily_Death"], 1)
    X_train.append(X_train_item)
    
    y_test_item = i[1]["Daily_Death"]
    y_test.append(y_test_item)
    
    X_test_item = i[1].drop(["Daily_Case","Daily_Death"], 1)
    X_test.append(X_test_item)

#concert to dtrain data:
dtrain_list = []
dtest_list = []

for i in range(0, len(y_test)):
    dtrain_item = xgb.DMatrix(X_train[i], label=y_train[i])
    dtrain_list.append(dtrain_item)
    
    dtest_item = xgb.DMatrix(X_test[i], label=y_test[i])
    dtest_list.append(dtest_item)

# calculate the base line (this part is not neccessary)
import numpy as np
mae_baseline = []
for i in range(0, len(y_train)):
    mean_train = np.mean(y_train[i])
    baseline_predictions = np.ones(y_test[i].shape) * mean_train
    mae_baseline_item = mean_absolute_error(y_test[i], baseline_predictions)
    mae_baseline.append(mae_baseline_item)

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
}

# tune the max_depth and min_child_weight:

max_depth_list=[]
min_child_weight_list = []

for i in range(0, 7):
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(0,10)
        for min_child_weight in range(0,8)]
    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
        
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain_list[i],
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10)
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        #print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth,min_child_weight)
    max_depth_list.append(best_params[0])
    min_child_weight_list.append(best_params[1])


#tune the subsample and colsample:
subsample_list = []
colsample_list = []

for i in range(0,7):
    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(7,11)]
        for colsample in [i/10. for i in range(7,11)]]

    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain_list[i],
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10)
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        #print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample,colsample)
    subsample_list.append(best_params[0])
    colsample_list.append(best_params[1])

    # tune the eta:

    eta_list = []
for i in range(0,7):
    min_mae = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(params,
                            dtrain_list[i],
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
    eta_list.append(best_params)

#combine the tunning results:

params_list = []

for i in range(0,7):
    params = {'colsample_bytree': colsample_list[i], 
              'eta': eta_list[i],
              'eval_metric': 'mae',
              'max_depth': max_depth_list[i],
              'min_child_weight': min_child_weight_list[i],
              'objective': 'reg:linear',
              'subsample': subsample_list[i]}
    params_list.append(params)

#result is the following:
# [{'colsample_bytree': 1.0,
#   'eta': 0.01,
#   'eval_metric': 'mae',
#   'max_depth': 8,
#   'min_child_weight': 7,
#   'objective': 'reg:linear',
#   'subsample': 1.0},
#  {'colsample_bytree': 1.0,
#   'eta': 0.005,
#   'eval_metric': 'mae',
#   'max_depth': 8,
#   'min_child_weight': 7,
#   'objective': 'reg:linear',
#   'subsample': 0.8},
#  {'colsample_bytree': 1.0,
#   'eta': 0.01,
#   'eval_metric': 'mae',
#   'max_depth': 7,
#   'min_child_weight': 7,
#   'objective': 'reg:linear',
#   'subsample': 1.0},
#  {'colsample_bytree': 0.9,
#   'eta': 0.005,
#   'eval_metric': 'mae',
#   'max_depth': 8,
#   'min_child_weight': 3,
#   'objective': 'reg:linear',
#   'subsample': 0.9},
#  {'colsample_bytree': 1.0,
#   'eta': 0.005,
#   'eval_metric': 'mae',
#   'max_depth': 8,
#   'min_child_weight': 4,
#   'objective': 'reg:linear',
#   'subsample': 1.0},
#  {'colsample_bytree': 1.0,
#   'eta': 0.005,
#   'eval_metric': 'mae',
#   'max_depth': 9,
#   'min_child_weight': 3,
#   'objective': 'reg:linear',
#   'subsample': 1.0},
#  {'colsample_bytree': 1.0,
#   'eta': 0.005,
#   'eval_metric': 'mae',
#   'max_depth': 5,
#   'min_child_weight': 2,
#   'objective': 'reg:linear',
#   'subsample': 0.9}]
