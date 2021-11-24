# COVID-19 Forecasting

![image.png](https://i.loli.net/2021/11/16/i2OmKlFyEXrpIsW.png)

**Team Name**: Nintendo

**Team Members**: Caihan Wang, Yifeng Tang, Yuxuan Chen  

## Dashboard
This is our final dashboard. Please feel free to explore it. **[link](https://covid19-project-823.herokuapp.com/)**  

## Data Source  
1. COVID-19 Trend Data: https://covid.cdc.gov/covid-data-tracker/#datatracker-home
2. Vaccination Data: https://github.com/govex/COVID-19/tree/master/data_tables/vaccine_data/us_data/time_series
3. State Characteristic Data: https://datacommons.org/; https://www.openicpsr.org/openicpsr/project/119446/version/V129/view?path=/openicpsr/119446/fcr:versions/V129/COVID-19-US-State-Policy-Database-master/COVID-19-US-state-policy-database-9_20_2021.xlsx&type=file  

## Objective
1. Train Random Forest and XGBoost models to forecast COVID-19 trend in following 7 days.
2. Test and Evaluate the model, choose the best one to predict the following 7 days.  
3. Deploy the model into a dashboard.

## Data Science Plan
**1. Data Plan**  
The initial data we had was COVID-19 daily cases and deaths by state from 2021/2/10 to 2021/11/14, vaccination data and state characteristic data. We merge them together and clean the data by creating lag variables and split it to 7 datasets. We will utilize these 7 datasets to train and test 7 models for daily cases prediction and 7 models for daily deaths prediction.  

**2. Machine Learning Plan**  
To predict daily case and daily death, based on literature review, we decide to fit Random Forest models and XGBoost models by 7 datasets mentioned above.  
* Parameter Tuning: For each model, we need to tune parameters to reduce error and reach optimal prediction on test dataset. So we need to tune the parameters by comparing RMSE
* Model Fitting: We will fit models using the parameter tuning results.  
* Model Evaluation: After model fiting, we choose to evaluate the models by calculating RMSE of models on test dataset. 
* Model Validation: We can choose the best models (Random Forest models or XGBoost models) to do our final prediction from 11/15 to 11/21 and compare our prediction values with true number of cases/deaths on CDC website.  

**3. Operations Plan (8~10 Days)**  
* Data Merging (1 day)
* Exploratory Data Analysis (1 day)
* Data Cleaning and Encoding (1 day)
* Model Parameter Tuning/ Fitting/ Evaluation (2~3 days)
* Model Validation (1 day)
* Dashboard Deployment (1~2 days)
* Fnial Report (1 day)


**4. Technology Stack**  
* Database: Data cleaning product will be stored in github page and jupyter notebookes
* Python Packages: [requirement.txt](https://github.com/Caihanwang/BIOS823_Final/blob/main/requirement.txt)
* Cloud Platform: deploy our dashboard by Heroku



## Roles, responsibilities and timed milestones
**Caihan Wang:**  
* Data download, merging and cleaning
* XGB model parameter tuning
* XGB model fitting
* XGB model Evaluation
* XGB model Validation
* Github organize
* Final report edit

**Yifeng Tang:**  
* Data download, merging and cleaning
* XGB model parameter tuning
* XGB model fitting
* XGB model Evaluation
* Dashboard construct
* Dashboard deployment
* Final report edit


**Yuxuan Chen:**  
* Data download, merging and cleaning
* Exploratory data analysis
* RF model parameter tuning
* RF model fitting
* RF model Evaluation
* RF model Validation
* Final report edit


**Timeline**:  
*due: 11/24/2021*

* 11/16/2021 Data download, merging and cleaning, EDA
* 11/20/2021 Modeling
* 11/22/2021 Dashboard and Final report


