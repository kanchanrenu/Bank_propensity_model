# Step 1 - load data - output of this step is a dataframe
raw_data=pd.read_csv('bank-additional-full.csv')
print('Step 1')
print(raw_data)

# Step 2 - create a dataframe with processed features - output of this step is a dataframe
processed_data= raw_data['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']

# drop null/ missing values, clean the data up to prepare for modelling
unknown_col_list=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
for column in unknown_col_list:
    processed_data=processed_data[processed_data[column]!='unknown']

##2.a) process age (end with a dataframe)
def age(data):
  data.loc[data['age'] <= 20, 'age'] = 1
  data.loc[(data['age'] > 20) & (data['age'] <= 25)  , 'age']    = 2
  data.loc[(data['age'] > 25) & (data['age'] <= 30)  , 'age']   = 3
  data.loc[(data['age'] > 30) & (data['age'] <= 35) , 'age'] = 4
  data.loc[(data['age'] > 35) & (data['age'] <= 40), 'age'] = 5
  data.loc[(data['age'] > 40) & (data['age'] <= 45), 'age'] = 6
  data.loc[(data['age'] > 45) & (data['age'] <= 50), 'age'] = 7
  data.loc[(data['age'] > 50) & (data['age'] <= 55), 'age'] = 8
  data.loc[(data['age'] > 55) & (data['age'] <= 60)  , 'age']    = 9
  data.loc[(data['age'] > 60) & (data['age'] <= 65)  , 'age']    = 10
  data.loc[(data['age'] > 65) & (data['age'] <= 70)  , 'age']    = 11
  data.loc[(data['age'] > 70) & (data['age'] <= 75)  , 'age']    = 12
  data.loc[(data['age'] > 75) & (data['age'] <= 80)  , 'age']    = 13
  data.loc[(data['age'] > 80) & (data['age'] <= 85)  , 'age']    = 14
  data.loc[(data['age'] > 85) & (data['age'] <= 90)  , 'age']    = 15
  data.loc[data['age']  > 90, 'age'] = 16
  return data
age(processed_data);
##2.b) process job,default,marital,housing,loan (end with a dataframe)
processed_data = pd.get_dummies(data=processed_data, prefix=['job','default','marital','housing','loan','contact',], columns=['job','default','marital','housing','loan'], drop_first=True)

##2.c) process education (end with a dataframe)
def education(data):
 data.loc[(data['education'] == 'basic.4y') , 'education'] = 1
 data.loc[(data['education'] == 'basic.6y') , 'education'] = 2
 data.loc[(data['education'] == 'basic.9y') , 'education'] = 3
 data.loc[(data['education'] == 'high.school') , 'education'] = 4
 data.loc[(data['education'] == 'professional.course') , 'education'] = 5
 data.loc[(data['education'] == 'university.degree') , 'education'] = 6
 data = data[~data.education.isin(['unknown','illiterate'])]
 return data
education(processed_data);

##2.d) process month (end with a dataframe)
months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

processed_data['month'] = processed_data['month'].apply(lambda x: months[x])

##2.e) process day of week (end with a dataframe)
days = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}

processed_data['day_of_week'] = processed_data['day_of_week'].apply(lambda x: days[x])

##2.f) drop pdays,poutcome,previous,emp.var.rate,cons.price.idx,cons.conf.idx,euribor3m,nr.employed,duration(end with a dataframe)

processed_data=processed_data.drop(columns =['pdays','poutcome','previous','emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed','duration'])

# 2.g) process 'y' column
processed_data=processed_data['y'].replace(['no', 'yes'], [0,1], inplace=True)
## 2.h) Save precessed_data as a clean data .csv
bank_model=processed_data.to_csv('clean_data.csv',index=False)
print('step 2')
print(bank_model)

#step 3a add index colum as id
bank_model.index = [x for x in range(1, len(bank_model.values)+1)]
bank_model.index.name = 'id'
# Step 3b - split into x and y, split into train, validation, test
feat=bank_model.drop(columns=['y'],axis=1)
label=bank_model['y']
X_train, X_test, y_train, y_test = train_test_split(feat,label, test_size = 0.1, random_state = 103)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.3, random_state = 103)
print('step 3')
print(x_train, x_test, x_validate, y_validate, y_train, y_test)

# step 4 train the model
xgb = XGBClassifier(max_depth=3,colsample_bylevel=0.3,
              colsample_bynode=0.3,min_child_weight=1,learning_rate=0.05,n_estimators=100,reg_alpha=0,reg_lambda=0.5,scale_pos_weight=8,
                   random_state=1)
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_val)
xgbprd_train=xgb.predict(X_train)
print('Train Precision',precision_score(y_train, xgbprd_train ))
print('Test Precision',precision_score(y_val, xgbprd ))

# Step 5 - validate the model
xgb = XGBClassifier(subsample=0.9,scale_pos_weight=6,reg_lambda=0.6,n_estimators=100,min_child_weight=5,
                    max_features=11,max_depth=6,learning_rate=0.1,colsample_bynpde=0.35,
                    colsample_bylevel=0.35)
xgb.fit(X_train, y_train)
xgbprd = xgb.predict(X_test)
xgbprd_train=xgb.predict(X_train)
print('Train Precision',precision_score(y_train, xgbprd_train ))
print('Test Precision',precision_score(y_test, xgbprd ))
print(classification_report(y_val,xgbprd))
print('step 4')
print(xgb_model)

# Step 6 ) calculating lift
# step 6a) Calculating probability
y_predict_proba=xgb.predict_proba(X_test)
y_predict_proba=pd.DataFrame(y_predict_proba,index=X_test.index)
y_predict_proba=y_predict_proba.drop(0,axis=1)
y_predict_proba=pd.DataFrame(y_predict_proba, index=X_test.index)
bank_churn=pd.concat([y_predict_proba,y_test],axis=1)
bank_churn=bank_churn.sort_values(by=1,ascending=False)
# step 6b)  calculating decile
bank_churn['DecileRank']=pd.qcut(bank_churn[1],q=10,labels=False)
bank_churn=bank_churn.groupby('DecileRank')[1].mean()
bank_churn=pd.DataFrame(bank_churn)
baseline_churn=freq_pos/(freq_pos+freq_neg)
lift = (bank_churn['churnrate']-0.13)/0.13
bank_churn['lift']=lift
