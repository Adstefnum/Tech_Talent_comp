import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pickle


df = pd.read_csv("./debt_dataset.csv")
df.replace([-np.inf,np.inf],np.nan,inplace=True)
df=df.dropna()

#choose relevant columns
df_model = df[["Net_debt", "Net_income", "Name", "Country", "Sector",
       "Industry_Group", "Industry", "Sub_Industry", "Debt_change",
       "total_cases", "new_cases", "total_deaths", "new_deaths",
       "reproduction_rate", "weekly_hosp_admissions", "new_tests",
       "total_tests", "positive_rate", "tests_per_case", "tests_units",
       "total_vaccinations", "people_vaccinated", "people_fully_vaccinated",
       "total_boosters", "new_vaccinations", "stringency_index","Debt_change_num"]]

#get dummy data
df_dummies = pd.get_dummies(df_model)

# train test split
x = df_dummies.drop(columns=["Debt_change_num"],axis=1)
y = df_dummies.Debt_change_num.values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#multiple linear regression
x_sm = x = sm.add_constant(x)
model = sm.OLS(y,x_sm)
model.fit().summary()

lm = LinearRegression()
lm.fit(x_train, y_train)

np.mean(cross_val_score(lm,x_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

#Lasso regression
lm_l = Lasso(alpha=.13)
lm_l.fit(x_train,y_train)
np.mean(cross_val_score(lm_l,x_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,x_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)
plt.show()

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

# random forest 
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,x_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
parameters = {'n_estimators':range(10,300,10), 'criterion':('squared_error','absolute_error'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(x_train,y_train)

print(
gs.best_score_
,gs.best_estimator_
)

# test ensembles 
tpred_lm = lm.predict(x_test)
tpred_lml = lm_l.predict(x_test)
tpred_rf = gs.best_estimator_.predict(x_test)

print("preds",tpred_lm,tpred_lml,tpred_rf)

#check the accuracy of the prediction
print("acc",
mean_absolute_error(y_test,tpred_lm)
,mean_absolute_error(y_test,tpred_lml)
,mean_absolute_error(y_test,tpred_rf)
,mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)
)

#pick and save best model
# pickl = {'model': gs.best_estimator_}
pickl = {'model': lm_l}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

#loading the model
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

print("model test",model.predict(np.array(list(x_test.iloc[1,:])).reshape(1,-1))[0])

