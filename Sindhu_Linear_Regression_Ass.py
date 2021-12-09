# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 19:33:29 2021

@author: Madhav
"""
'''
PREDICTE CURRENT MILK PRODUCTION FROM THE SET OF MEARSURED VARIABLES

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# function method train_test_split from the sklearn 
# package to split the data into train_test_split
from sklearn.model_selection import train_test_split

# from sklearn package importing the Linear Regression Class
from sklearn.linear_model import LinearRegression

# from the metrics sub package importing the mean_squared_error 
# class to calculate the RMSE
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.metrics import r2_score

import statsmodels.api as sm

# Setting the plot size
sns.set(rc={'figure.figsize':(12,9)})




# Reading the data
milk_data = pd.read_csv('Milk Production Data.txt',delimiter='\t' )

# Creating a Deep Copy 
milk=milk_data.copy()


# ====================  Attributes ==========================================
milk.columns
milk.shape
milk.info()
milk.index
milk.size
milk.ndim
milk.dtypes.value_counts()


#======================= Checking the unique values =========================  
np.unique(milk['CurrentMilk'])
np.unique(milk['Previous'])
np.unique(milk['Fat'])  #3.6 % Fat
np.unique(milk['Protein']) #3.2%protein
np.unique(milk['Days'])
np.unique(milk['Lactation'])
np.unique(milk['I79'])


#======================= Checking the Null values ============================

milk.isnull().sum()

milk.describe()


'''
-For the CurrentMilk, Previous mean is almost equal to median.
- For the above mentioned variables mximum values are close to 75% .
-Fat slightly mean>median.So it is slightly right- skewed distribution.
- Protein median lamost equals mean. Sould be a normal distribution.
- Hence,CurrentMilk, Previous,Fat and Protein are normally distributed.
- For Days the mean is slightly greater than median , hence it is 
right skewed distribution.
- For I79 mean is lesser than median,hence but diff between 25% dnd
min is higher, the data is left skewed.
'''

#========================== Data Cleaning ===================================

#************************** Variable CurrentMilk ***************************

milk['CurrentMilk'].value_counts()
sns.histplot(milk['CurrentMilk'])

'''
-histogram shows that around 35 cows produced milk in the current month to about
55-60 pounds.
-Shows a bell shaped curve with a slight variation at the end which can be 
 outliers

'''
sns.boxplot(y=milk['CurrentMilk'])
'''
-Box plot shows the normal distribution except for the two outlier points 
 which is even visible in the histogram plot.

'''


milk['CurrentMilk'].describe()

# Checking for Outlier


Q1=milk['CurrentMilk'].quantile(0.25)
Q3=milk['CurrentMilk'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print(lower_limit,upper_limit)
milk[(milk['CurrentMilk']<lower_limit)|(milk['CurrentMilk']>upper_limit)]

''' 
CurrentMilk values 113 and 14 are the outliers
 Maintaining the outliers assuming dropping them would lead to lossing of
 information as the it is a smaller dataset .
 
'''


#************************** Variable Previous *******************************
milk['Previous'].value_counts()

sns.histplot(milk['Previous'])
'''
-histogram shows that above 40 cows produced milk in the previous month around 
55-65 pounds of  milk.
-Shows a bell shaped curve with a slight variation at the end which can be 
 outliers

'''
sns.boxplot(y=milk['Previous'])
'''
-Box plot shows the normal distribution except for the three outlier points.

 

'''

# Checking for Outlier

milk['Previous'].describe()
Q1=milk['Previous'].quantile(0.25)
Q3=milk['Previous'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print(lower_limit,upper_limit)
milk[(milk['Previous']<lower_limit)|(milk['Previous']>upper_limit)]

''' 
Previous points less than equal to 20 and greater than equal to 104 are outliers
 Maintaining the outliers assuming dropping them would lead to lossing of
 information as the it is a smaller dataset .
 
'''




#************************** Variable Fat ***********************************


# Milk fat ranged from 1.77% to 5.98%

milk['Fat'].value_counts().sort_index()



sns.histplot(milk['Fat'])
'''
-Histogram aroung 38 cows milk have percentage of Fat between 3.6-4%.
- It is not a normal distibution. it is a mild right skewed distribution.

'''
sns.boxplot(y=milk['Fat'])
'''
-Mildly right skewed with outiers above the upper wiskers.
-Maitaining the Outliers to assuming some cow produce high fat milk.

'''

# Checking for Outliers

milk['Fat'].describe()
Q1=milk['Fat'].quantile(0.25)
Q3=milk['Fat'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print(lower_limit,upper_limit)
milk[(milk['Fat']<lower_limit)|(milk['Fat']>upper_limit)]

''' 
Outliers from the point in greater than equal to 5.7 value of Fat
 Maintaining the outliers assuming dropping them would lead to lossing of
 information as the it is a smaller dataset .
 
'''





#************************** Variable Protein *********************************
# herd average milk protein ranged from 1.57% to 4.66%

milk['Protein'].value_counts().sort_index()


sns.histplot(milk['Protein'])

'''
Shows a Right Skewed distribution with two clear outlier points

'''

sns.boxplot(y=milk['Protein'])

'''
As the end of the whisker is shorter so it ia right skewed distibution

'''
# Checking for Outliers

milk['Protein'].describe()
Q1=milk['Protein'].quantile(0.25)
Q3=milk['Protein'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print(lower_limit,upper_limit)
milk[(milk['Protein']<lower_limit)|(milk['Protein']>upper_limit)]

''' 
Outliers from the point in greater than equal to 4.8 value and 1.8 value 
of Protein
 Maintaining the outliers assuming dropping them would lead to lossing of
 information as the it is a smaller dataset .
 
'''
# Totally 17 rows of outliers in 200 rows of data



#************************** Variable Days *********************************

milk['Days'].value_counts()

sns.histplot(milk['Days'])


sns.boxplot(y=milk['Days'])

'''
The maximum no. of cows are in the mid lactation period (120-240 days)

Normal Distribution of the for the no. of Lactation days
with median at around 150 days

'''



#************************** Variable Lactation *********************************

milk['Lactation'].value_counts()

sns.histplot(milk['Lactation'])

'''

The plots shows heavily right skewed distribution
Maximun cows had maximum of 2 lactations only

'''

sns.boxplot(y=milk['Lactation'])
'''

The plots shows heavily right skewed distribution

Very few cows are in the 7th & 8th lacatation cycle.

'''

# Checking Outliers
milk['Lactation'].describe()
Q1=milk['Lactation'].quantile(0.25)
Q3=milk['Lactation'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print(lower_limit,upper_limit)
milk[(milk['Lactation']<lower_limit)|(milk['Lactation']>upper_limit)]

'''
Not Considering as Outlier as the cows are in lactating for 8th and 9th time 
which is possible

'''

#************************** Variable I79 *********************************
milk['I79'].value_counts()
sns.histplot(milk['I79'])


sns.boxplot(y=milk['I79'])



# ======================== Visiualization Parameter ==========================
'''
========================== Question ===========================================
a) Do you think a linear model is appropriate to describe the association
 between response and explanatory? If yes, is the association/correlation 
 between response and each explanatory variable are reasonably strong or 
 quite weak? By plotting each independent variable vs. dependent variable 
 suggest an increasing or decreasing relationship between them? 
b) Describe all the possible relationships (independent vs. dependent). 
Which relationship seems most appropriate for a linear model


Ans.
a) 
i)Here our problem statement is to PREDICTE CURRENT MILK PRODUCTION FROM 
THE SET OF MEARSURED VARIABLES in which our variable CurrentMilk is a DEPENDENT
variable and all other variables are independent variable.
ii) Also we are predicting a quantum value 
Therefore, we can classify this problem as Regression Problem and we will use 
linear model.
b) Below are the scatter plots and correlation matrix for the data

=============================================================================
'''
#***********************  Previous vs CurrentMilk **************************
sns.regplot(x='Previous', y='CurrentMilk', scatter=True,fit_reg=False, data=milk)
'''

Scatter plot shows that CurrentMilk  is linearly increasing
with respect to previous. However, there are few points scattered away 
from the linear line wich can be due to the outliers.

The linearly increasing curve shows that maximum cows either produced the same 
quantity of milk or more milk compared  to previous month.

Only very few cows produced less milk compared to previous month.

'''
#***********************  Fat vs CurrentMilk ********************************
sns.regplot(x='Fat', y='CurrentMilk', scatter=True,fit_reg=False, data=milk)
'''
Scatter plot does not show any linear/ non lienar relationship between
Fat and CurrentMilk. The plot is a scattered one.

'''
#***********************  Protien vs CurrentMilk ******************************

sns.regplot(x='Protein', y='CurrentMilk', scatter=True,fit_reg=False, data=milk)

'''
The plot is accumlated in the range with 2-4 with as few outliers visiable.
Does not show much relationship between the two variables


'''
#***********************  Days vs CurrentMilk ********************************
sns.regplot(x='Days', y='CurrentMilk', scatter=True,fit_reg=False, data=milk)
'''
Linearly decreasing Trend. As the Days of Lactaion increase 
the production decrease

'''
#***********************  Lactation vs CurrentMilk *******************************

sns.regplot(x='Lactation', y='CurrentMilk', scatter=True,fit_reg=False, data=milk)
'''

It shows some relationship between CurrentMilk and Lacataion.It shows that there
many cows with 1st and 2nd lactation cycle. cows in 1st lactation cycle 
have productin of  milk above 40 pounds.The cows in 2nd cycle range from 14 to100
 Very few cows are in lactation cycle above 2 and maximu produce milk more than
 40pounds
 
'''

#***********************  I79 vs CurrentMilk *******************************

sns.regplot(x='I79', y='CurrentMilk', scatter=True,fit_reg=False, data=milk)
'''
Most cows are in mid or late lactation stage and production of milk vary from 
14 to 113 pound. However, as this variable has a relation with Days so not 
considering it into Model t avoid Multicollinearity

'''

# ======================== Correlation =======================================


correlation=milk.corr()
print(correlation)
'''

Correlation Matrix shows
-Positive correlation between CurrentMilk and Previous which we saw in the 
scatter plot between the two variables.
- Fat is weakly correlated with the CurrentMilk which shows the with increase
the yeild of milk there is slight decrease in Fat content in the milk
- Protein is negatively correlated with the with CurrentMilk implies 
the increase in CurrentMilk decreases the content of Protein
-Days of Lactation is negative correlation with CurrentMilk. As the cows 
lactation cycle moves from early stage to mid stage there is a decrease in the 
production of milk
- Lactation shows a weak positive correlation with CurrentMilk.
-I79 and days are highly correlated 


'''

'''
Ans
b) Based on the plots and correlation matrix:
    Reduced Model-CurrentMilk~Previous+Days+Lactation

'''
    


# =========================== LINEAR REGRESSION MODEL  =======================
'''

Assumption :
    
    1.There should be linear and additive relationship between dependent and
    independent varaible.
    2.There should be no correlation between the residual terms i.e. absence of
    'Autocorrelation'.
    3. The independent variable should not be correlated i.e. the absence of 
    'Multicollinearity'
    4. The error terms must have constant variance i.e. 'Homoskedasticity' to 
    exsist.
    5.The error terms must be normally distributed.
    
    
'''   
# =========================== Full Model =====================================    
'''  
QUESTION : 
a) Build a regression model using response and explanatory variables.

SOLUTION:
 Full Model - CurrentMilk~Previous+Fat+Protein+Days+Lactation   
 
 Not including I79 in the Model as it is the Indicator Flag of variable Days
 which may cause duplicating of varaible. Also the correlation matrix show 
 high correlation between the two so abiding the Mulicollinearity point.

'''
# Separating input and output feature
x1=milk.drop(['CurrentMilk','I79'], axis=1)
y1=milk.filter(['CurrentMilk'],axis=1)


# splitting data in test and train
# test_size set at 30% data
# seed value using random_state parameter for reproducing the output 
# consistently every time we use the sets of code
X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# Function to Measure Multicollinearity
def calculateVIF(data):      
    features = list(data.columns)   
    num_features = len(features)   
    model = LinearRegression()   
    result = pd.DataFrame(index = ['VIF'], columns = features)   
    result = result.fillna(0)   
    for ite in range(num_features):     
        x_features = features[:]     
        y_feature = features[ite]     
        x_features.remove(y_feature)          
        model.fit(data[x_features], data[y_feature])     
        result[y_feature] = 1 / (1 - model.score(data[x_features], data[y_feature]))   
    return result

# Function for rmse to Evaluate the model
def rmse(test_y, predicted_y):
    rmse_test=np.sqrt(mean_squared_error(test_y,predicted_y))
    base_pred=np.repeat(np.mean(test_y),len(test_y))
    rmse_base=np.sqrt(mean_squared_error(test_y,base_pred))
    values={'RMSE-test from Model': rmse_test, 'Base RMSE': rmse_base}
    return values

# Model Building
X_train2=sm.add_constant(X_train)
model_lin1=sm.OLS(y_train,X_train2)
results1=model_lin1.fit()
print(results1.summary())

'''
Adj. R-squared:                  0.708
The Adjusted R-squared shows 70.8% which falls between 60-80%. This shows that 
the model has 70.8% correlation between varaiables which shows the model is good. 

Prob (F-statistic):           5.57e-35
The model is significant as the value is less than 0.05

Also the value of coeff. lies in between the confidence interval which implies
the coeff values chosen for the model are significant.

Durbin-Watson:                   1.871
The value is closer to 2 which shows no AutoCorrelation


Here, in p-value for Fat-0.295,Protein -0.063 and Lactation -0.310

'''

# Predicting the model over train data for diagnostics 
milk_predictions_lin1_train = results1.predict(X_train2)
print(milk_predictions_lin1_train)

# Model Prediction
X_test=sm.add_constant(X_test)
milk_predictions_lin1_test=results1.predict(X_test)
print(rmse(y_test,milk_predictions_lin1_test))


'''
The predicted RMSE from the test dataset is less than base RMSE with a 
difference of 5.
This Confirms the Model is good

'''

# # Measuring multicollinearity using VIF 
vif_val=calculateVIF(X_train)
print(vif_val.transpose())

'''
QUESTION:
How many significant variables were used to fit the reduced model?
SOLUTION:
For very column’s p-value is <0.005 and VIF is <5 

When VIF greater than 5 then it shows a critical level of multicolliniartiy
and coeff are poorly estimated and variables should not be considered for 
model building. But here our value is 1.05 which is b/w/ 1 and 5 which mean 
there is no multicolliniarity b/w independent variables.

But we see the p-values for Fat, Protein and Lactation are not <0.005
Fat-0.295,Protein -0.063 and Lactation -0.310
So, these varaibles fit to use for Reduced Models are Previous & Days


'''

# ****************************  Diagnostics *********************************
residuals = y_train.iloc[:, 0] - milk_predictions_lin1_train 


# Residual plot 
sns.regplot(x = milk_predictions_lin1_train, y = residuals) 
plt.xlabel("Fitted values") 
plt.ylabel("Residuals") 
plt.title('Residual plot')


'''

Same variability is observered in the data which shows Homoskedasticity

'''


# Q-Q plot
sm.qqplot(residuals) 
plt.title("Normal Q-Q Plot")
'''
Q-Q plot shows the points close to the 45degree line, which shows that the model
good to apply to the population

'''
'''
QUESTION:
Insert the line into the corresponding scatterplot and paste it into your 
report. Does the line provide a good fit?

'''

sns.regplot(x=y_test, y=milk_predictions_lin1_test, scatter=True,fit_reg=True)
plt.xlabel("Actual Value for the Test") 
plt.ylabel("Predicted Value for the Test") 
plt.title('Actual vs Predicted plot')

'''
QUESTION:
What percent of the variation in current month milk production is explained 
by influencing variables?
'''



r2 = r2_score(y_train, milk_predictions_lin1_train)
print('Variation in current month milk production is explained by \
      influencing variables for Perfect Model is', r2)
# =========================== Reduced Model =====================================

'''
Reduced Model-CurrentMilk~Previous+Days

'''
x2=milk.filter(['Previous','Days'], axis=1)

X2_train,X2_test=train_test_split(x2,test_size=0.3,random_state=3)
print(X2_train.shape,X2_test.shape)

# Model building 
X_train3=sm.add_constant(X2_train)
model_lin2=sm.OLS(y_train,X_train3)
results2=model_lin2.fit()
print(results2.summary())


'''
 Adj. R-squared:                  0.707
as against 0.708 Good Model
 
 Prob (F-statistic):                   5.98e-37
The model is significant as the value is less than 0.05

Also the value of coeff. lies in between the confidence interval which implies
the coeff values chosen for the model are significant.

Durbin-Watson:                   1.864
The value is closer to 2 which shows no AutoCorrelation

'''

# Model prediction 
X2_test=sm.add_constant(X2_test)
milk_predictions_lin2_test=results2.predict(X2_test)




# Model prediction 
print(rmse(y_test, milk_predictions_lin2_test))
'''
The predicted RMSE from the test dataset is less than base RMSE with a 
difference of 5.
This Confirms the Model is good
Full Model-
{'RMSE-test from Model': 14.925498772171514, 'Base RMSE': 20.04618971830363}

Reduced Model-
{'RMSE-test from Model': 14.961078653835512, 'Base RMSE': 20.04618971830363}
'''


# For diagnostics, we need to predict the model on the train data 
milk_predictions_lin2_train = results2.predict(X_train3)

# Diagnostics 
residuals = y_train.iloc[:, 0] - milk_predictions_lin2_train
# Residual plot 
sns.regplot(x=milk_predictions_lin2_train,y=residuals)
plt.xlabel("Fitted values") 
plt.ylabel("Residuals") 
plt.title('Residual plot')

'''

Same variability is observered in the data which shows Homoskedasticity

'''


# QQ plot 
sm.qqplot(residuals) 
plt.title("Normal Q-Q Plot")

'''
Q-Q plot shows the points close to the 45degree line, which shows that the model
good to apply to the population

'''

'''
The Full Model and the Reduced Model are 

Adj. R-squared:                  0.707
as against 0.708 for Full Model

Full Model-
{'RMSE-test from Model': 14.925498772171514, 'Base RMSE': 20.04618971830363}

Reduced Model-
{'RMSE-test from Model': 14.961078653835512, 'Base RMSE': 20.04618971830363}

All the check parameters are very close 
Full Model shows a better RMSE compared to Reduced Model
So, We can consider Full Model for our Prediction.

'''

'''
QUESTION:
Consider you as a Data analyst in The dairy Herd Improvement Cooperative (DHI),
suggest a most important factor which will contribute in the production of 
Current month milk production in pounds (explain it with the direction in 
which the rate changes)

standardized regression coefficients provide an easy way to estimate 
effect size that is indepedent of units.

'''



Y_norm = stats.zscore(y_train)
X1_norm = X_train.loc[:, X_train.columns != "const"]
X1_norm = pd.DataFrame(stats.zscore(X1_norm))
X1_norm.columns = X_train.columns
X1_norm = sm.add_constant(X1_norm)
check = pd.concat([round(X1_norm.mean(axis=0), 5), round(X1_norm.std(axis=0, ddof=0), 5)], axis=1)
check.columns=["mean", "std dev"]
check

modstd = sm.OLS(Y_norm, X1_norm)
modstd_res = modstd.fit()
modstd_res.summary()
print(modstd_res.params) 



'''
Previous month production and Days of Lactation contribute more in the 
production of Current Month Milk.

'''








