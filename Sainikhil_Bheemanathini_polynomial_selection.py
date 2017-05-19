
# coding: utf-8

# In[1]:

#import packages
import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
import pandas as pd
get_ipython().magic(u'matplotlib inline')


# ### 1. Import Data

# In[3]:

#Read in data from a data file to data_df in DateFrame format

## Import data from poly_data.csv to data_df
data_df = pd.read_csv("poly_data.csv")

print(data_df.head(6))


# ### 2. Observe Data

# In[4]:

#joint plot (or scatter plot) of X1 and y
sns.jointplot(data_df['X1'], data_df['y'])


# In[5]:

#joint plot (or scatter plot) of X2 and y
sns.jointplot(data_df['X2'], data_df['y'])


# In[6]:

#joint plot (or scatter plot) of X1 and X2
sns.jointplot(data_df['X1'], data_df['X2'])


# ### Based on observing the above 3 diagrams and the p-values displayed, we found both X1 and X2 have close correlation with y. X1 and X2 are independent from each other. 

# ### 3. Split the Data

# In[7]:

# split the data into training and testing datasets
# the percentage of training data is 75%

#split point 
percentage_for_training = 0.75

#number of training data 
number_of_training_data = int(data_df.shape[0]*percentage_for_training)

#create training and testing datasets
train_df  = data_df[0:number_of_training_data]
test_df = data_df[number_of_training_data:]
print(train_df.shape)
print(test_df.shape)


# ### 4. Create Polynomial Features

# In[8]:

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

polynomial_features = PolynomialFeatures(degree=3)


# In[9]:

X_poly = polynomial_features.fit_transform(data_df[['X1','X2']])

X_train = X_poly[0:number_of_training_data]
X_test = X_poly[number_of_training_data:]


# ### 5. Create and Train a Linear Regression Model

# In[10]:

# mse() calculates mean square error of a model on given X and y
def mse(X, y, model):
    return  ((y-model.predict(X))**2).sum()/y.shape[0]


# In[11]:

# use all the features to train the linear model 
lm = LinearRegression()
lm.fit(X_train, train_df['y'])
train_mse = mse(X_train, train_df['y'], lm)
print("Training Data Set's MSE is: \t", train_mse)
test_mse = mse(X_test, test_df['y'], lm)
print("Testing Data Set's MSE is : \t", test_mse)


# # REFLECTION

# ### 6. Use Lasso in Linear Regression to Penalize Large Number of Features

# In[12]:

from sklearn.linear_model import Lasso

#Train the model, try different alpha values.
Lasso_model = Lasso(alpha=0.15,normalize=True, max_iter=1e5, )
Lasso_model.fit(X_train, train_df['y'])


# In[13]:

#see the trained parameters. Zero means the feature can be removed from the model
Lasso_model.coef_


# In[14]:

#let's see the train_mse and test_mse from Lasso when 
#alpha = 0.15

train_mse = mse(X_train, train_df['y'], Lasso_model)
print("Training Data Set's MSE is: \t", train_mse)
test_mse = mse(X_test, test_df['y'], Lasso_model)
print("Testing Data Set's MSE is : \t", test_mse)


# In[15]:

#let's try a large range of values for alpha first
#create 50 alphas from 100 to 0.00001 in logspace
alphas = np.logspace(2, -5, base=10, num=50)
alphas


# In[16]:

#use arrays to keep track of the MSE of each alpha used. 
train_mse_array =[]
test_mse_array=[]

#try each alpha
for alpha in alphas:
    
    #create Lasso model using alpha
    Lasso_model = Lasso(alpha=alpha,normalize=True, max_iter=1e5, )
    Lasso_model.fit(X_train, train_df['y'])
    
    #Calculate MSEs of train and test datasets 
    train_mse = mse(X_train, train_df['y'], Lasso_model)
    test_mse = mse(X_test, test_df['y'], Lasso_model)
    
    #add the MSEs to the arrays
    train_mse_array.append(train_mse)
    test_mse_array.append(test_mse)
    


# In[17]:

#plot the MSEs based on alpha values
#blue line is for training data
#red line is for the testing data
plt.plot(np.log10(alphas), train_mse_array)
plt.plot(np.log10(alphas), test_mse_array, color='r')


# ### There is something interesting between 0 and 1 in the above diagram. 0 mean 10^0=1 While 1 means 10^1 = 10  so, we will look closely within this range to find the optimal alpha value
# 

# In[18]:

# We can try a smaller search space now (a line space between 1 and 10)
alphas = np.linspace(1, 10, 1000)
train_mse_array =[]
test_mse_array=[]
alp = []

i = 0
for alpha in alphas:
    
    #create Lasso model using alpha
    Lasso_model = Lasso(alpha=alpha,normalize=True, max_iter=1e5, )
    Lasso_model.fit(X_train, train_df['y'])
    
    #Calculate MSEs of train and test datasets 
    train_mse = mse(X_train, train_df['y'], Lasso_model)
    test_mse = mse(X_test, test_df['y'], Lasso_model)
    
    #add the MSEs to the arrays
    train_mse_array.append(train_mse)
    test_mse_array.append(test_mse)
    alp.append(alpha)
    
    diff = train_mse - test_mse
    
    if (diff > 0 and i != 1):
        opt_alpha = alpha
        tr_mse = train_mse
        te_mse = test_mse
        i = 1

print("The optimal alpha is", opt_alpha)
print("Train MSE is", tr_mse)
print("Test MSE is", te_mse)

plt.plot(alp, train_mse_array)
plt.plot(alp, test_mse_array, color='r')


# ### By observing a smaller range of alpha, we can clearly see how the MSEs change as we change the model and features. Use the diagram to explain the trends of the two lines and summarize what you learned so far. 

# In[20]:

print("From the above diagram, I found that the optimized alpha value is 3.567. This means that, at this alpha value, we can prevent overfitting and underfitting. Lasso model with alpha value below 3.567 is underfitting and above 3.567 is overfitting the data. Instead of using linear model, I used Lasso linear model which is used to penalize number of features with low variance or correlation. These features can be found by looking at the coefficients of the Lasso linear model.\n")

print("Overall, from the dataset, I found the polynomial that best fits the data. I checked my model by dividing the dataset into training data and testing data. I generated the model using the training data of 75% and then test the model using the testing data of 25%. To prevent overfitting and underfitting, I generated Lasso linear model with the optimized alpha value to penalize the features and retain only the required features. Thus, I reached the best alpha value by minimizing the mean square error between training data and testing data.")


# In[21]:

##Thus, above cells fulfill the assignment.

