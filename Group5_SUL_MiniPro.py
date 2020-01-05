
# coding: utf-8

# # Predicting the personal loan borrowers for Thera Bank 

# In[1]:


#Importing relevant Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx',sheet_name='Data')


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


# We can drop the Column ID since its not relevent:
data.drop("ID", axis=1,inplace=True)


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.describe()


# ## 1) Read the column description and ensure you understand each attribute well.
#         The Dataset has 13 columns 5000 rows, its a good representation of population:
#         
#         Out of 13 Columns, "Personal Loan" is the target variable and remaining 12 are independent attributes:
#         1) Age,Experience,Income,Zip Code, Family and Education: These coulmns represent customer demographic information.
#         2) CCAvg, Mortgage, Personal Loan, Securities Account, CD Account, Online, Creditcard: These columns represent customers         relationship with bank.
#         Categorical Variables: The columns Family,Education, Personal Loan, Securities Account, CD Account, Online and                   CreditCard are categorical variables.
#         Expercience: The variable Experience is having Negative values.
#         Income,CCAvg and Mortgage: These three columns have skewed distribution and there might be few outliers.
#         
#         
#         

# In[9]:


get_ipython().magic('matplotlib inline')
sns.pairplot(data)


# In[10]:


# Plotting Box plots for attributes "Income","CCAvg" and "Mortgage":


# In[11]:


sns.set(style="whitegrid", color_codes=True)
px= sns.boxplot(x="Income", data= data)


# In[12]:


sns.set(style="whitegrid", color_codes=True)
px= sns.boxplot(x="CCAvg", data= data)


# In[13]:


sns.set(style="whitegrid", color_codes=True)
px= sns.boxplot(x="Mortgage", data= data)


# In[14]:


# From the Box plots it is evident that the skewed distribution in the attributes "Income","CCAvg" and "Mortgage" is not
# because of outliers. But the data itself is skewed.


# In[15]:


# Checking the distribution in "Experience" column, to check negative values in distribution.


# In[16]:


sns.set(style="whitegrid", color_codes=True)
px= sns.boxplot(x="Experience", data= data, hue=True)


# # 2) studying data distribution in each attributes, Share your findings:

# In[17]:


categorical = ["Family","Education","Securities Account","CD Account","Online","CreditCard"]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 25))
for col, ax in zip(categorical[0:], axs.ravel()):
    sns.countplot(x=col, data=data, ax=ax)


# # From the distribution of the independent attribute we can infer that:
#     1) Majority of the banks customers are having a family size of 1
#     2) Majority of banks customers are Undergraduates followed by Advanced/Professional and Graduates.
#     3) Majority of banks existing customers are not holding "Securities account" and "CD account"
#     4) Majority of existing customers are using Online banking facility
#     5) Majority of banks customers do not have a Credit card.

# # Observations based on the distribution of People who have taken the personal loan:

# In[18]:


data_loan_yes = data.loc[data["Personal Loan"]==1]


# In[19]:


# Relationship between Income and People who have taken personal loan:
from matplotlib import pyplot as plt
plt.figure(figsize=(10,5))
sns.countplot(x='Income', data=data_loan_yes)
plt.xticks(rotation='vertical')
plt.title('Distribution')
plt.show()


# In[20]:


data_loan_yes.Income.mean()


# # Income > 144k are more likely to take Personal Loan

# In[21]:


# Relationship between Mortgage and People who have taken personal loan:
from matplotlib import pyplot as plt
plt.figure(figsize=(10,5))
sns.countplot(x='Mortgage', data=data_loan_yes)
plt.xticks(rotation='vertical')
plt.title('Distribution')
plt.show()


# # No Mortgage people are more prone to take personal loans

# In[22]:


# Relationship between Education and People who have taken personal loan:
from matplotlib import pyplot as plt
plt.figure(figsize=(10,5))
sns.countplot(x='Education', data=data_loan_yes)
plt.xticks(rotation='vertical')
plt.title('Distribution')
plt.show()


# # higher level of education are more likely to take personal loan followed by graduate

# In[23]:


# Relationship between Age and People who have taken personal loan:
from matplotlib import pyplot as plt
plt.figure(figsize=(10,5))
sns.countplot(x='Age', data=data_loan_yes)
plt.xticks(rotation='vertical')
plt.title('Distribution')
plt.show()
data_loan_yes.Age.mean()


# # Age on higer side are more likely to take personal loan. The Average age of people who have taken loan is 45 years.

# # 3) Get the target column distribution. Your comments:

# In[24]:


Target = data["Personal Loan"]
sns.countplot(x= Target, data=data)


# # It is evident from the distribution of Target variable("Personal Loan") that approximately only 10% of the customers have accepted the personal Loan.

# In[25]:


data.corr()


# In[26]:


plt.figure(figsize=(15, 15))
ax = sns.heatmap(data.corr(), annot = True, fmt='.2g', linewidths = 0.01)
plt.title('Cross correlation between attributes')
plt.show()


# #  From the correlation matrix we can infer that:
#     1)The attributes "Experience" and "Age" are having high positive correlation (correlation cofficient = 0.99)
#     2)The attributes "Income" and "CCAvg" has good positive correlation (correlation cofficient = 0.65)
#     
#     The attributes that are having good correlation with Target varibale("Personal Loan") are:
#     1) Income (0.5)
#     2) CCAvg (0.37)
#     3) CD Account (0.32)
#     
#     The attribute "Zip Code" has having very least correlation with respect to Target variable (0.00011)

# In[27]:


# Since "Zip code" is having least coefficient of correlation we can drop it from Dependant variables;
# Since "Experience" and "Age" are having high correlation of 0.99 we can drop either one of them from Dependant variables,Lets drop "Experience" since its having negative values.
data.drop(labels=['ZIP Code','Experience'],axis=1,inplace=True)
#data.drop(labels=['ID','ZIP Code','Experience'],axis=1,inplace=True)


# In[28]:


data[~data.applymap(np.isreal).all(1)]


# In[29]:


# Decision tree in Python can take only numerical / categorical colums. It cannot take string / obeject types. 
# The following code loops through each column and checks if the column type is object then converts those columns
# into categorical with each distinct value becoming a category or code.
data = pd.get_dummies(data,columns=['Family','Education','Personal Loan','Securities Account','CD Account','Online','CreditCard'],drop_first=True)


# # 4) Split the data into training and test set in the ratio of 70:30 respectively.

# In[30]:


from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[31]:


X = data.drop("Personal Loan_1",axis=1)
Y = data["Personal Loan_1"]


# In[32]:


# Splitting Data into 70% Training data and 30% Testing Data:
x_train, x_test, y_train,  y_test = train_test_split(X, Y,train_size=0.7, test_size=0.3, random_state=0)
print(len(x_train))
print(len(x_test))


# # Model-1 : Logistic Regression

# In[33]:


# Lets use the classification algorithm "LogisticRegression":
from sklearn.linear_model import LogisticRegression
logisticregression = LogisticRegression()
Model1 = logisticregression.fit(x_train, y_train)


# In[34]:


print("Model score on training : "+str(Model1.score(x_train,y_train)))
print("Model score on test : "+str(Model1.score(x_test,y_test)))


# In[35]:


y1_predict = Model1.predict(x_test)
print("Mean Absolute Error : " + str(metrics.mean_absolute_error(y1_predict, y_test)))
print("Confusion Matrix/n")
print(metrics.confusion_matrix(y_test, y1_predict))
print( metrics.accuracy_score(y_test, y1_predict) )


# In[36]:


print(metrics.classification_report(y_test, y1_predict))


# # Analyzing the confusion matrix:
# True Positives (TP): we correctly predicted the people have taken Personal loan 73
# 
# True Negatives (TN): we correctly predicted the people who have not taken Personal loan 1360
# 
# False Positives (FP): we incorrectly predicted that people have taken loan (a "Type I error") 12 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that people have not taken loan (a "Type II error") 55 Falsely predict negative Type II error

# # Model-2: K-Nearset neighbor's

# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


kNeighborsClassifier = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance')


# In[39]:


Model2 = kNeighborsClassifier.fit(x_train, y_train)


# In[40]:


print("Model score on training : "+str(Model2.score(x_train,y_train)))
print("Model score on test : "+str(Model2.score(x_test,y_test)))


# In[41]:


y2_predict = Model2.predict(x_test)
print("Mean Absolute Error : " + str(metrics.mean_absolute_error(y2_predict, y_test)))
print("Confusion Matrix/n")
print(metrics.confusion_matrix(y_test, y2_predict))
print( metrics.accuracy_score(y_test, y2_predict) )


# In[42]:


print(metrics.classification_report(y_test, y2_predict))


# # Analyzing the confusion matrix:
# True Positives (TP): we correctly predicted the people have taken Personal loan 42
# 
# True Negatives (TN): we correctly predicted the people who have not taken Personal loan 1325
# 
# False Positives (FP): we incorrectly predicted that people have taken loan (a "Type I error") 47 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that people have not taken loan (a "Type II error") 86 Falsely predict negative Type II error

# # Model-3: Naive bayes

# In[43]:


from sklearn.naive_bayes import GaussianNB


# In[44]:


gaussianNB = GaussianNB()


# In[45]:


Model3 = gaussianNB.fit(x_train, y_train)


# In[46]:


print("Model score on training : "+str(Model3.score(x_train,y_train)))
print("Model score on test : "+str(Model3.score(x_test,y_test)))


# In[47]:


y3_predict = Model3.predict(x_test)
print("Mean Absolute Error : " + str(metrics.mean_absolute_error(y3_predict, y_test)))
print("Confusion Matrix/n")
print(metrics.confusion_matrix(y_test, y3_predict))
print( metrics.accuracy_score(y_test, y3_predict) )


# In[48]:


print(metrics.classification_report(y_test, y3_predict))


# # Analyzing the confusion matrix:
# True Positives (TP): we correctly predicted the people have taken Personal loan as 76
# 
# True Negatives (TN): we correctly predicted the people who have not taken Personal loan as 1257
# 
# False Positives (FP): we incorrectly predicted that people have taken loan (a "Type I error") 115 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that people have not taken loan (a "Type II error") 52 Falsely predict negative Type II error

# # Model-4: Decision tree classifier:

# In[49]:


from sklearn.tree import DecisionTreeClassifier


# In[50]:


decisionTreeClassifier = DecisionTreeClassifier(criterion = 'entropy' )


# In[51]:


Model4 = decisionTreeClassifier.fit(x_train, y_train)


# In[52]:


from IPython.display import Image  
#import pydotplus as pydot
from sklearn import tree
from os import system

train_char_label = ['No', 'Yes']
Credit_Tree_File = open('credit_tree.dot','w')
dot_data = tree.export_graphviz(Model4, out_file=Credit_Tree_File, feature_names = list(x_train), class_names = list(train_char_label))

Credit_Tree_File.close()


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(Model4.feature_importances_, columns = ["Imp"], index = x_train.columns))


# In[53]:


# You can also copy the script in the .dot file and paste it at http://webgraphviz.com/ to get tree view 
#or create a .png as below

system("dot -Tpng credit_tree.dot -o credit_tree.png")
Image("credit_tree.png")


# In[54]:


print("Model score on training : "+str(Model4.score(x_train,y_train)))
print("Model score on test : "+str(Model4.score(x_test,y_test)))


# In[55]:


y4_predict = Model4.predict(x_test)
print("Mean Absolute Error : " + str(metrics.mean_absolute_error(y4_predict, y_test)))
print("Confusion Matrix/n")
print(metrics.confusion_matrix(y_test, y4_predict))
print( metrics.accuracy_score(y_test, y4_predict) )


# In[56]:


print(metrics.classification_report(y_test, y4_predict))


# # Analyzing the confusion matrix:
# True Positives (TP): we correctly predicted the people have taken Personal loan as 116
# 
# True Negatives (TN): we correctly predicted the people who have not taken Personal loan as 1365
# 
# False Positives (FP): we incorrectly predicted that people have taken loan (a "Type I error") 7 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that people have not taken loan (a "Type II error") 12 Falsely predict negative Type II error

# # Conclusion : 
# 
# From the above analysis all models overall accuracy is good, But when we look at precision and recall value, Recall accuracy is low,
# Which indicated that our models are overfitting on training data due to imbalance dataset.
# Even without any model one can say that the given test set belongs to Majority class with 90% confidence
# ### But Still Decesion tree models accuracy is good in terms of overall,precesion and recall as trees won't effect much from imbalance data

# ### From the data set , we have only 480 records which are positive and rest as negetive .
# ### This is Imbalanced Class data with 1:9 ratio

# In[57]:


# Let us look at the target column which is 'Personal Loan' to understand how the data is distributed amongst the various values
# Most are not negatives. The ratio is almost 1:9 in favor or class 0.  The model's ability to predict class 0 will 
# be better than predicting class 1. 
data.groupby(["Personal Loan_1"]).count()


# We have many ways to handle to imbalance data sets we are considering 2 methods 
# ### Case 1 : Changing model parameters or using Ensemble techniques with trees.
# ### Case 2 : Upsampling of minority class data

# # Case 1 :

# #                                      Regularising the Decision Tree

# In[58]:


#dt_model = DecisionTreeClassifier(criterion = 'entropy', class_weight={0:.5,1:.5}, max_depth = 5, min_samples_leaf=5 )
#dt_model.fit(train_set, train_labels)

model5 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
model5.fit(x_train, y_train)


# In[59]:


from IPython.display import Image  
#import pydotplus as pydot
from sklearn import tree
from os import system

train_char_label = ['No', 'Yes']
RegCredit_Tree_File = open('Regcredit_tree.dot','w')
dot_data = tree.export_graphviz(model5, out_file=RegCredit_Tree_File, feature_names = list(x_train), class_names = list(train_char_label))

RegCredit_Tree_File.close()


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(model5.feature_importances_, columns = ["Imp"], index = x_train.columns))


# In[60]:


# You can also copy the script in the .dot file and paste it at http://webgraphviz.com/ to get tree view 
#or create a .png as below

system("dot -Tpng Regcredit_tree.dot -o Regcredit_tree.png")
Image("Regcredit_tree.png")


# In[61]:


print("Model score on training : "+str(model5.score(x_train,y_train)))
print("Model score on test : "+str(model5.score(x_test,y_test)))


# In[62]:


y5_predict = model5.predict(x_test)
print("Mean Absolute Error : " + str(metrics.mean_absolute_error(y5_predict, y_test)))
print("Confusion Matrix/n")
print(metrics.confusion_matrix(y_test, y5_predict))
print( metrics.accuracy_score(y_test, y5_predict) )


# In[63]:


print(metrics.classification_report(y_test, y5_predict))


# # Analyzing the confusion matrix:
# True Positives (TP): we correctly predicted the people have taken Personal loan as 116
# 
# True Negatives (TN): we correctly predicted the people who have not taken Personal loan as 1368
# 
# False Positives (FP): we incorrectly predicted that people have taken loan (a "Type I error") 4 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that people have not taken loan (a "Type II error") 12 Falsely predict negative Type II error

# In[64]:


#Validating model with K-Fold Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model5, x_train, y_train, cv=5)
print(scores)


# # Gradient Boosting Models with XGBoost

# ### XGBoost takes care of Imbalance in datasets and helps in prediction accuracy on unseen data

# In[65]:


from xgboost import XGBClassifier

model6 = XGBClassifier(n_estimators=1000, learning_rate=0.05)
# Add silent=True to avoid printing out updates with each cycle
model6.fit(x_train, y_train,verbose=False)


# In[66]:


print("Model score on training : "+str(model6.score(x_train,y_train)))
print("Model score on test : "+str(model6.score(x_test,y_test)))


# In[67]:


y6_predict = model6.predict(x_test)
print("Mean Absolute Error : " + str(metrics.mean_absolute_error(y6_predict, y_test)))
print("Confusion Matrix/n")
print(metrics.confusion_matrix(y_test, y6_predict))
print( metrics.accuracy_score(y_test, y6_predict) )


# In[68]:


print(metrics.classification_report(y_test, y6_predict))


# In[69]:


#Validating model with K-Fold Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model6, x_train, y_train, cv=5)
print(scores)


# # Case 2:

# In[70]:


# Therefore, if we were to always predict 0 - people who will not take loan, we'd achieve an accuracy more than 90%.
# On the contrary , for people who will take loan , hardly we will achieve 10% accuracy


# In[71]:


# Up Sampling is one technique which can be used to make this data unbiased


# In[72]:


# Separate majority and minority classes
df_majority = data[data['Personal Loan_1']==0]
df_minority = data[data['Personal Loan_1']==1]


# In[73]:


from  sklearn.utils import resample
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=4520,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled['Personal Loan_1'].value_counts()


# In[74]:


Sampled_X = df_upsampled.drop("Personal Loan_1",axis=1)
Sampled_Y = df_upsampled["Personal Loan_1"]


# In[75]:


x_train, x_test, y_train,  y_test = train_test_split(Sampled_X, Sampled_Y,train_size=0.7, test_size=0.3, random_state=0)


# In[76]:


# Lets use the classification algorithm "LogisticRegression":
logisticregressionOnSampledData = LogisticRegression()
Model7 = logisticregressionOnSampledData.fit(x_train, y_train)


# In[77]:


print("Model score on training : "+str(Model7.score(x_train,y_train)))
print("Model score on test : "+str(Model7.score(x_test,y_test)))


# In[78]:


y7_predict = Model7.predict(x_test)
print("Mean Absolute Error : " + str(metrics.mean_absolute_error(y7_predict, y_test)))
print("Confusion Matrix/n")
print(metrics.confusion_matrix(y_test, y7_predict))
print( metrics.accuracy_score(y_test, y7_predict) )


# In[79]:


print(metrics.classification_report(y_test, y7_predict))


# In[80]:


#Validating model with K-Fold Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(Model7, x_train, y_train, cv=5)
print(scores)


# ### Need to repeat the process on KNN, Naive Base and Decesion Tree

# # 6 Explain why you chose one model over the other 

# Decession Tree is best suited model for the given dataset over other models.
# Because of imbalance in dataset, Decesion Tree classification is not effected much as compared to Logistic, KNN and Naive Base.
# 
# There are many techniques to use for this kind of data set.
# We have taken 2 cases: 1) By Changing Model parameters 2) Upsampling of Minarity Class data.
# 
# We can see the Overall accuracy, Precesion , Recall and f1-score are good in Decesion Tree , XGBClassifier and Logistic on Upsampled data. 

# In[81]:


import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


# In[84]:


models = []
models.append(('LR', Model1))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', Model2))
models.append(('CART', Model4))
models.append(('NB', Model3))
models.append(('SVM', SVC()))
models.append(('RegCART', model5))
models.append(('XGB', model6))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=12345)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

