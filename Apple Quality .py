#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv("apple_quality.csv")
df


# In[4]:


df.info()


# In[5]:


# Printing shape and size of dataframe
print(f"shape -> {df.shape}")
print(f"size  -> {df.size}")


# In[6]:


# Investigating the 'Acidity' column to check if it can be converted to numeric
df['Acidity'] = pd.to_numeric(df['Acidity'], errors='coerce')


# In[7]:


## Checking data types
df.dtypes


# In[8]:


df.columns


# In[9]:


#check duplicate values
df.duplicated().sum()


# In[10]:


# Checking number of missing values
df.isnull().sum()


# In[11]:


#used to remove missing or NaN (Not a Number) values from a DataFrame or Series.
df = df.dropna()


# In[12]:


#A_id won't serve us , it's only numbers adding up by 1 until the end .
df = df.drop(columns=['A_id'])


# In[13]:


df.isnull().sum()


#  Let's check unique values in quality so we would convert it to a numeric values and we would use that in our predecting model     as y

# In[14]:


df.Quality.unique()


# In[15]:


#converting categorical (text-based) labels into numerical labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Encoding the 'Quality' column
df['Quality'] = label_encoder.fit_transform(df['Quality'])

# Displaying the first few rows of the dataframe with the new encoded column
df.head()


# In[16]:


# used to generate descriptive statistics of a DataFrame
df.describe()


# In[17]:


# Create a boxplot for all numerical columns
df.boxplot()

# Set plot labels and title
plt.title('Boxplot of All Columns')
plt.xlabel('Columns')
plt.ylabel('Values')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[18]:


# Calculate IQR for minimum nights
q1 = df['Size'].quantile(0.25)
q3 = df['Size'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 +1.5 * IQR
lower_bound = q1 -1.5 * IQR
upper_bound, lower_bound


# In[19]:


#find the outliers
df.loc[(df['Size']>upper_bound)|(df['Size']<lower_bound)]


# In[20]:


#trimming _delete the outlier data 
d=df.loc[(df['Size']<upper_bound)&(df['Size']>lower_bound)]
print("before:",len(df))
print("after:",len(d))
print("outliers:",len(df)-len(d))


# In[21]:


sns.boxplot(d['Size'])


# In[22]:


#capping -change the outliers to upper ()or lower limit values
d=df.copy()
d.loc[(d['Size']>upper_bound), 'Size']=upper_bound
d.loc[(d['Size']<lower_bound), 'Size']=lower_bound


# In[23]:


sns.boxplot(d['Size'])


# In[24]:


# Calculate IQR number of reviews
q1 = d['Weight'].quantile(0.25)
q3 = d['Weight'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 +1.5 * IQR
lower_bound = q1 -1.5 * IQR
upper_bound, lower_bound


# In[25]:


#trimming _delete the outlier data 
e=d.loc[(d['Weight']<upper_bound)&(d['Weight']>lower_bound)]
print("before:",len(d))
print("after:",len(e))
print("outliers:",len(d)-len(e))


# In[26]:


#capping -change the outliers to upper ()or lower limit values
e=d.copy()
e.loc[(e['Weight']>upper_bound), 'Weight']=upper_bound
e.loc[(e['Weight']<lower_bound), 'Weight']=lower_bound


# In[ ]:





# In[27]:


# Calculate IQR reviews per month
q1 = e['Sweetness'].quantile(0.25)
q3 = e['Sweetness'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 +1.5 * IQR
lower_bound = q1 -1.5 * IQR
upper_bound, lower_bound


# In[28]:


#trimming _delete the outlier data 
f=e.loc[(e['Sweetness']<upper_bound)&(e['Sweetness']>lower_bound)]
print("before:",len(e))
print("after:",len(f))
print("outliers:",len(e)-len(f))


# In[29]:


#capping -change the outliers to upper ()or lower limit values
f=e.copy()
f.loc[(f['Sweetness']>upper_bound), 'Sweetness']=upper_bound
f.loc[(f['Sweetness']<lower_bound), 'Sweetness']=lower_bound


# In[ ]:





# In[30]:


# Calculate IQR calculated host listings count
q1 = f['Crunchiness'].quantile(0.25)
q3 = f['Crunchiness'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 +1.5 * IQR
lower_bound = q1 -1.5 * IQR
upper_bound, lower_bound


# In[31]:


#trimming _delete the outlier data 
g=f.loc[(f['Crunchiness']<upper_bound)&(f['Crunchiness']>lower_bound)]
print("before:",len(f))
print("after:",len(g))
print("outliers:",len(f)-len(g))


# In[32]:


#capping -change the outliers to upper ()or lower limit values
g=f.copy()
g.loc[(g['Crunchiness']>upper_bound), 'Crunchiness']=upper_bound
g.loc[(g['Crunchiness']<lower_bound), 'Crunchiness']=lower_bound


# In[33]:


# Calculate IQR availability 365
q1 = g['Juiciness'].quantile(0.25)
q3 = g['Juiciness'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 +1.5 * IQR
lower_bound = q1 -1.5 * IQR
upper_bound, lower_bound


# In[34]:


#trimming _delete the outlier data 
h=g.loc[(g['Juiciness']<upper_bound)&(g['Juiciness']>lower_bound)]
print("before:",len(g))
print("after:",len(h))
print("outliers:",len(g)-len(h))


# In[35]:


#capping -change the outliers to upper ()or lower limit values
h=g.copy()
h.loc[(h['Juiciness']>upper_bound), 'Juiciness']=upper_bound
h.loc[(h['Juiciness']<lower_bound), 'Juiciness']=lower_bound


# In[36]:


# Calculate IQR availability 365
q1 = h['Ripeness'].quantile(0.25)
q3 = h['Ripeness'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 +1.5 * IQR
lower_bound = q1 -1.5 * IQR
upper_bound, lower_bound


# In[37]:


#trimming _delete the outlier data 
i=h.loc[(h['Ripeness']<upper_bound)&(h['Ripeness']>lower_bound)]
print("before:",len(h))
print("after:",len(i))
print("outliers:",len(h)-len(i))


# In[38]:


#capping -change the outliers to upper ()or lower limit values
i=h.copy()
i.loc[(h['Ripeness']>upper_bound), 'Ripeness']=upper_bound
i.loc[(h['Ripeness']<lower_bound), 'Ripeness']=lower_bound


# In[39]:


# Calculate IQR availability 365
q1 = i['Acidity'].quantile(0.25)
q3 = i['Acidity'].quantile(0.75)
IQR = q3 - q1
upper_bound = q3 +1.5 * IQR
lower_bound = q1 -1.5 * IQR
upper_bound, lower_bound


# In[40]:


#trimming _delete the outlier data 
j=i.loc[(i['Acidity']<upper_bound)&(i['Acidity']>lower_bound)]
print("before:",len(i))
print("after:",len(j))
print("outliers:",len(i)-len(j))


# In[41]:


#capping -change the outliers to upper ()or lower limit values
j=i.copy()
j.loc[(h['Acidity']>upper_bound), 'Acidity']=upper_bound
j.loc[(h['Acidity']<lower_bound), 'Acidity']=lower_bound


# In[42]:


j


# In[43]:


apple=j


# In[44]:


# Create a boxplot for all numerical columns
apple.boxplot()

# Set plot labels and title
plt.title('Boxplot of All Columns')
plt.xlabel('Columns')
plt.ylabel('Values')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[45]:


apple.duplicated().sum()


# In[46]:


apple=apple.dropna()


# In[47]:


apple.isnull().sum()


# In[48]:


apple.duplicated().sum()


# In[49]:


apple


# In[50]:


apple.shape


# In[51]:


# Assuming 'df' is your DataFrame
sns.scatterplot(x='Size', y='Weight', data=apple,hue='Quality')
plt.title('Size vs Weight')
plt.xlabel('Size')
plt.ylabel('Weight')
plt.show()


# In[52]:


# Pair Plots
sns.pairplot(apple, hue='Quality', diag_kind='kde')
plt.show()



# In[54]:


apple.hist(figsize=(15, 10), color='green')
plt.show()


# In[71]:


from sklearn.metrics import RocCurveDisplay, roc_auc_score
roc_auc = roc_auc_score(ytest, output)
# Create ROC curve display
roc_display = RocCurveDisplay.from_estimator(m,xtest, ytest)
#roc_display.plot()
plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
plt.show()





# In[72]:


# Get feature importances (coefficients)
feature_importances = m.coef_[0]

# Create a DataFrame to store feature names and their importances
feature_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})

# Sort features by importance
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('Feature Importance Plot (Logistic Regression)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[73]:


# Confusion Matrix (for Classification)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(ytest,output)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.show()

# Extract TP, TN, FP, FN
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

print(f'True Positives (TP): {TP}')
print(f'True Negatives (TN): {TN}')
print(f'False Positives (FP): {FP}')
print(f'False Negatives (FN): {FN}')


# In[74]:


# Count Plots (assuming 'Quality' is categorical)
sns.countplot(x='Quality', data=apple,palette='viridis')
plt.show()


# In[64]:


plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(apple.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[65]:


apple.corr()


# In[59]:


x = apple[['Size','Weight','Sweetness','Juiciness']]
y = apple['Quality']


# In[60]:


xtrain,  xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[61]:


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

#linear_regression
from sklearn.linear_model import LinearRegression
m = LinearRegression()
m.fit(xtrain,ytrain)
# In[62]:


# Initialize and train the logistic regression model
m = LogisticRegression(random_state=42)
m.fit(xtrain, ytrain)


# In[63]:


output= m.predict(xtest)
output


# In[66]:


ytest


# In[67]:


compaire = pd.DataFrame({'actual_value':ytest,'predict_value':output})
compaire


# In[68]:


# Make predictions on the test set (logistic)
#y_pred_logistic =m.predict(xtest)
accuracy = accuracy_score(ytest,output)
print("LogisticRegression:", accuracy)


# In[108]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(xtrain)
X_test_scaled = scaler.transform(xtest)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Train the SVM classifier
svm_classifier.fit(X_train_scaled, ytrain)

# Predictions on the testing set
y_pred = svm_classifier.predict(X_test_scaled)

# Calculate accuracy
svm_accuracy = accuracy_score(ytest, y_pred)
print("Accuracy:", svm_accuracy)


# In[69]:


from sklearn import metrics
mean_aberror = metrics.mean_absolute_error(ytest,output)
mean_sqerror = metrics.mean_squared_error(ytest,output)
rmsqurrerror = np.sqrt(metrics.mean_squared_error(ytest,output))
print(m.score(x,y)*100)
print(mean_aberror) 
print(mean_sqerror)
print(rmsqurrerror) 


# In[70]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train KNN classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(xtrain, ytrain)
knn_predictions = knn_classifier.predict(xtest)
knn_accuracy = accuracy_score(ytest, knn_predictions)
print("KNN Accuracy:", knn_accuracy)


# In[75]:


dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(xtrain, ytrain)
dt_predictions = dt_classifier.predict(xtest)
dt_accuracy = accuracy_score(ytest, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)


# In[76]:


# Initialize and train Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(xtrain, ytrain)
rf_predictions = rf_classifier.predict(xtest)
rf_accuracy = accuracy_score(ytest, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)


# In[78]:


from sklearn.naive_bayes import GaussianNB
# Initialize and train the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(xtrain, ytrain)

# Make predictions on the test set
nb_predictions = nb_classifier.predict(xtest)

# Calculate the accuracy score
accuracy = accuracy_score(ytest, nb_predictions)

print("Naive Bayes Accuracy:", accuracy)


# In[113]:


from sklearn.metrics import f1_score

# Assuming y_test contains the true labels and output contains the predicted labels
# y_test and output should be arrays/lists of binary values (0s and 1s)

# Calculate the F1 score
f1 = f1_score(ytest,output)

print("F1 Score:", f1)


# In[115]:


# Create a bar graph for knn, decision tree, random forest, and logistic regression
models = ['KNN', 'Decision Tree', 'Random Forest',  'Logistic Regression','Naive Bayes','svm','f1 score']
accuracy_values = [knn_accuracy, dt_accuracy, rf_accuracy, accuracy,accuracy,svm_accuracy,f1]
plt.figure(figsize=(13, 5))


# Plot the bar graph
bars = plt.bar(models, accuracy_values, color=['blue', 'green', 'red', 'orange','purple','skyblue','pink'])

# Add accuracy values on top of each bar
plt.bar_label(bars, labels=[f'{acc:.2f}' for acc in accuracy_values])

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Models')

# Show the plot
plt.show()


# In[79]:


#k-nearest neighbors
from sklearn.neighbors import KNeighborsRegressor
knn_model=KNeighborsRegressor(n_neighbors=3)
knn_model.fit(xtrain,ytrain)
knn_predictions=knn_model.predict(xtest)
#mean squared error
knn_mse=mean_squared_error(ytest,knn_predictions)
print("means squared error:",knn_mse)


# In[80]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor

#decision tree
dt_model = DecisionTreeRegressor()
dt_model.fit(xtrain, ytrain)
dt_predictions = dt_model.predict(xtest)
#mean squared error
d_mse=mean_squared_error(ytest,dt_predictions)
print("means squared error:",d_mse)


# In[81]:


#random forest
rf_model = RandomForestRegressor()
rf_model.fit(xtrain, ytrain)
rf_predictions = rf_model.predict(xtest)

#mean squared error
r_mse=mean_squared_error(ytest,rf_predictions)
print("means squared error:",r_mse)


# In[96]:


# Create a bar graph   for knn,decision tree , random forest 
models = ['KNN', 'Decision Tree', 'Random Forest']
mse_values = [knn_mse, d_mse, r_mse]
bars=plt.bar(models, mse_values, color=['blue', 'green', 'red'])

# Add accuracy values on top of each bar
plt.bar_label(bars, labels=[f'{mse:.2f}' for mse in mse_values])

plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error Comparison for Different Models')
plt.show()




