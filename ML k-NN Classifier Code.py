# ML k-NN Classifier Using Capital Gain, Education-Num, and Marital-Status as Features to Predict Income
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

colnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
tree_df = pd.read_csv('adult.data', names=colnames)

delete_colnames = ['age', 'workclass', 'fnlwgt', 'education', 'occupation', 'relationship', 'race', 'sex', 
                   'capital-loss', 'hours-per-week', 'native-country']
salient_col = ['capital-gain', 'education-num','marital-status', 'income']

tree_df = tree_df.drop(delete_colnames, axis = 1)
tree_df = tree_df.dropna()


#find percentage of total individuals < $50K

count_under50k=0
count_over50k=0

for i in range(len(tree_df)):
    
    if tree_df.income[i] == ' <=50K':
        count_under50k+=1
    else:
        count_over50k+=1

total_count = count_under50k + count_over50k

print("count <= $50K: ", count_under50k)

print("count > $50K: ", count_over50k)

print("total count of individuals: ", total_count)

print("Percent of Total <= $50K: ", count_under50k/total_count)

print("Percent of Total > $50K: ", count_over50k/total_count)

print("Before Codes:\n", tree_df)
print(tree_df.dtypes)

tree_df['marital-status'] = tree_df['marital-status'].astype('category')
tree_df['income'] = tree_df['income'].astype('category')

#assign the encoded variable to a new column using the cat.codes accessor:
tree_df['marital-status'] = tree_df['marital-status'].cat.codes
tree_df['income'] = tree_df['income'].cat.codes

#print what codes go with which category


print("After Codes:\n", tree_df)
print(tree_df.dtypes)

#tree_df.to_csv('tree_df.csv') #print df to csv file

#split dataset in features and target variable

feature_cols = ['capital-gain', 'education-num','marital-status']

X = tree_df[feature_cols]  # Features

y = tree_df.income   #Target Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

# Model Accuracy, how often is the classifier correct?

print("\n\nRESULTS OF k-NN CLASSIFICATION \nUSING 'Capital Gain', 'Education Number, \nand Marital Status \nto Predict Income Above/Below $50,000': \n")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred))

print("F1 Score:", metrics.f1_score(y_test, y_pred))
