import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle

# Loading Data
df = pd.read_csv('balanced_reviews.csv')


# Dropping the NA rows from the data
df.dropna(inplace = True)


# Subsetting the data by not including the overall column having 3 
df = df[df['overall'] != 3]


# creating a new column using overall column
df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )


# Splitting the data to train and test
features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 ) 


vect = TfidfVectorizer(min_df = 5).fit(features_train)
features_train_vectorized = vect.transform(features_train)


# initializing the logistic regression
model = LogisticRegression()

model.fit(features_train_vectorized, labels_train)
predictions = model.predict(vect.transform(features_test))

# gettin the score 
roc_auc_score(labels_test, predictions)


pkl_filename = "pickle_model.pkl"

# to save the model in the pickle file
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# save the count vectorizer
pickle.dump(vect.vocabulary_, open(
    "feature.pkl","wb"))


"""
#To load the pickle file of the model

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    


pred = pickle_model.predict(vect.transform(features_test))

roc_auc_score(labels_test, pred)
"""
