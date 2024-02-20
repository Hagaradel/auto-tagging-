import re
import string
from typing import Union
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.translate import metrics
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# from sklearn.cluster import KMeans
from sklearn import svm,datasets
# from sklearn.linear_model import LogisticRegression

#TODO: Reading and Displaying Datasets

# Set display options to show all columns
pd.set_option('display.max_columns', None)

# Read datasets and handle their text in Latin
Questions = pd.read_csv('Questions.csv', encoding='latin')
Answers = pd.read_csv('Answers.csv', encoding='latin')
Tags = pd.read_csv('Tags.csv', encoding='latin')

#print(Questions.head(2))
print("##################################################################################################")
#print(Answers.head(2))
print("##################################################################################################")
#print(Tags.head(2))
# Check the data types of each column
#data_types_tags= Tags.dtypes
#print(data_types_tags)


#############################################################################################################################
#TODO:Dataset Preparation

#rename columns names of question
Questions.columns=['Id','OwnerUserId',	'CreationDate',	'CloseDate' , 'Score' , 'Title' , 'Question']#body to Question
Answers.columns=['Id_normal', 'OwnerUserId', 'CreationDate','Id','Score','Answer']#parentId to Id/body to Answer


# Dropping unecessary columns
Answers.drop(columns=['Id_normal', 'OwnerUserId', 'CreationDate'], inplace=True)

#grouping the answers based on the 'Id' column and then joining the individual answers within each group into a single string
Answers = Answers.groupby('Id')['Answer'].apply(lambda answer: ' '.join(answer))
Answers = Answers.to_frame().reset_index()

# Changing the data type of 'Tag' column from object to string
Tags['Tag']= Tags['Tag'].astype(str)

# Joining tags grouped by 'Id'
Tags = Tags.groupby('Id')['Tag'].apply(lambda tag: ' '.join(tag))
Tags = Tags.to_frame().reset_index()

#Merging all dataset to a Single dataset
new_data = Questions.merge(Answers, how='left', on='Id')
new_data = new_data.merge(Tags, how='left', on='Id')

# Dropping unecessary columns
new_data.drop(columns=[ 'OwnerUserId' , 'CreationDate' ,	'CloseDate' ], inplace=True)

#rename columns names of new data
new_data.columns = ['id','score','title','question','answer','tag']

# Creating 'tagcount' column,counts the occurrences of each tag
count = new_data.groupby('tag')['tag'].count()
count = count.to_frame()

#rename column name
count.columns = ['TagCount']
count = count.reset_index()

# Merging created column to the existing dataframe
new_data = pd.merge(new_data, count, how='left', on='tag')

#check null values
null_values = new_data.isnull().sum()
print("#########################null values##################################################")
#print(null_values)#answer

new_data = new_data.dropna()#drop null values

#note that : for better accuracy may can drop answer column which had null values  -_-

print(new_data.shape)
##reduce data
new_data = new_data[(new_data['TagCount'] >= 1100) & (new_data['score'] > 7)]


print("#################################Data after Preparation################################ ")
#print(new_data)



new_data.drop(columns=['score', 'id','TagCount'], inplace=True)
######################################################################################################################################
#TODO:Dataset Preprocessing

Lematizer = WordNetLemmatizer()
# Defining a function to remove punctuation
def punctuation_remover(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


#Defining a lemmatizer function
def Word_Lemmatizer(text):
    lemma = [Lematizer.lemmatize(word) for word in text]
    return lemma


# Changing the data type of 'title' , 'answer' and 'question' columns to string
new_data['title'] = new_data['title'].astype(str)
new_data['question'] = new_data['question'].astype(str)
new_data['answer'] = new_data['answer'].astype(str)


# Applying 'punctuation_remover' function on 'title' , 'answer' and 'question' columns
new_data['title'] = new_data['title'].apply(punctuation_remover)
new_data['question'] = new_data['question'].apply(punctuation_remover)
new_data['answer'] = new_data['answer'].apply(punctuation_remover)
print("#################################Data after punctuation_remover################################ ")
#print(new_data)

# Changing texts into lowercase
new_data['title'] = new_data['title'].str.lower()
new_data['question'] = new_data['question'].str.lower()
new_data['answer'] = new_data['answer'].str.lower()

# Removing HTML tags on 'title' , 'answer' and 'question' columns
new_data['question'] = new_data['question'].apply(lambda question: re.sub('<[^<]+?>', '', question))
new_data['answer'] = new_data['answer'].apply(lambda answer: re.sub('<[^<]+?>', '', answer))
new_data['title'] = new_data['title'].apply(lambda title: re.sub('<[^<]+?>', '', title))
print("##################Data after Removing HTML tags and Changing texts into lowercase ################# ")
#print(new_data)

# Splitting the texts into words
new_data['question'] = new_data['question'].str.split()
new_data['answer'] = new_data['answer'].str.split()
new_data['title'] = new_data['title'].str.split()

# Applying lemmatizer function to 'title' , 'answer' and 'question' columns
new_data['title'] = new_data['title'].apply(lambda title: Word_Lemmatizer(title))
new_data['answer'] = new_data['answer'].apply(lambda answer: Word_Lemmatizer(answer))
new_data['question'] = new_data['question'].apply(lambda question: Word_Lemmatizer(question))
print("#################################Data after Applying lemmatizer################################ ")
#print(new_data)
# Removing Stopword from 'title' , 'answer' and 'question' columns
new_data['title'] = new_data['title'].apply(lambda title: [word for word in title if word not in stopwords.words('english')])
new_data['question'] = new_data['question'].apply(lambda question: [word for word in question if word not in stopwords.words('english')])
new_data['answer'] = new_data['answer'].apply(lambda answer: [word for word in answer if word not in stopwords.words('english')])
print("#################################Data after Removing Stopword################################ ")
#print(new_data)


##########################################################################################################################################
#TODO: Features extraction(Word2Vec):-

vectorizer = TfidfVectorizer()

# Changing the data type of 'title' and 'question' columns to string
new_data['title'] = new_data['title'].astype(str)
new_data['answer'] = new_data['answer'].astype(str)

X1 = vectorizer.fit_transform(new_data['title'].str.lower())
X2 = vectorizer.fit_transform(new_data['answer'].str.lower())

# Initialize LabelEncoder
label_encoder = LabelEncoder()
# X_for_Kmean_Answer=vectorizer.fit_transform(new_data['answer'].str.lower())
#

new_data['tag'] = label_encoder.fit_transform(new_data['tag'])
y = new_data['tag'].values


##########################################################################################################################################
# TODO: Model training and testing:-

x_train, x_test, y_train, y_test = train_test_split(X2, new_data['tag'], test_size=0.4, random_state=10)
x_train_svm, x_test_svm, y_train_svm, y_test_svm= train_test_split(X2, new_data['tag'], test_size=0.35, random_state=10)
# x_train_Kmean=np.array(x_train_Kmean)
#
# print(list(x_train_Kmean[0]))
# # x_train_Kmean = label_encoder.fit_transform(x_train_Kmean)


accuracy = []
accuracy_kmean=[]


for i in range(1, 100):
    KNN = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
    prediction = KNN.predict(x_test)
    accuracy.append(metrics.accuracy_score(y_test, prediction))


svm_model=svm.SVC(kernel='linear',C=10 ,random_state=0).fit(x_train_svm,y_train_svm)
pred_svm=svm_model.predict(x_test_svm)
accuracy_svm=metrics.accuracy_score(y_test_svm,pred_svm)

rfc=RandomForestClassifier(n_estimators=2000)
rfc.fit(x_train,y_train)
pred=rfc.predict(x_test)
accuracy_RDF=metrics.accuracy_score(y_test,pred)


decision=DecisionTreeClassifier(random_state=10)
decision.fit(x_train,y_train)
pred_decision=decision.predict(x_test)


acc_decision=metrics.accuracy_score(y_test,pred_decision)

#
# for k in range(1,20):
#      kme=KMeans(n_clusters=k,random_state=0,n_init='auto').fit(x_train)
#      pre=kme.predict(x_test)
#      accuracy_kmean.append(metrics.accuracy_score(y_test,pre))


##########################################################################################################################################
#TODO: Results visualization:-

plt.figure(figsize=(10, 6))
plt.plot(range(1, 100 ), accuracy, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

max_accuracy = max(accuracy)
max_accuracy_index = accuracy.index(max_accuracy) + 1
print("Maximum Accuracy:", max_accuracy, "at K =", max_accuracy_index)

#
# max_accuracy_k = max(accuracy_kmean)
# max_accuracy_index_k = accuracy.index(max_accuracy_k) + 1
# print("Maximum Accuracy:", max_accuracy_k, "at K =", max_accuracy_index_k)

print("############################################")
print("acc of svm :",accuracy_svm)
print("acc of rfc :",accuracy_RDF )
print("acc of decision :",acc_decision)
# print ("acc of logisitc:",accuracy_log)
