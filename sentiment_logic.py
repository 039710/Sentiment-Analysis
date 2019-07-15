import re
import string
import sklearn
import pickle
import csv
from csv import DictReader
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

# Load Model
with open('model_svm', 'rb') as file:
    model = pickle.load(file)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocessing(dataset):
    for row in dataset:
        row['review'] = row['review'].casefold()
        row['review'] = re.sub('[^A-Za-z0-9 ]+','', row['review'])

        word_tokens = word_tokenize(row['review']) 

        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w)

        row['review'] = " ".join(filtered_sentence)

        stop_sentence = [] 

        for v in filtered_sentence:
            stop_sentence.append(ps.stem(v))

        row['review'] = " ".join(stop_sentence)
        



dataset = []
with open ('data2.csv') as csvfile:
    readCSV = csv.DictReader(csvfile, delimiter=',')
    for row in readCSV:
        rating = row['Rating']
        intent = ""
        if rating == "1" or rating == "2":
            intent = "Bad"
            dataset.append(
            {
                'review': row ['Review Text'],
                'intent': intent
            }
            )
        elif rating == "4" or rating == "5" or rating == "3":
            intent = "Good"
            dataset.append(
            {
                'review': row ['Review Text'],
                'intent': intent
            }
          )
            
            
            
#Remove review text without rating
for row in dataset:
    if row['review'] == '' or row['intent'] == '':
        dataset.remove(row)
        
preprocessing(dataset)




datatrain, datatest = train_test_split(dataset, test_size=0.4, random_state=40)
xtrain_counts = CountVectorizer().fit_transform([x['review'] for x in datatrain])
xtrain_tfidf = TfidfTransformer().fit_transform(xtrain_counts)
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf-svm', SVC(C=1.0, kernel='linear', gamma=0.5,probability=True)),])

text_clf_svm = text_clf_svm.fit([row['review'] for row in datatrain], [row['intent'] for row in datatrain])

predicted_svm = text_clf_svm.predict(row['review'] for row in datatest)

#PREPROCESS INPUT


def preprocessing_input(input_text):
    input_text = input_text.casefold()
    input_text = re.sub('[^A-Za-z0-9 ]+','', input_text)
    filtered_sentence = []
    word_tokens = word_tokenize(input_text) 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    input_text= " ".join(filtered_sentence)
    
    return input_text

#Predict
def predict_input(input_text):
    print("After preprocessing : ",preprocessing_input(input_text))
    pred = text_clf_svm.predict([preprocessing_input(input_text)])
    decision = text_clf_svm.predict_proba([preprocessing_input(input_text)])
    print()
    print("Result : " ,pred)
    print("---Bad---- ","---Good---")
    print(decision)
    result = pred 
    result = str("Predicted as ") + str(result) + str(" review with probability ")+ str(" Bad / Good : ") + str(decision)
   
    return result

predict_input("very bad,uncomfortable, cheap quality")