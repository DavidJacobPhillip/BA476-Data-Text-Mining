import text_normalizer as tn
import pandas as pd


# -----------
# Splitting Text
# -----------
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

file = 'NewsgroupTopic.csv'
df = pd.read_csv(file)
df['cleaned_text'] = tn.normalize_corpus(df['text'], html_stripping=False, contraction_expansion=False,accented_char_removal=False, text_lower_case=True, text_lemmatization=False, special_char_removal=False, stopword_removal=False, misspelled_words_correction=False)
print(df['cleaned_text'])

train, test = train_test_split(df, test_size=0.2, random_state=42)  #the "0.2" represents the test size, and random_state allows us to have the exact same split every single time

print('PRINTING DATA:')
print(len(train))
print(train.head())
print(test.head())

# fit_transform will put the terms into the inverse document frequency matrix
tv = TfidfVectorizer()
article_train = tv.fit_transform(train['cleaned_text'])
article_test = tv.transform(test['cleaned_text'])

# -----------
# Naive Bayes
# -----------
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from time import time

start_time = time()
mnb = MultinomialNB()

# first column is all of the columns we are training, and the second column is the "anser"
mnb.fit(article_train, train['Class'])

# this is where we are actually predicting
article_predict = mnb.predict(article_test)
cm = confusion_matrix(article_predict, test['Class'])  # confusion matrix

# accuracy will be different every time because of the way the machine learns
# print(cm)
print('\n', "NAIVE BAYES:")
print('Multinomial Naive Bayes Accuracy: ' + str(accuracy_score(article_predict, test['Class'])))
end_time = time()
print('Multinomial Naive Bayes Time Elapsed: ' + str(end_time - start_time) + ' seconds')

# -----------
# Support Vector Machine: Stochastic Gradient Descent
# -----------
from sklearn.linear_model import SGDClassifier

start_time = time()

svm_sgd = SGDClassifier()
svm_sgd.fit(article_train, train['Class'])
article_predict = svm_sgd.predict(article_test)
cm = confusion_matrix(article_predict, test['Class'])
#print(cm)
print('\n', "SUPPORT VECTOR MACHINES:")
print('SVM SGD Accuracy: ' + str(accuracy_score(article_predict, test['Class'])))
end_time = time()
print('SVM SGD Accuracy: ' + str(end_time - start_time) + ' seconds')

# -----------
# Neural Network
# -----------
from sklearn.neural_network import MLPClassifier

start_time = time()

mlp = MLPClassifier(hidden_layer_sizes=(512, 512, 512), random_state=42,
                    solver='adam', learning_rate='adaptive', activation='relu')
mlp.fit(article_train, train['Class'])
article_predict = mlp.predict(article_test)
cm = confusion_matrix(article_predict, test['Class'])
print('\n', "NEURAL NETWORK:")
print('Neural Network Accuracy: ' + str(accuracy_score(article_predict, test['Class'])))
end_time = time()
print('Neural Network Accuracy: ' + str(end_time - start_time) + ' seconds')