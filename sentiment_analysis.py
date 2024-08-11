from functools import total_ordering
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

'''
VADER Analysis
'''
file = 'amazon_reviews.csv'
df = pd.read_csv(file)
sentiment = SentimentIntensityAnalyzer()
df['compound_sent'] = [sentiment.polarity_scores(row)['compound'] for row in df['text']]
df['neg_sent'] = [sentiment.polarity_scores(row)['neg'] for row in df['text']]
df['pos_sent'] = [sentiment.polarity_scores(row)['pos'] for row in df['text']]
df['neu_sent'] = [sentiment.polarity_scores(row)['neu'] for row in df['text']]

# printing text and the scores
print(df['text'])
print(df['compound_sent'])
print()

# Looking at review #0
print('\n', "REVIEW #0")
print(df['text'][0])
print(df['compound_sent'][0])

# Looking at review #1
print('\n', "REVIEW #1")
print(df['text'][4])
print(df['compound_sent'][4])