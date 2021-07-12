import os
import sys
import nltk
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import yfinance as yf

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity as cs


def main():
    sentiment_scores_test('fb', 'tokenized_words.json')
    pass


def sentiment_scores_test(ticker, input_filename):
    sentiments = pd.read_csv('supporting_data/sentiment_dataframe.csv')
    
    with open(input_filename, 'r') as f:
        tokens = json.loads(f.read())
    
    documents = reformat_documents(ticker, tokens)
    sbow = sentiment_bag_of_words(documents, sentiments)
    scores = sentiment_scores(sbow)
    print(scores)
    plot_sentiment_scores(ticker, scores)


def sentiment_scores(sbow):
    scores = {}

    for year in sbow['positive']:
        scores[year] = 0
        scores[year] += sum(sbow['positive'][year]) / len(sbow['positive'][year])

    for year in sbow['superfluous']:
        scores[year] += sum(sbow['superfluous'][year]) / len(sbow['superfluous'][year])
    
    for year in sbow['interesting']:
        scores[year] += sum(sbow['interesting'][year]) / len(sbow['interesting'][year])

    for year in sbow['negative']:
        scores[year] -= sum(sbow['negative'][year]) / len(sbow['negative'][year])
    
    for year in sbow['litigious']:
        scores[year] -= sum(sbow['litigious'][year]) / len(sbow['litigious'][year])

    for year in sbow['uncertainty']:
        scores[year] -= sum(sbow['uncertainty'][year]) / len(sbow['uncertainty'][year])
    
    for year in sbow['constraining']:
        scores[year] -= sum(sbow['constraining'][year]) / len(sbow['constraining'][year])
    
    for item in scores:
        scores[item] = math.tanh(scores[item])

    return dict(sorted(scores.items()))


def plot_sentiment_scores(ticker, scores):
    m = min(scores.keys())
    M = max(scores.keys())
    
    df = yf.download(ticker, start = f'20{m}-01-01', end = f'20{M}-03-01', interval = '3mo')
    df.to_csv(f'price_time_series/{ticker}.csv')
    df = pd.read_csv(f'price_time_series/{ticker}.csv')
    price = {}

    for year in scores.keys():
        price[year] = 0

        for index, row in df.iterrows():
            if row['Date'][2:4] == year and row['High'] > price[year]:
                price[year] = row['High']

    try:
        price.pop('99')
        scores.pop('99')
    except:
        pass
    
    figure, (ax1, ax2) = plt.subplots(2)#, sharex = True
    
    ax1.plot(price.keys(), price.values())
    ax1.set(title = f'{ticker}', xlabel = 'years', ylabel = 'stock price')
    ax2.plot(scores.keys(), scores.values())
    ax2.set(xlabel = 'years', ylabel = 'sentiment score')

    #plt.savefig('sample', dpi=300)
        
    plt.show()


def similarity_test(input_filename, ticker):

    sentiments = pd.read_csv('supporting_data/sentiment_dataframe.csv')
    
    with open(input_filename, 'r') as f:
        tokens = json.loads(f.read())
    
    documents = reformat_documents(ticker, tokens)
    
    sbow = sentiment_bag_of_words(documents, sentiments)

    similarities = jaccard_similarity(sbow)

    with open('supporting_data/jaccard_similarities.json', 'w') as f:
        json.dump(similarities, f)

    stfidf = sentiment_tfidf(documents, sentiments)

    similarities = cosine_similarity(stfidf)

    with open('supporting_data/cosine_similarities.json', 'w') as f:
        json.dump(similarities, f)


def create_sentiment_dataframe():

    df = pd.read_csv('supporting_data/LoughranMcDonald_MasterDictionary_2018.csv')
    
    # Set column names and words to lower case
    df.columns = df.columns.str.lower()
    df['word'] = [str(word).lower() for word in df['word']]

    # Select sentiment word and word columns
    sentiment_words = list(df.columns[7:14])
    df = df[['word'] + sentiment_words]

    # Remove words with 0 occurences
    df[sentiment_words] = df[sentiment_words].astype(bool)
    df = df[(df[sentiment_words]).any(1)]

    # Stem words and remove duplicates
    wnl = WordNetLemmatizer()
    #df['word'] = WordNetLemmatizer().lemmatize(df['word'])
    df['word'] = [wnl.lemmatize(str(word)) for word in df['word']]
    df = df.drop_duplicates('word')

    return df


def reformat_documents(ticker, tokens):
    documents = {}

    for year in tokens[ticker]:
        documents[year] = ' '.join([item for sublist in tokens[ticker][year] for item in sublist])

    return documents


# Analysis
def sentiment_bag_of_words(documents, sentiments):
    sentiment_words = list(sentiments.columns[2:9])
    sbow = {}
    
    for word in sentiment_words:
        sbow[word] = {}
        vectorizer = CountVectorizer(vocabulary = sentiments[sentiments[word]]['word'], 
                analyzer = 'word', 
                lowercase = False, 
                dtype = np.int8)

        model = vectorizer.fit(documents.values())

        for year in documents.keys():
            sbow[word][year] = model.transform([documents[year]]).toarray()[0]

    return sbow


def sentiment_tfidf(documents, sentiments):
    sentiment_words = list(sentiments.columns[2:9])
    
    stfidf = {}
    
    for word in sentiment_words:
        stfidf[word] = {}
        vectorizer = TfidfVectorizer(vocabulary = sentiments[sentiments[word]]['word'], 
                analyzer = 'word', 
                lowercase = False, 
                dtype = np.int8)

        model = vectorizer.fit(documents.values())

        for year in documents.keys():
            stfidf[word][year] = vectorizer.transform([documents[year]]).toarray()[0]

    return stfidf


def jaccard_similarity(sbow):

    similarities = {}

    for word in sbow:
        similarities[word] = {}

        years = sorted(sbow[word].keys())
        for i in range(len(years)-1):
            x = sbow[word][years[i]].astype(bool)
            y = sbow[word][years[i + 1]].astype(bool)
            similarities[word][years[i]] = jaccard_score(x, y)
            
    return similarities


def cosine_similarity(stfidf):

    similarities = {}

    for word in stfidf:
        similarities[word] = {}

        years = sorted(stfidf[word].keys())
        for i in range(len(years)-1):
            x = stfidf[word][years[i]].reshape(1, -1)
            y = stfidf[word][years[i+1]].reshape(1, -1)
            sim = cs(x, y)[0,0]
            similarities[word][years[i]] = cs(x, y)[0,0]
            
    return similarities


if __name__ == '__main__':
    main()
