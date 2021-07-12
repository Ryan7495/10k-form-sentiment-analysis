import sys, os
import multiprocessing as mp
import re
import nltk
import random
import pandas as pd
import matplotlib as plt
from tqdm import tqdm
import json

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


#Usage: preprocess.py <output filename>

def main():
	path = os.path.join(os.getcwd(), 'documents')
	files = mp_import_files(path)
	processed_files = mp_sentence_tokenize(files)
	print()
	print("Writing to JSON file...")
	write_to_json(sys.argv[1], processed_files)
	print("Done!")


def tokenize_sentence(item):
    key, fold_dict = item
    fp = FormProcessor()
    result = {}
    for file_key in fold_dict.keys():
        result[file_key] = fp.process_form(fold_dict[file_key])
    
    return key, result


def mp_sentence_tokenize(file_dict,chunksize=10):
    with mp.Pool() as pool:
        tokens = list(tqdm(pool.imap_unordered(tokenize_sentence, file_dict.items(),chunksize), total=len(file_dict), desc="Processing forms"))
        
    return {key:lis for (key,lis) in tokens}


def mpp_import_files(item):
    year_pos = 11
    whitespace = re.compile('\s+')
    otherstuff = re.compile('[^A-Za-z .]+')
    folder, path = item
    result = {}
    fold_path = os.path.join(path, folder)
    for file in os.listdir(fold_path):
        year = file[year_pos:year_pos+2]
        file_path = os.path.join(fold_path, file)
        with open(file_path, 'r') as opened_file:
            result[year] = whitespace.sub(' ',otherstuff.sub(' ', opened_file.read()))
    
    return folder, result


def mp_import_files(path,chunksize=10):
    with mp.Pool() as pool:
        tokens = list(tqdm(pool.imap_unordered(mpp_import_files, [(f,path) for f in os.listdir(path)], chunksize), total=len(os.listdir(path)),desc="Reading in forms"))
        
    return {key:lis for (key,lis) in tokens}


def write_to_json (filename, forms):
    with open(filename, "w") as outfile:
        json.dump(forms, outfile)


def load_json (filename):
    with open(filename, "r") as infile:
        return json.load(infile)


class FormProcessor:
    def __init__(self):
        self.stopWords = set(stopwords.words('english'))
        self.wnl = WordNetLemmatizer()
        
    def process_form(self, form):
        sentences = sent_tokenize(form)
        words = [word_tokenize(sentence) for sentence in sentences]
        wordsFiltered = []

        for sent in words:
            sentFiltered = []
            for w in sent:
                if w not in self.stopWords:
                    sentFiltered.append(self.wnl.lemmatize(w))
            wordsFiltered.append(sentFiltered)
            
        return wordsFiltered

def split_up_json(json_obj, folder_name):
    if folder_name not in os.listdir():
        os.mkdir(folder_name)

    path = os.path.join(os.getcwd(), folder_name)

    for ticker in json_obj.keys():
        folder = os.path.join(path, ticker)
        if ticker not in os.listdir(path):
            os.mkdir(folder)

        for year in json_obj[ticker].keys():
            filename = os.path.join(folder, year)
            with open(filename,'w') as f:
                json.dump(json_obj[ticker][year],f)

def tokenized_words_subset(input_filename):

    #f = open('tokenized_words.json', 'r')
    f = open(input_filename, 'r')
    tokens = json.loads(f.read())
    f.close()

    subset = {}

    keys = []

    for key in tokens:
        keys.append(key)

    subset_keys = random.choices(keys, k = 250)

    for ticker in subset_keys:
        subset[ticker] = {}

        for year in tokens[ticker]:
            subset[ticker][year] = tokens[ticker][year]

    with open('supporting_data/tokenized_words_subset.json', 'w') as f:
        json.dump(subset, f)


if __name__ == '__main__':
    main()
