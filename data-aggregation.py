import os
import requests
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date
import yfinance as yf

from selenium import webdriver
from time import sleep
import string
import random


# https://sec-api.io/docs
def main():
    #build_large_url_csv()
    #generate_api_keys(6)
    #df = pd.read_csv('supporting_data/document_urls2.csv', index_col = 0)
    #get_documents(df)
    #get_ticker_cik_map()
    tickers = ['nvda', 'amd', 'intc', 'goog', 'aapl', 'amzn', 'msft', 'fb', 'uber', 'tsla', 'csco', 'spy']
    #build_small_url_csv(tickers)


def build_small_url_csv(tickers):
    cik_map = get_ticker_cik_map()
    url_file = 'supporting_data/document_urls2.csv'
    df = pd.read_csv(url_file, index_col = 0)

    counter = 0

    index, key = get_api_key()
    print(key)

    for ticker in tickers:
        if key is None:
            return
        counter += 1
        if ticker not in df.ticker.values:
            try:
                query_document_urls(ticker, key, url_file)
                key_counter += 1
            except:
                print(f'Cannot fetch urls for {ticker}.')
        
    update_api_key(index)


def build_large_url_csv():
    cik_map = get_ticker_cik_map()
    url_file = 'supporting_data/document_urls.csv'
    df = pd.read_csv(url_file, index_col = 0)
    
    counter = 0
    start = 1844
    key_counter = 0

    index, key = get_api_key()
    print(key)

    for ticker in cik_map.keys():
        if key is None:
            return
        counter += 1
        if ticker not in df.ticker.values and counter > start:
            try:
                query_document_urls(ticker, key, url_file)
                key_counter += 1
            except:
                print(f'Cannot fetch urls for {ticker}.')
        
        if key_counter == 100:
            key_counter = 0
            update_api_key(index)
            index, key = get_api_key()
            print(key)


def extract_text(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')

    for doc in soup.find_all('document'):
        if '<type>10-K' in str(doc.decode_contents()):
            return doc.get_text().lower()


def clean_text(text):
    text = re.sub('<[^>]*>', '', text) \
    .replace('\xa0', '') \
    .replace('\n', '') \
    .strip() \
    .lower()
    return re.sub('[^A-Za-z .]+', '', text)


def get_documents(df):
    if 'documents' not in os.listdir():
        os.mkdir('documents')

    path = os.path.join(os.getcwd(), 'documents')

    for index, row in df.iterrows():
        folder = os.path.join(path, f"{row['ticker']}")
        if row['ticker'] not in os.listdir(path):
            os.mkdir(folder)

        try:
            content = requests.get(row['url']).text
            f = open(f"{folder}/{re.split('/', row['url'])[-1]}", 'w')
            f.write(extract_text(content))
            f.close()
        except requests.exceptions.RequestException as e:
            print(f'Cannot request url for {row["ticker"]}')
        except:
            print(f'Cannot process text file')
            if os.stat(f"{folder}/{re.split('/', row['url'])[-1]}").st_size == 0:
                os.remove(f"{folder}/{re.split('/', row['url'])[-1]}")


def query_document_urls(ticker, key, output_file):
    cik_map = get_ticker_cik_map()

    # Query API: https://api.sec-api.io
    # Streaming API: https://api.sec-api.io:3334/all-filings
    base_url = f'https://api.sec-api.io?token={key}'

    #start_date = '2016-01-01'
    start_date = '1999-01-01'
    end_date = str(date.today())

    search_query = f'cik:{cik_map[ticker]} AND filedAt:{{{start_date} TO {end_date}}} AND formType:\"10-K\"'

    body = {
        "query": { "query_string": { "query": search_query } },
        "from": "0",
        "size": "25",
        "sort": [{ "filedAt": { "order": "desc" } }]
    }

    body = json.dumps(body).encode('utf-8')

    header = {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Length': f'{len(body)}'
    }

    response = requests.post(
        base_url, 
        data = body,
        headers = header)

    urls = []

    df = pd.read_csv(output_file, index_col = 0)

    if ticker not in df.ticker.values:
        for filing in response.json()['filings']:
            urls.append(filing['linkToTxt'])
            df = df.append({'ticker': ticker, 'date': filing['filedAt'], 'url': filing['linkToTxt']}, ignore_index = True)
        df.to_csv(output_file)
    
    else:
        for filing in response.json()['filings']:
            urls.append(filing['linkToTxt'])
    
    return urls


def get_ticker_cik_map():
    # https://www.sec.gov/Archives/edgar/cik-lookup-data.txt
    # https://www.sec.gov/include/ticker.txt
    cik_map = {}

    if 'ticker_cik_map.txt' not in os.listdir():
        url = 'https://www.sec.gov/include/ticker.txt'
        response = requests.get(url)

        with open('ticker_cik_map.txt', 'w') as f:
            f.write(response.text)

    with open('ticker_cik_map.txt', 'r') as f:
        for line in f:
            (ticker, cik) = line.split()
            cik_map[ticker] = cik

    return cik_map


# This is why websites should use captchas and 
# and require emails to be verified first
def generate_api_keys(amount):
    # you might need to change the driver and provide the path to the executable
    driver = webdriver.Safari()
    df = pd.read_csv('supporting_data/api_keys.csv', index_col = 0)
    url = 'https://sec-api.io/register'

    for _ in range(0, amount):
        try:
            driver.get(url)
            driver.maximize_window()

            letters = string.ascii_lowercase
            email = f'{"".join(random.choice(letters) for i in range(10))}@domain.tld'
            passcode = "".join(random.choice(letters) for i in range(10))

            driver.find_element_by_xpath('//*[@id="email"]').send_keys(email)
            sleep(1)
            driver.find_element_by_xpath('//*[@id="password"]').send_keys(passcode)
            sleep(1)
            driver.find_element_by_xpath('//*[@id="root"]/section/main/div/div[2]/div/form/div[3]/button').click()
            sleep(3)
            api_key = driver.find_element_by_xpath('//*[@id="root"]/div/div[3]/div/div/div[2]/div/div/table/tbody/tr[2]/td/code').text
            sleep(1)
            df = df.append({'email':email, 'passcode':passcode, 'apikey':api_key, 'capped':False}, ignore_index = True)
        except:
            pass

    df.to_csv('supporting_data/api_keys.csv')
    driver.close()


def get_api_key():
    df = pd.read_csv('supporting_data/api_keys.csv', index_col = 0)

    for index, row in df.iterrows():
        if not row['capped']:
            return index, row['apikey']


def update_api_key(index):
    df = pd.read_csv('supporting_data/api_keys.csv', index_col = 0)
    df.loc[index, 'capped'] = True
    df.to_csv('supporting_data/api_keys.csv')


if __name__ == '__main__':
    main()