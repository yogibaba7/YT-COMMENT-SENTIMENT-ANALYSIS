import pandas as pd 
import numpy as np 
import os 
import logging
import sys
import re
import yaml
import nltk
from nltk.corpus import stopwords
# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# configure logging
logger = logging.getLogger('Data_preprocessing_log')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# LOAD DATA
def load_data(data_path:str)->pd.DataFrame:
    try:
        # read data from url
        df = pd.read_csv(data_path)
        
        logger.debug(f"{df.shape[0]} rows and {df.shape[1]} columns {df.columns} loaded from data")
        logger.debug(f"data loaded successfully from {data_path}")

        return df
    except Exception as e:
        logger.error(f"error occured -> {e}")

# remove stop words
def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
        return text
    except Exception as e:
        logger.error(f"Error -> {e}")
        return text
    
# lemmatizations
def lemmatizor(text):
    try:
        lemmatizor = WordNetLemmatizer()
        text = " ".join([lemmatizor.lemmatize(word) for word in text.split()])
        return text
    except Exception as e:
        logger.error(f"Error -> {e}")
        return text



# PREPROCESSING
def preprocessing(data:pd.DataFrame):
    try:
        data['clean_comment'] = data['clean_comment'].apply(remove_stop_words)
        data['clean_comment'] = data['clean_comment'].apply(lemmatizor)
        logger.debug('Preprocessing applied')
        return data
    
    except Exception as e:
        logger.debug("Preprocessing Failed..")
        logger.error(f"Error -> {e}")
        return data


# save data 
def save_data(data:pd.DataFrame,save_path:str,save_name:str):
    try:
        os.makedirs(save_path,exist_ok=True)
        data.to_csv(os.path.join(save_path,save_name),index=False)

        logger.debug(f"data saved in {save_path}")
    except Exception as e:
        logger.error(f"error occured -> {e}")


# main
def main():
    # load training data
    training_data = load_data('data/raw/training_set.csv')
    # load testing data
    testing_data = load_data('data/raw/testing_set.csv')

    training_data = training_data.dropna()
    testing_data = testing_data.dropna()

    # apply preprocessing on training data
    training_data = preprocessing(training_data)
    # apply preprocessing on testing data
    testing_data = preprocessing(testing_data)

    # create save path
    save_path = os.path.join('data','interim')

    # save processed training data
    save_data(training_data,save_path=save_path,save_name='train_processed.csv')
    
    # save processed testing data
    save_data(testing_data,save_path=save_path,save_name='testing_processed.csv')

if __name__=="__main__":
    main()

