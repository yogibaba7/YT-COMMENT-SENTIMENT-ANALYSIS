import pandas as pd 
import numpy as np 
import os 
import logging
import sys
import re
import yaml
from sklearn.model_selection import train_test_split


# configure logging
logger = logging.getLogger('Data_ingestion_log')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# LOAD PARAMS
def load_params(module_name:str,parameter_name:str):
    with open('params.yaml','r') as f:
        file = yaml.safe_load(f)
    param = file[module_name][parameter_name]
    return param

# LOAD DATA
def load_data(data_path:str)->pd.DataFrame:
    try:
        # read data from url
        df = pd.read_csv(data_path)
        print(df['category'].value_counts())
        logger.debug(f"{df.shape[0]} rows and {df.shape[1]} columns {df.columns} loaded from data")
        logger.debug(f"data loaded successfully from {data_path}")

        return df
    except Exception as e:
        logger.error(f"error occured -> {e}")

# PREPROCESSING
def preprocessing(data:pd.DataFrame)->pd.DataFrame:
    try:
        # drop null values
        data = data.dropna()
        
        # drop rows where comment length is less then 2
        data[data['clean_comment'].apply(lambda x: len(x))<=2]

        # remove urls
        broken_url_pattern =r'https?\s+(www\s+)?(?:[a-zA-Z0-9\-]+\s+){1,3}(com|org|net|gov|edu|nic|tumblr)'
        # Remove broken URLs
        data['clean_comment'] = data['clean_comment'].apply(lambda x: re.sub(broken_url_pattern, '', x))

        # replace new line characters
        data['clean_comment'] = data['clean_comment'].str.replace('\n', ' ', regex=True)
        data['clean_comment'] = data['clean_comment'].str.replace('\t', ' ', regex=True)

        # Remove non-English characters from the 'clean_comment' column , Keeping only standard English letters, digits, and common punctuation
        data['clean_comment'] = data['clean_comment'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(x)))

        data['clean_comment'] = data['clean_comment'].apply(lambda x : x.strip())

        # drop duplicate
        data.drop_duplicates(inplace=True)

        data['category'] = data['category'].map({0:0,1:1,-1:2})

        logger.debug(f"Basic preprocessing applied now the shape of data is {data.shape}")
        
        return data
    
    except Exception as e:
        logger.error(f"error occured -> {e}")

# train test split the data.
def create_train_test_set(data:pd.DataFrame):
    try:
        # train test split
        X_train,X_test,y_train,y_test = train_test_split(data.drop(columns=['category']),data['category'],test_size=load_params('data_ingestion','test_size'),random_state=42,stratify=data['category'])


        logger.debug(f"trainset shape {X_train.shape}")
        logger.debug(f"testingset shape {X_test.shape}")

        return X_train,X_test,y_train,y_test
    
    except Exception as e:
        logger.error(f"error occured -> {e}")

# save data 
def save_data(data:pd.DataFrame,save_path:str,save_name:str):
    try:
        os.makedirs(save_path,exist_ok=True)
        data.to_csv(os.path.join(save_path,save_name),index=False)

        logger.debug(f"data saved in {save_path}")
    except Exception as e:
        logger.error(f"error occured -> {e}")


# def main
def main():
    try:
        # load data
        data = load_data('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        # preprocess data
        data = preprocessing(data)
        # make training and testing set
        X_train,X_test,y_train,y_test = create_train_test_set(data)
        # create save 
        save_path = os.path.join('data','raw')
        # save training data
        save_data(X_train,save_path=save_path,save_name='X_train.csv')
        save_data(y_train,save_path=save_path,save_name='y_train.csv')
        # save testing data
        save_data(X_test,save_path=save_path,save_name='X_test.csv')
        save_data(y_test,save_path=save_path,save_name='y_test.csv')

        logger.debug('DataIngestion Completed.....')
    except Exception as e:
        logger.error(f"DataIngestion Failed")

if __name__=="__main__":
    main()












    




df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()

