import pandas as pd 
import numpy as np 
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from imblearn.over_sampling import SMOTE
from scipy.sparse import save_npz,csr_matrix

# configure logging
logger = logging.getLogger('feature_engineering_logs')
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

# train and save vectorization model
def TrainVector(data:pd.DataFrame,save_path:str)->pd.DataFrame:
    try:
        tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=10000)
        data = tfidf.fit_transform(data['clean_comment'])
        logger.debug('Vector Trained..')
        os.makedirs(save_path,exist_ok=True)
        joblib.dump(tfidf,os.path.join(save_path,'vector.pkl'))
        logger.debug(f"Vector Saved on {save_path}")
        logger.debug('TrainVector Completed..')
        return data
    except Exception as e:
        logger.error(f"Error : {e}")


# apply vectorization
def ApplyVectorization(data:pd.DataFrame,vector_path:str)->pd.DataFrame:
    try:
        vector = joblib.load(vector_path)
        data = vector.transform(data['clean_comment'])
        return data
        logger.debug('Vectorization applied..')
    except Exception as e:
        logger.error(f"Error : {e}")

# apply Resampling on training data
def ApplyResampling(X_train:pd.DataFrame,y_train:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        smt = SMOTE(random_state=42)
        X_train,y_train = smt.fit_resample(X_train,y_train)

        joblib.dump(smt,'models/resampler.pkl')

        logger.debug('Model Saved on models/resampler.pkl')
        logger.debug('Resampling Applied')
        return X_train,y_train
    except Exception as e:
        logger.error(f"Error : {e}")

# save data 
def save_data(data:pd.DataFrame,save_path:str,save_name:str):
    try:
        
        os.makedirs(save_path,exist_ok=True)
        if isinstance(data, pd.DataFrame):
            data.to_csv(os.path.join(save_path, save_name), index=False)
        elif isinstance(data, csr_matrix):
            save_npz(os.path.join(save_path, save_name.replace('.csv', '.npz')), data)
        
        logger.debug(f"data saved in {save_path}")
    except Exception as e:
        logger.error(f"error occured -> {e}")

# main
def main():
    # load training data
    X_train = load_data('data/interim/X_train.csv')
    y_train = load_data('data/interim/y_train.csv')
    # load testing data
    X_test = load_data('data/interim/X_test.csv')
    y_test = load_data('data/interim/y_test.csv')

    # Find indices with null values in 'clean_comment'
    nan_index_train = X_train[X_train['clean_comment'].isnull()].index
    nan_index_test = X_test[X_test['clean_comment'].isnull()].index

    # Drop those indices from both X and y
    X_train.drop(nan_index_train, inplace=True)
    y_train.drop(nan_index_train, inplace=True)

    X_test.drop(nan_index_test, inplace=True)
    y_test.drop(nan_index_test, inplace=True)

    # vectorsave path
    vectorsave_path = os.path.join('models')
    X_train = TrainVector(X_train,vectorsave_path)
    X_test = ApplyVectorization(X_test,'models/vector.pkl')

    # apply resampling
    X_train,y_train = ApplyResampling(X_train,y_train)

    # create save path
    save_path = os.path.join('data','processed')
    # save processed training data
    save_data(X_train,save_path=save_path,save_name='X_train.csv')
    save_data(y_train,save_path=save_path,save_name='y_train.csv')
    
    # save processed testing data
    save_data(X_test,save_path=save_path,save_name='X_test.csv')
    save_data(y_test,save_path=save_path,save_name='y_test.csv')

if __name__=="__main__":
    main()
