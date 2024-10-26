import numpy as np
import pandas as pd
import yaml

import os

from sklearn.feature_extraction.text import CountVectorizer

def read_yaml(path: str)-> float:
    max_feature=yaml.safe_load(open(path,'r'))['feature']["max_features"]
    return max_feature

def read_data_set(path:str)-> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(train,test,max_feature) -> pd.DataFrame:
    train.fillna('',inplace=True)
    test.fillna('',inplace=True)

    # apply BoW
    X_train = train['content'].values
    y_train = train['sentiment'].values

    X_test = test['content'].values
    y_test = test['sentiment'].values

    # Apply Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=max_feature)

    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_bow.toarray())

    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())

    test_df['label'] = y_test
    
    return train_df, test_df

def save(train,test:pd.DataFrame)->None:
    data_path = os.path.join("data","features")
    os.makedirs(data_path)

    train.to_csv(os.path.join(data_path,"train_bow.csv"))
    test.to_csv(os.path.join(data_path,"test_bow.csv"))


def main():
    
    # Read th yaml File
    max_feature=read_yaml('params.yaml')
    
    # read Data file
    data_path = os.path.join("data","processed")
    train_data = read_data_set(os.path.join(data_path,"train_processed.csv"))
    test_data = read_data_set(os.path.join(data_path,"test_processed.csv"))
    
    
    # pre processed data
    train_df,test_df=preprocess(train_data, test_data, max_feature)

    # store the data inside data/features
    save(train_df,test_df)
    
    
    
if __name__ == "__main__":
    main()