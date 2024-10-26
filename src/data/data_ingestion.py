import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split

#from yaml import data ingestion in test size
def read_yaml(path: str)-> float:
    test_size=yaml.safe_load(open(path,'r'))['data_ingestion']['test_size']
    return test_size


# read data set
def read_data_set(url:str)-> pd.DataFrame:
    df = pd.read_csv(url)
    df.drop(columns=['tweet_id'],inplace=True)
    return df


def processed_data(df:pd.DataFrame)->pd.DataFrame:
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]

    final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
    return final_df


def save_data(df,train,test:pd.DataFrame)->None:
    data_path=os.path.join("data","raw")
    os.makedirs(data_path)
    train.to_csv(os.path.join(data_path,"train_data.csv"), index=False)
    test.to_csv(os.path.join(data_path,"test_data.csv"), index=False)
    
    
def main():
    url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    df = read_data_set(url)
    final_df = processed_data(df)
    test_size = read_yaml('params.yaml')
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    save_data(final_df,train_data,test_data)
    
if __name__ == '__main__':
    main()