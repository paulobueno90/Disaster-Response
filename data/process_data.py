import os
from os.path import isfile, join, isdir
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Messages data and categories data to be handled.

    :param messages_filepath: string.  Path to csv with messages data
    :param categories_filepath: string.  Path to csv with categories data
    :return: Merged Dataframe with messages and categories
    """

    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    """
    Clean data and create categories Columns.

    :param df: Pandas Dataframe. Dataframe Input
    :return: Pandas Dataframe. Dataframe cleaned with categories at your own column
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Extract a list of new column names for categories.
    category_colnames = df['categories'].values[0].replace('-1', '').replace('-0', '').split(';')

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        # checking for values other than 0 and 1 and converting number > 1 to 1
        if len(categories[column].unique()) > 2:
            categories[column] = categories[column].map(lambda x: 1 if x == 2 else x)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    """

    Save cleaned and merged dataframe in a SQL database

    :param df: Pandas Dataframe. Data to save in a SQL database
    :param database_filename: String. String containing the name of the database
    :return: None
    """
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('message', engine, index=False, if_exists='replace')


def main():
    """
    Runs script step by step until it saves data in a SQL db
    :return: None
    """

    messages_filepath = ''
    categories_filepath = ''
    database_filepath = ''
    model_filepath = ''
    data_path = ''

    basedir = [f for f in os.listdir("../") if isdir(join("../", f))]

    if 'data' in basedir:

        data_path = "../data"
        data_files = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]

        for file in data_files:
            if "messages.csv" in file:
                messages_filepath = f"{data_path}/{file}"

            elif "categories.csv" in file:
                categories_filepath = f"{data_path}/{file}"

            elif "data.db" in file:
                database_filepath = f"{data_path}/{file}"

    if 'models' in basedir:

        models_path = "../models"
        models_files = [f for f in os.listdir(models_path) if isfile(join(models_path, f))]

        for file in models_files:

            if "classifier.pkl" in file:
                model_filepath = f"{models_path}/{file}"


    if model_filepath:
        print("Model Path: Ok")
        print(model_filepath)
    if messages_filepath:
        print("Messages Data Path: Ok")
        print(messages_filepath)
    if categories_filepath:
        print("Categories Data Path: Ok")
        print(categories_filepath)
    if database_filepath:
        print("Database Path: Ok")
        print(database_filepath)
    else:
        database_filepath = f"{data_path}/disaster_data.db"

    try:
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    except:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

