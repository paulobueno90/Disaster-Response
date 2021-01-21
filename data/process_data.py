import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):

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
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('message', engine, index=False, if_exists='replace')


def main():

    # Print the system arguments
    print(sys.argv)
    if len(sys.argv) == 1:

        messages_filepath = '../data/disaster_messages.csv'
        categories_filepath = '../data/disaster_categories.csv'
        database_filepath = '../data/disaster_data.db'
        print(messages_filepath)
        print(categories_filepath)

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()