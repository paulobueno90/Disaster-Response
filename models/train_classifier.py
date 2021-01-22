import sys
import os
from os.path import isfile, join, isdir
import warnings

import nltk
import numpy as np
import pandas as pd
import pickle
import re
from random import randint

from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from textaugment import Wordnet


def synonym_tokenize(text):
    """
    Takes a text and return tokenize synonyms
    """
    t = Wordnet(v=True, n=True, p=0.7)
    return [t.augment(word) for word in tokenize(text)]


def oversample_nlp(dataframe):
    """
    Oversample the minor categories using synonyms.

    :param dataframe: pandas format dataframe to be oversampled
    :return: oversampled dataframe
    """
    obj = dataframe.mean()
    obj = obj.sort_values(ascending=False)

    columns_perc = {}

    new_dataframe = pd.DataFrame(columns=dataframe.columns)

    for key, value in obj.iteritems():
        if key not in ['id', 'message', 'original', 'genre']:
            columns_perc[key] = round(value * 100, 1)

    for column in dataframe.columns:

        if column not in ['id', 'message', 'original', 'genre']:

            if columns_perc[column] < 10:

                # Selecting class slice
                feature = column
                mask = dataframe[feature] == 1
                df_temp = dataframe.loc[mask].copy()

                rows_target = int(dataframe.shape[0] * 0.10) - df_temp.shape[0]

                obj_temp = df_temp.mean()
                obj_temp = obj_temp.sort_values(ascending=False)

                # Filtering data
                for key, value in obj_temp.iteritems():
                    if key not in ['id', 'message', 'original', 'genre']:
                        if (value * 100) > 50:
                            if value * 100 == 100:
                                pass
                            else:
                                mask = df_temp[key] == 0
                                df_temp = df_temp.loc[mask]

                df_temp.reset_index(inplace=True, drop=True)
                new_data = pd.DataFrame(columns=df_temp.columns)

                if df_temp.shape[0] > 0:

                    for _ in range(rows_target):
                        row_number = randint(0, df_temp.shape[0] - 1)

                        new_row = df_temp.iloc[row_number]
                        transformed_msg = synonym_tokenize(new_row['message'])
                        transformed_msg = ' '.join(transformed_msg)
                        new_row[0:]['message'] = transformed_msg
                        new_row[0:]['id'] = 'Artificial Data'
                        new_data = new_data.append(new_row)

                    print(f"{column} ADDED: {new_data.shape[0]}")

                    new_dataframe = new_dataframe.append(new_data)

    dataframe = dataframe.append(new_dataframe)

    return dataframe


def tokenize(text):
    """
    Takes text and return a tokenized, cleaned and lemmatized list
    :param
    text: string. input text
    :return:
    clean_tokens: list. List of relevant words lemmatized
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stopwords.words('english')]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def load_data(database_filepath):
    """
    Load data from a SQL db, oversample the minor categories and return data ready to model.
    :param database_filepath:
    :return:
    X: Array. Array with messages to use as input
    Y: Array. Array containing labels
    categories: List. List of strings containing names for each column of 'Y'
    """
    # load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('message', con=engine)

    print("Oversampling Data...")
    df = oversample_nlp(df)

    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    for column in Y.columns:
        Y[column] = pd.to_numeric(Y[column])
    categories = list(Y.columns)

    return X, Y, categories

def performance_score(y_true, y_pred):
    """
    Calculate median F1 score for all of the output classifiers

    Args:
    y_true: Array containing real labels.
    y_pred: Array containing model predictied labels.

    Returns:
    score: float. Median F1 score for all of the output classifiers
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)

    score = np.median(f1_list)
    return score

def build_model(gridsearch=False):
    """
    Create a classification model

    :param gridsearch: bool. Activate GridSearch, default = False
    :return: pipeline
    """


    # Create pipeline with Classifier
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
    ])

    if gridsearch:

        parameters = {'clf__estimator__min_samples_leaf': [1, 2, 5, 10],
                      'clf__estimator__min_samples_split': [2, 5, 10, 15, 100]}

        scorer = make_scorer(performance_score)

        cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, verbose=10)

        return cv

    else:
        return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    Calculate and print evaluation metrics for ML model

        Args:
        model: Pipeline. Model for prediction
        X_test: array. Array containing inputs
        Y_test: array. Array containing actual labels.
        category_names: list of strings. List containing names for each of the predicted fields.

        Returns: None
    """
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics for each set of labels
    for i, col in enumerate(category_names):
        print(f'{"_" * 80}\n{col.upper():^70}\n')
        print(classification_report(list(y_test.values[:, i]), list(y_pred[:, i])))


def save_model(model, model_filepath):
    """
        Save model to a pickle file
    """
    with open(str(model_filepath), 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Runs script step by step until it saves a new trained model
    :return: None
    """

    messages_filepath = ''
    categories_filepath = ''
    database_filepath = ''
    model_filepath = ''
    models_path = ''

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

    if database_filepath:
        print("Database Path: Ok")
        print(database_filepath)
    if messages_filepath:
        print("Messages Data Path: Ok")
        print(messages_filepath)
    if categories_filepath:
        print("Categories Data Path: Ok")
        print(categories_filepath)
    if model_filepath:
        print("Model Path: Ok")
        print(model_filepath)
    else:
        model_filepath = f"{models_path}/classifier.pkl"

    # Print the system arguments
    try:
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
        
        print('Building model...')

        grid_tag = False

        if grid_tag:
            model = build_model(gridsearch=grid_tag)
        else:
            warnings.warn("\nGridSearch is off due to timeXperformance tradeoff:\n"
                  "Without Gridsearch it takes:\n 11 to 20 minutes to train the model in my machine and 4 minutes in Udacity Machine\n"
                  "With Gridsearch it takes 2 hours and 28 minutes. Gridsearch with the parameters selected improved only 0.13%.\n"
                  "If you want to test with GridSearch change the variable 'grid_tag' and assign True")
            model = build_model(gridsearch=grid_tag)

        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    except:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()






