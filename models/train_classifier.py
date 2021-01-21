import sys
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    Oversample the miner categories using synonyms.

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

def build_model():
    # Create pipeline with Classifier
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
    ])

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
    if len(sys.argv) == 1:
        database_filepath = '../data/disaster_data.db'
        model_filepath = '../models/classifier.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()






