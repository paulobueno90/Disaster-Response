import json
import plotly
import pandas as pd
import numpy as np
import pickle
import joblib
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
#import plotly.graph_objs as go
from plotly.graph_objs import Bar, Heatmap

from sqlalchemy import create_engine


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///' + '../data/disaster_data.db')
df = pd.read_sql_table('message', engine)

print(df.head())





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



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ###########################
    # Number of Messages per Category
    Message_counts = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1).sum().sort_values()
    Message_names = list(Message_counts.index)

    ###########################

    # Top ten categories count
    top_category_count = df.iloc[:, 4:].sum().sort_values(ascending=False)[1:11]
    top_category_names = list(top_category_count.index)

    ###########################

    # extract categories
    category_map = df.iloc[:, 4:].corr().values
    category_names = list(df.iloc[:, 4:].columns)

    #############################
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        #############################
        {
            'data': [
                Bar(
                    x=Message_counts,
                    y=Message_names,
                    orientation='h',

                )
            ],

            'layout': {
                'title': 'Number of Messages per Category',

                'xaxis': {
                    'title': "Number of Messages"

                },
            }
        },
        ########################
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_count
                )
            ],

            'layout': {
                'title': 'Top Ten Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        #############################
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=category_map
                )
            ],

            'layout': {
                'title': 'Correlation Heatmap of Categories',
                'xaxis': {'tickangle': -45}
            }
        },
        #############################

        ########################
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, figuresJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# Load the Model back from file
with open('../models/classifier.pkl', 'rb') as file:
    model = pickle.load(file)



if __name__ == '__main__':
    app.run(debug=True)