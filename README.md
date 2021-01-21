
[![author](https://img.shields.io/badge/author-Paulo%20Bueno-blue.svg)](https://www.linkedin.com/in/paulo-bueno-06a4b34a/) [![](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-385/) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://www.mit.edu/~amini/LICENSE.md) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/paulobueno90/AirBnB-Udacity-Project/issues)

<p align="center">
   <img src="banner.png" >
</p> 

# Disaster Response Pipeline Project

## Installation

Python Version: 3.8.5
##### Libraries Used

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3
- textaugment


### Files and Process

#### Data:
The dataset is provided by [Appen]("https://appen.com/figure-eight-is-now-appen/") (former Figure Eight):

#### data/disaster_categories.csv: 
36 categories possible for the messages
#### data/disaster_messages.csv: 
Disaster response messages

#### Data Processing: 
##### data/process_data.py: 
ETL (Extract, Transform, and Load) pipeline that clean and process data from a CSV and store data in a SQLite database

#### Machine Learning 
##### models/train_classifier.py
- Oversampling dataset minor categories with synonyms
- It split data into training and test set
- Create an ML pipeline that uses NLTK, multi-output classifier and Random Forest as classifier  
- Predicts message classifications for the 36 categories (multi-output)

Ps: It was not used GridSearch model, once it improved very litle(0.13%) performance with the oversampled data.

### Web App: 
- Web application that classifies messages.

#### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/


### Acknowledgements

Thanks Udacity for the knowledge and opportunity, the reviewers for the comments and suggestions.
