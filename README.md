
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


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
