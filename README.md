# Disaster Response Pipeline Project
## Installations
Please install python and import following packages and libraries: import nltk, numpy, sklearn, TransformerMixin, pickle, unittest, string, nltk, nltk.tokenize, RegexpTokenizer,sent_tokenize, nltk.stem, pandas, sqlalchemy, re, sklearn, FeatureUnion
You can find below instructions useful:
### Instructions:
   1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  2. Run the following command in the app's directory to run your web app.
    `python run.py`

  3. Go to http://0.0.0.0:3001/

## Project Screenshots:
![image1](https://video.udacity-data.com/topher/2018/September/5b967bef_disaster-response-project1/disaster-response-project1.png)
![image1](https://video.udacity-data.com/topher/2018/September/5b967cda_disaster-response-project2/disaster-response-project2.png)

## Project Motivation
In the Project, there is a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

## File Descriptions
The disaster data is gathered from the "Figure Eight" company

## How to Interact with this project
There two different models (one of them is optimized by using GridSearch) to predict disaster relief.  

## Licensing
Authors, Acknowledgements, etc. You are more tham welcome to use the code. I appreciate if you refer to my github.
