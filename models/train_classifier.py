# import all necessary packages
import sys
import pandas as pd
import numpy as np
import re
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Load data from specified database, generate 3 outputs"""
    """X: features, Y: targets to predict, category_names"""
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('NewTable',engine)

    targets = list(df.columns)[5:]
    X = df['message']
    Y = df[targets]
    category_names = targets
    return X, Y, category_names
    


def tokenize(text):
    """Take each message as input, normalize each text"""
    """Remove stopwords, reduce each word to their stem"""
    """Return a list of strings"""
     # normalize text by eliminating punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)

    # remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    # reduce word to their stem
    # stemmed = [PorterStemmer().stem(w) for w in words]

    # reduce word to their root
    lemmed = [WordNetLemmatizer().lemmatize(w).lower().strip() for w in words]
    
    return lemmed


def build_model():
    """Using pipeline to standardize 2 steps of data prreprocessing steps"""
    """The 3rd step is the classifier"""
    
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tdif',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    """Using grid search to find the best parameter for a classifier"""
    parameters = {'clf__estimator__n_estimators': [10, 20]}
    # create grid search object
    cv = GridSearchCV(estimator=pipeline,param_grid=parameters)
    
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names=category_names)
    print("Classification Report:\n", class_report)
    # accuracy = (Y_pred==Y_test).mean()
    # print("Accuracy:", accuracy)

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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