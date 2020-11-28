import sys
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
import sqlite3
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from custom_transformer import tokenize
from custom_transformer import StartingVerbExtractor
def load_data(database_filepath):
    '''

    :param database_filepath: file path of database file containing processed and clean
                              disaster response data

    :return: X - array of raw disaster messages
             y - an array of binary values (0 or 1) indicating categories of a given message.
                 A message may belong to multiple categories
             categories - array of category names
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    # create a data frame from table
    df = pd.read_sql_table('DisasterResponse', con=engine)
    # raw messages
    X = df.message
    # categories of given message
    y = df.iloc[:, 4:].values

    return X, y, df.columns.values[4:]

def build_model():
    '''
    Build a model to learn categories of disaster messages.  Use pipelines.
    Use tokenize method to get an array of words which is used by CountVectorizer for bag of words
    ML algorithms require numerical values so use TFIDF and feed it into multi-label classifier
    :return: Model to be used for learning
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def build_tuned_model():
    '''
       Build a model to learn categories of disaster messages.  Use pipelines.
       Use tokenize method to get an array of words which is used by CountVectorizer for bag of words
       ML algorithms require numerical values so use TFIDF and feed it into multi-label classifier
       :return: Model to be used for learning
       '''
    from sklearn.pipeline import FeatureUnion
    from custom_transformer import TextLengthExtractor
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                 ])),

             ('start_verb', StartingVerbExtractor())
             ])),

         ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=300),n_jobs=-1))
         ])

    return pipeline
def multilabel_auccuracy_score(y_test, y_pred):
    '''
    computes average accuracy score of predicted categories for each disaster message
    Here, we have a multi labels for each message and accuracy score is simply how many
    categories for a message was correctly predicted.  Each
    :param y_test: a 2d array with categories of test disaster messages
    :param y_pred: a 2d array with categories of predicted disaster messages
    :param debug:
    :return:
    '''

    if len(y_test) != len(y_pred):
        print("Error: Length of y_test and y_pred must be same")
        return -1.0
    if isinstance(y_test, np.ndarray) != isinstance(y_pred, np.ndarray):
        print("Error: Type of y_test and y_pred must be numpy array")
        return -1.0

    score = []
    for i in range(0, len(y_test)):
        # multi label score
        score.append(accuracy_score(y_test[i, :], y_pred[i, :]))
    # return average of accuracy_score.  This is overall average score
    return (sum(score) / len(score))
def display_results(y_test,y_pred,categories,debug=False):
    '''

    :param y_test: a 2d array with categories of test disaster messages
    :param y_pred: a 2d array with categories of predicted disaster messaged
    :param categories: array of category names
    :param debug: flag to control print information
    :return: None
    '''
    if debug is False:
        return
    for i in range(0, len(categories)):
        print("-------------------------------------------\n")
        print('***Category: {}\n'.format(categories[i]))
        print(classification_report(y_test[:, i], y_pred[:, i]))

def evaluate_model(model, X_test, Y_test, category_names,debug=False):
    '''

    :param model: model that has been trained
    :param X_test: test disaster messages
    :param Y_test: categories of messages in X_test array.  Use this ground truth to evaluate model
    :param category_names: array of category names for disaster messages
    :return: average accuracy score
    '''
    Y_pred = model.predict(X_test)
    display_results(Y_test, Y_pred, category_names, debug)
    auc_score = multilabel_auccuracy_score(Y_test, Y_pred)
    if debug is True:
        print(auc_score)
    return auc_score

def tune_model(model, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    parameters = {'clf__estimator__n_estimators': [10, 150, 300],
                  'clf__estimator__max_depth': [30, 60, 90, None],
                  'vect__ngram_range': [(1, 1), (2, 2)]
                  }

    cv = GridSearchCV(model, param_grid=parameters, scoring=make_scorer(multilabel_auccuracy_score),
                      cv=5, verbose=2,n_jobs=-1)
    gs_fit = cv.fit(X_train, y_train)
    df_cv = pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)
    df_cv.to_csv("gridsearch_results.csv")
    print(gs_fit.best_params_)
def save_model(model, model_filepath):
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_tuned_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names,debug=True)

        print('Tuning Model...')
        #tune_model(model, X_train, Y_train)

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