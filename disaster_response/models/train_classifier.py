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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, classification_report,recall_score
from sklearn.base import BaseEstimator, TransformerMixin

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
    # child_alone is all 0's.  No messages classify child alone so don't include it in ML
    df.drop(columns=['child_alone'], inplace=True)
    # raw messages
    X = df.message
    # categories of given message
    y = df.iloc[:, 4:].values

    return X, y, df.columns.values[4:]

def build_svm_model():
    from sklearn.svm import SVC
    '''
    Build a model to learn categories of disaster messages.  Use pipelines.
    Use tokenize method to get an array of words which is used by CountVectorizer for bag of words
    ML algorithms require numerical values so use TFIDF and feed it into multi-label classifier
    :return: Model to be used for learning
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SVC(class_weight='balanced')))
    ])
    return pipeline
def build_tuned_model():
    '''
       Build a model to learn categories of disaster messages.  Use pipelines.
       Use tokenize method to get an array of words which is used by CountVectorizer for bag of words
       ML algorithms require numerical values so use TFIDF and feed it into multi-label classifier
       The classifier for this method is a result of tuning the model using GridSearchCV.
       :return: Model to be used for learning
       '''
    from sklearn.pipeline import FeatureUnion
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize,ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer())
                 ])),

             ('start_verb', StartingVerbExtractor())
             ])),

         ('clf', MultiOutputClassifier(LogisticRegression(C=0.0464, class_weight='balanced'), n_jobs=-1))
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
def multilabel_recall_score(y_test, y_pred):
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
        score.append(recall_score(y_test[i, :], y_pred[i, :],zero_division=1))
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
    rec_score = multilabel_recall_score(Y_test, Y_pred)
    if debug is True:
        print("average accuracy score:{}".format(auc_score))
        print("average recall score:{}".format(rec_score))

    return auc_score

def tune_model(model, X_train, y_train):
    from sklearn.model_selection import GridSearchCV

    param_grid = [
        {'clf': [MultiOutputClassifier(LogisticRegression(), n_jobs=-1)],
         'clf__estimator__solver': ['lbfgs', 'sag'],
         'clf__estimator__C': np.logspace(-4, 4, 4),
         'clf__estimator__class_weight': [None, 'balanced'],
         'vect__ngram_range': [(1, 1), (1, 2)]},
        {'clf': [MultiOutputClassifier(RandomForestClassifier())],
         'clf__estimator__n_estimators': list(range(100, 301, 100)),
         'vect__ngram_range': [(1, 1), (1, 2)]}
    ]

    cv = GridSearchCV(model, param_grid=param_grid, scoring=make_scorer(multilabel_recall_score),
                      cv=5, verbose=2,n_jobs=-1)
    gs_fit = cv.fit(X_train, y_train)
    df_cv = pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)
    df_cv.to_csv("gridsearch_results.csv")
    print("best parameters={}".format(gs_fit.best_params_))
    return gs_fit
def save_model(model, model_filepath):
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) >= 3:
        database_filepath =  sys.argv[1]
        model_filepath = sys.argv[2]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        print('Building model...')
        model = build_tuned_model()
        # tuning model takes too long.  Initially, try tuning it and afterwards just use the tuned parameters
        if len(sys.argv) == 4 and sys.argv[3] == '--tune_model':
            print('Tuning Model...')
            model = tune_model(model, X_train, Y_train)
        else:
            print('Training model...')
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names,debug=True)



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