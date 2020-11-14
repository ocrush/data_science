### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Standard Anaconda distribution of Python 3.6.7.

## Project Motivation<a name="motivation"></a>

For this project, I was interested in analyzing and classifying real messages from disaster events:

1. Descriptive analysis by Extracting, Transforming, and Loading disaster messages data set.
2. How to automatically notify an appropriate agency depending on the message?
3. What can be done to help emergency workers classify disaster messages?


## File Descriptions <a name="files"></a>

There is 2 Python files and Flask web app available here to showcase work related to the above questions.  

#### ETL Pipeline - process_data.py
* Loads the messages and categories dataset
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

#### ML Pipeline - train_classifier.py
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

#### Flask Web App
* Allows emergency worker to enter a disaster message and classify result in several categories
* Provides visualizations of the data using Plotly


## Results<a name="results"></a>


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Please give credit to author and feel free to use the code here as you would like!

