import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''

    :param messages_filepath: path for disaster messages file
    :param categories_filepath: path for categories of messages
    :return: a combined dataframe which includes each messsage and their categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge data sets
    df = pd.merge(messages, categories, on='id')

    return df

def split_categories_to_cols(df):
    '''

    :param df:  dataframe containing disaster messages and their categories separated by ";"
    :return: a dataframe of categories where each category is represented in its own column
    '''

    # split the values in the categories column on ; so that each column becomes a separate column.  Use expand
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:len(x) - 2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    return categories

def categories_to_num(categories):
    '''

    :param categories: dataframe with 36 categories.
    :return: a dataframe where each column contains either 0 or 1 for each category
    '''

    # iterate through each column.  keep last character of each string and convert to number
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    return categories
def clean_data(df):
    '''

    :param df: data frame containing disaster messages and their categories
    :return: a cleaned data frame containing column for each category and removed duplicates
    '''

    # create a df of categories where each category is its own column
    categories = split_categories_to_cols(df)

    # convert category values
    categories = categories_to_num(categories)

    # replace categories column in df with new category columns

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df
    # Convert category values to just numbers 0 or 1
def save_data(df, database_filename):
    '''

    :param df: A dataframe that contains cleaned data for disaster messages and their categories
    :param database_filename: file name to be use by sqllite to save the data
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
