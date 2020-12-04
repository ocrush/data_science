import json
import plotly
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Scatter
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
import re
from wordcloud import WordCloud, STOPWORDS
app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        #print("text:{}".format(text))
        pos_tags = nltk.pos_tag(tokenize(text))
        if len(pos_tags) == 0:
            return False
        first_word, first_tag = pos_tags[0]
        if first_tag in ['VB', 'VBP'] or first_word == 'RT':
            return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    '''
    :param text: document to be tokenized
    :return: array of tokens where the original document is reduced by removing punctuation, stop words, lemmatized
    '''
    # convert to lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize text
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.strip())
        clean_tokens.append(clean_tok)
    return clean_tokens

def plotly_category_labels(df):
    '''

    :param df: data frame of disaster messages and their categories.  Each category has
               a 0 or 1
    :return: a plotly graph object displaying stacked bar chart of each category and its classes
             A good visualization to see the imbalance in classes
    '''
    cat_label_counts = df.iloc[:, 4:].apply(pd.value_counts).T
    x = cat_label_counts.index.values
    y_0 = cat_label_counts.iloc[:, 0].values
    y_1 = cat_label_counts.iloc[:, 1].values

    cat_label_graph = {
        'data': [
            Bar(
                name='0',
                x=x,
                y=y_0
            ),
            Bar(
                name='1',
                x = x,
                y = y_1
            )
        ],

        'layout': {
            'title': 'Target labels per categorgy',
            'barmode': 'stack',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Categories"
            }
        }
    }
    return cat_label_graph
def plotly_wordcloud(text,title):
    '''

    :param text: text to be used for word cloud
    :param title: Title for word cloud
    :return: plotly graph object to be returned to the front-end
    '''
    wc = WordCloud(stopwords=STOPWORDS, collocations=False)
    wc.generate(" ".join(text))

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 100)
    new_freq_list

    wc_graph = {
        'data': [
            Scatter(
                x=x,
                y=y,
                textfont=dict(size=fontsize_list,
                              color=color_list),
                mode='text',
                text=word_list
            )
        ],

        'layout': {'title': title,'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}}
    }
    return wc_graph

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
# child_alone is all 0's.  No messages classify child alone so don't include it in ML
df.drop(columns=['child_alone'], inplace=True)
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories = df.columns.values[4:]
    disaster_msg_per_category = df.iloc[:,4:].sum().values

    row_sums = df.iloc[:, 4:].sum(axis=1)
    multi_label_counts = row_sums.value_counts()
    num_labels = multi_label_counts.index
    multi_label_counts = multi_label_counts.values
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [{
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
    }, {
        'data': [
            Bar(
                x=num_labels,
                y=multi_label_counts
            )
        ],

        'layout': {
            'title': 'Messages having multiple labels',
            'yaxis': {
                'title': "Number of Messages"
            },
            'xaxis': {
                'title': "Number of labels"
            }
        }
    }, {
        'data': [
            Bar(
                x=categories,
                y=disaster_msg_per_category
            )
        ],

        'layout': {
            'title': 'Disaster messages in each category',
            'yaxis': {
                'title': "Number of messages"
            },
            'xaxis': {
                'title': "Message Type"
            }
        }
    }, plotly_wordcloud(df[df.aid_related == True].message.values, title='Aid Related'),
        plotly_wordcloud(df[df.weather_related == True].message.values, title='Weather Related')]
    # add class distribution graph to the front
    graphs.insert(0,plotly_category_labels(df))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()