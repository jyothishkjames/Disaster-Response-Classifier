import json
import sys
import plotly

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib

sys.path.append('../')
from models.train_classifier import *

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_Response_Table', engine)

# load model
model = joblib.load("../models/pickle_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    true_label_count = []
    label_names = []
    false_label_count = []
    for col in df.columns[3:]:
        true_label_count.append(df[col][df[col] == 1].count())
        false_label_count.append(df[col][df[col] == 0].count())
        label_names.append(col)

    # create visuals
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
                    'title': "Message Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=label_names,
                    y=true_label_count
                )
            ],

            'layout': {
                'title': 'Distribution of True Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': -38
                },
                'height': 500
            }
        },

        {
            'data': [
                Bar(
                    x=label_names,
                    y=false_label_count
                )
            ],

            'layout': {
                'title': 'Distribution of False Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': -38
                },
                'height': 500
            }
        }
    ]

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
