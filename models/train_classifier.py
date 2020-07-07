# import libraries
import pickle
import sys

import pandas as pd
from sqlalchemy import create_engine
import nltk

nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import re


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    A class used to generate custom features. The custom feature
    is a Starting Verb Extractor which return True if the stating
    sentence is a Verb, returns False otherwise

    """

    @staticmethod
    def starting_verb(text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            try:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except:
                pass
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    Function to load the data from database into dataframe

    INPUT:
    database_filepath - file path of the database

    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='Disaster_Response_Table', con=engine)
    X = df.message.values
    Y = df.drop(columns=['id', 'message', 'genre'], axis=1).values
    category_names = df.drop(columns=['id', 'message', 'genre'], axis=1).columns.values
    return X, Y, category_names


def tokenize(text):
    """
    Function to tokenize the input text

    INPUT:
    text - text to be tokenized

    """

    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text data
    words = word_tokenize(text)
    # remove stop words
    words = [w for w in words if words not in stopwords.words("english")]
    return words


def build_model():
    """
    Function to build the machine learning pipeline

    """

    # build machine learning pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10],
        'clf__estimator__max_depth':[8],
        'clf__estimator__random_state':[42],
        'clf__estimator__class_weight': ['balanced'],
        'clf__estimator__max_features': ['auto'],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1}
        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Function to evaluate the model

    INPUT:
    model - model to be evaluated
    X_test - test features
    Y_test - test labels

    """

    # predict on test data
    Y_pred = model.predict(X_test)
    # Iterating through each column
    for col1, col2 in zip(Y_pred, Y_test):
        print(classification_report(col1, col2))


def save_model(model, model_filepath):
    """
    Function to save the model as a pickle file

    INPUT:
    model - model to be saved
    model_filepath - file path to save the model

    """

    # Save file to the given path
    pkl_filename = model_filepath + "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
