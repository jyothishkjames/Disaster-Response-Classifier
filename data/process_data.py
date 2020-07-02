import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath, dtype='str')
    # load categories dataset
    categories = pd.read_csv(categories_filepath, dtype='str')

    return messages, categories


def clean_data(messages_df, categories_df):
    # drop the column 'original'
    messages_df.drop(columns=['original'], axis=1, inplace=True)

    # drop duplicates
    messages_df.drop_duplicates(keep='first', inplace=True)
    categories_df.drop_duplicates(keep='first', inplace=True)

    # merge datasets
    df = pd.merge(messages_df, categories_df, on=['id'])

    # create a dataframe of the 36 individual category columns
    categories_df = categories_df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories_df[:1]

    # use this row to extract a list of new column names for categories.
    func = lambda x: x.str.split('-')[0][0]
    category_colnames = row.apply(func)
    category_colnames = list(category_colnames)

    # rename the columns of `categories`
    categories_df.columns = category_colnames

    # convert category values to just numbers 0 or 1
    func = lambda x: x.split('-')[1]
    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].apply(func)
        # convert column from string to numeric
        categories_df[column] = pd.to_numeric(categories_df[column])

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_df], axis=1)

    return df


def save_data(df, database_filepath):
    df.head()
    engine = create_engine('sqlite:///Disaster_Response.db')
    df.to_sql('Disaster_Response_Table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages_df, categories_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages_df, categories_df)

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
