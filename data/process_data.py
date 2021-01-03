import sys
import pandas as pd
import numpy as np



def load_data(messages_filepath, categories_filepath):
    """Load messages data and categories data into pandas dataframe"""
    """drop duplicates then merge them together,return the joint dataframe."""

    messages = pd.read_csv(messages_filepath).drop_duplicates(subset=['id'],keep='first')
    categories = pd.read_csv(categories_filepath).drop_duplicates(subset=['id'],keep='first')
    df = pd.merge(messages,categories,how='left',on='id')
    return df

def clean_data(df):
    """Split categories to category columns, assign relevant column names"""
    """Convert column values to values 0 or 1 based on original value in the string"""
    """Drop the categories column from the data frame, append the category columns with converted values"""

    categories = df['categories'].str.split(pat=';',expand=True)

    row = categories.iloc[0]
    category_colnames = list(row.str.split(pat='-',expand=True)[0])
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the splited string
        categories[column] = categories[column].str.split(pat='-',expand=True)[1].astype('int')

    df = df.drop(['categories'],axis=1)
    df = pd.merge(df, categories, left_index=True, right_index=True)

    return df

def save_data(df, database_filename):
    """Use SQLALCHEMY package to create a database"""
    """Store the data frame in table called 'NewTable'"""

    from sqlalchemy import create_engine
    # database_filename = 'sqlite:///DisasterDatabase.db'
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('NewTable', engine, index=False, if_exists='replace')  


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
