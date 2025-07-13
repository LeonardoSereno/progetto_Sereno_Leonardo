from pandas import DataFrame

def nlp_preprocess(df: DataFrame):
    import nltk, contractions, re, os
    
    from unidecode import unidecode
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    df['title'] = df['title'] + ' ' + df['description']
    df = df.drop(columns=['description'])
    df = df.rename(columns={'title': 'title_desc'})
    # First we need to clean the text
    df['title_desc'] = df['title_desc'].apply(lambda x: x.lower())
    df['title_desc'] = df['title_desc'].apply(lambda x: x.replace('‘', ''))
    df['title_desc'] = df['title_desc'].apply(lambda x: x.replace('’', "'"))
    df['title_desc'] = df['title_desc'].apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))
    df['title_desc'] = df['title_desc'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
    df['title_desc'] = df['title_desc'].apply(lambda x: re.sub(' +', ' ', x))
    df['title_desc'] = df['title_desc'].apply(lambda x: unidecode(x, errors='preserve'))
    
    df['title_desc'] = df['title_desc'].apply(word_tokenize)

    # Remove the stopwords
    stop_words = set(stopwords.words('english'))
    df['title_desc'] = df['title_desc'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    df['title_desc'] = df['title_desc'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # make a single string
    df['title_desc'] = df['title_desc'].apply(lambda x: ' '.join(x))

    # save on csv
    os.makedirs('data/lemmatized', exist_ok=True)
    df.to_csv('data/lemmatized/lemmatized.csv', index=False)

    return df

def main():
    import pandas as pd
    df = pd.read_csv('data/_preprocessed/metadata.csv')
    nlp_preprocess(df)
    
if __name__ == "__main__":
    main()