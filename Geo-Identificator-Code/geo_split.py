from pickle import load, dump
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# load a clean dataset
def load_clean_dataset(filename):
    return load(open(filename, 'rb'))

# save a list of clean words to file
def save_clean_dataset(words, filename):
    dump(words, open(filename, 'wb'))
    print('Saved: %s' % filename)

def create_model_df(geo_data):
    # Make the dataframe
    columns = ['word', 'pos_tag', 'lemma', 'id']
    df = pd.DataFrame(geo_data, columns=columns)
    df["id"] = df["id"].fillna(0)
    df['id'] = pd.to_numeric(df['id'])
    df['has_id'] = df['id'].apply(lambda x: 1 if x != 0 else 0)
    df['has_id'] = pd.to_numeric(df['has_id'])
    df = df.dropna()

    return df

# load dataset
raw_dataset = load_clean_dataset('Geo-Identificator-Files/geo_data.pkl')
df = create_model_df(raw_dataset)

# separate features and labels
X = df[['word', 'pos_tag', 'lemma', 'id']]
y = df['has_id']

# shuffle indices
indices = np.random.permutation(len(X))
X = X.iloc[indices]
y = y.iloc[indices]

# stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# save only features (X) without labels (y)
train_features = X_train.values.tolist()
test_features = X_test.values.tolist()

# save
save_clean_dataset(train_features, 'Geo-Identificator-Files/geo_data_train.pkl')
save_clean_dataset(test_features, 'Geo-Identificator-Files/geo_data_test.pkl')
