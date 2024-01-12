from pickle import load, dump
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
train_dataset = load_clean_dataset('Geo-Identificator-Files/geo_data_test.pkl')
test_dataset = load_clean_dataset('Geo-Identificator-Files/geo_data_train.pkl')
df_raw = create_model_df(raw_dataset)
df_train = create_model_df(train_dataset)
df_test = create_model_df(test_dataset)


# Check distribution in df_raw
distribution_raw = df_raw['has_id'].value_counts(normalize=True)
print("Distribution in df_raw:")
print(distribution_raw)
print("Percentage of no geo-id:", distribution_raw[0] * 100)
print("Percentage of geo-id:", distribution_raw[1] * 100)

# Check distribution in df_train
distribution_train = df_train['has_id'].value_counts(normalize=True)
print("\nDistribution in df_train:")
print(distribution_train)
print("Percentage of no geo-id:", distribution_train[0] * 100)
print("Percentage of geo-id:", distribution_train[1] * 100)

# Check distribution in df_test
distribution_test = df_test['has_id'].value_counts(normalize=True)
print("\nDistribution in df_test:")
print(distribution_test)
print("Percentage of no geo-id:", distribution_test[0] * 100)
print("Percentage of geo-id:", distribution_test[1] * 100)
