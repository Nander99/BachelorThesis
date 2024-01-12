import numpy as np
import pandas as pd
from pickle import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# load a clean dataset
def load_cleaned_data(filename):
    return load(open(filename, 'rb'))


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


# Plot decision tree with word features
def plot_geo_tree_w(df):
    # Train the model
    vectorizer = CountVectorizer()
    X_combined = vectorizer.fit_transform(df['word'])
    y = df['has_id']
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Train a shallow decision tree for visualization
    shallow_model_w = DecisionTreeClassifier(max_depth=5)
    shallow_model_w.fit(X_train, y_train)

    # Plot and save the decision tree for the shallow model
    plt.figure(figsize=(30, 20))
    plot_tree(shallow_model_w, filled=True, feature_names=vectorizer.get_feature_names_out(),
              class_names=['No ID', 'Has ID'])

    # Save the plot as a PNG file
    plt.savefig('model_w_decision_tree_plot.png')

# Plot decision tree with combined word and lemma features
def plot_geo_tree_wl(df):
    # Train the model
    vectorizer = CountVectorizer()
    X_combined = vectorizer.fit_transform(df['word'] + ' ' + df['lemma'])
    y = df['has_id']
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Train a shallow decision tree for visualization
    shallow_model_wl = DecisionTreeClassifier(max_depth=5)
    shallow_model_wl.fit(X_train, y_train)

    # Plot and save the decision tree for the shallow model
    plt.figure(figsize=(30, 20))
    plot_tree(shallow_model_wl, filled=True, feature_names=vectorizer.get_feature_names_out(),
              class_names=['No ID', 'Has ID'])

    # Save the plot as a PNG file
    plt.savefig('model_wl_decision_tree_plot.png')

# Main function
def main():
    data_list = load_cleaned_data('Geo-Identificator-Files/geo_data_train.pkl')
    df = create_model_df(data_list)
    plot_geo_tree_w(df)
    plot_geo_tree_wl(df)

if __name__ == "__main__":
    main()
