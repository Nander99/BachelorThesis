import numpy as np
import pandas as pd
import joblib
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


def train_model_wl(df):
    # train the model
    vectorizer = CountVectorizer()
    X_combined = vectorizer.fit_transform(df['word'] + ' ' + df['lemma'])
    y = df['has_id']
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # do cross validation
    X = X_combined
    y = df['has_id']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    fold = 1
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        print(f'Fold {fold} - Accuracy: {accuracy_scores[-1]}, Precision: {precision_scores[-1]}, Recall: {recall_scores[-1]}, F1-Score: {f1_scores[-1]}')
        fold = fold + 1

    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    print(f'Average Accuracy: {avg_accuracy}')
    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')
    print(f'Average F1-Score: {avg_f1}')

    # Save the trained model
    joblib.dump(model, 'Geo-Identificator-Model/geo_id_predictor_wl.pkl')

    # Save the CountVectorizer along with its vocabulary
    joblib.dump(vectorizer, 'Geo-Identificator-Model/count_vectorizer_wl.pkl')


def train_model_w(df):
    # train the model
    vectorizer = CountVectorizer()
    X_combined = vectorizer.fit_transform(df['word'])
    y = df['has_id']
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # do cross validation
    X = X_combined
    y = df['has_id']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    fold = 1
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        print(f'Fold {fold} - Accuracy: {accuracy_scores[-1]}, Precision: {precision_scores[-1]}, Recall: {recall_scores[-1]}, F1-Score: {f1_scores[-1]}')
        fold = fold + 1

    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    print(f'Average Accuracy: {avg_accuracy}')
    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')
    print(f'Average F1-Score: {avg_f1}')

    # Save the trained model
    joblib.dump(model, 'Geo-Identificator-Model/geo_id_predictor_w.pkl')

    # Save the CountVectorizer along with its vocabulary
    joblib.dump(vectorizer, 'Geo-Identificator-Model/count_vectorizer_w.pkl')


def main():
    data_list = load_cleaned_data('Geo-Identificator-Files/geo_data_train.pkl')
    df = create_model_df(data_list)
    print('Training scores only words:')
    train_model_w(df)
    print('\n')
    print('Training scores words + lemma:')
    train_model_wl(df)



if __name__ == "__main__":
    main()
