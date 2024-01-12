import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def test_model(test_data):
    # Load the trained model
    model_w = joblib.load('Geo-Identificator-Model\geo_id_predictor_w.pkl')
    model_wl = joblib.load('Geo-Identificator-Model\geo_id_predictor_wl.pkl')
    
    # Load the CountVectorizer, including its vocabulary
    vectorizer_w = joblib.load('Geo-Identificator-Model\count_vectorizer_w.pkl')
    vectorizer_wl = joblib.load('Geo-Identificator-Model\count_vectorizer_wl.pkl')
   
    # Prepare the input data for prediction
    predictions = {
        'actual' : [],
        'predicted_w' : [],
        'predicted_wl' : []
    }

    for line in test_data:
        word = line[0]
        pos_tag = line[1]
        lemma = line[2]
        geo_id = int(line[3])

        # Vectorize the input data
        data = [word + ' ' + lemma]
        only_word = [word]
        only_word_vector = vectorizer_w.transform(only_word)
        data_vector = vectorizer_wl.transform(data)

        # Make a prediction
        prediction_w = model_w.predict(only_word_vector)
        prediction_wl = model_wl.predict(data_vector)

        # Add actual values to dictionary
        if geo_id == 0:
            predictions['actual'].append(0)
        else:
            predictions['actual'].append(1)

        # Add prediction values to dictionary
        if prediction_w == 0:
            predictions['predicted_w'].append(0)
        else: 
            predictions['predicted_w'].append(1)
        
        if prediction_wl == 0:
            predictions['predicted_wl'].append(0)
        else: 
            predictions['predicted_wl'].append(1)

    actual = predictions['actual']
    predicted_w = predictions['predicted_w']
    predicted_wl = predictions['predicted_wl']

    # Calculate metrics for word predictions
    accuracy_w = accuracy_score(actual, predicted_w)
    precision_w = precision_score(actual, predicted_w)
    recall_w = recall_score(actual, predicted_w)
    f1_w = f1_score(actual, predicted_w)

    # Calculate metrics for word + lemma predictions
    accuracy_wl = accuracy_score(actual, predicted_wl)
    precision_wl = precision_score(actual, predicted_wl)
    recall_wl = recall_score(actual, predicted_wl)
    f1_wl = f1_score(actual, predicted_wl)

    # Print the results
    print("Word Predictions:")
    print(f"Accuracy: {accuracy_w}")
    print(f"Precision: {precision_w}")
    print(f"Recall: {recall_w}")
    print(f"F1 Score: {f1_w}")

    print("\nWord + Lemma Predictions:")
    print(f"Accuracy: {accuracy_wl}")
    print(f"Precision: {precision_wl}")
    print(f"Recall: {recall_wl}")
    print(f"F1 Score: {f1_wl}")


def main():
    test_data = joblib.load('Geo-Identificator-Files/geo_data_test.pkl')
    test_model(test_data)


if __name__ == "__main__":
    main()
