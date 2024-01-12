from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


actual = []
predicted = []

# Open the pickle file for reading in binary mode
with open('Test-Data/translated-old-new-test-data.pkl', 'rb') as file:
    # Load the contents of the pickle file
    data = pickle.load(file)

    for line in data:
        actual.append(line[1])
        predicted.append(line[2])

   
# calculate evaluation metrics
accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted, average='weighted')
recall = recall_score(actual, predicted, average='weighted')
f1 = f1_score(actual, predicted, average='weighted')

print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1-score: %f' % f1)


