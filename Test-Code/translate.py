from pickle import load
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pickle import dump

def combine_phrases(word_list):
    combined_list = []
    for sublist in word_list:
        if len(sublist) == 1:
            combined_list.append(sublist[0])
        else:
            combined_list.append(' '.join(sublist))
    return combined_list

# save a list of clean words to file
def save_clean_data(data, filename):
    dump(data, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load a clean dataset
def load_clean_dataset(filename):
    return load(open(filename, 'rb'))
 
# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)
 
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
    
        target.append(word)

    return ' '.join(target)
 
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    translation_list = []
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_src, raw_target, raw_id = raw_dataset[i]
        print(f'Evaluation: {i + 1}/{len(sources)}')
        # print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append([raw_target.split()])
        predicted.append(translation.split())
        translation_list.append([raw_src, raw_target, translation, raw_id])


    # Flatten nested lists
    actual_list = [word for act in actual for word in act]
    actual_flat = combine_phrases(actual_list)
    predicted_flat = combine_phrases(predicted)

    # calculate evaluation metrics
    accuracy = accuracy_score(actual_flat, predicted_flat)
    precision = precision_score(actual_flat, predicted_flat, average='weighted')
    recall = recall_score(actual_flat, predicted_flat, average='weighted')
    f1 = f1_score(actual_flat, predicted_flat, average='weighted')

    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1-score: %f' % f1)

    return translation_list

 
# load datasets
dataset = load_clean_dataset('Translation-Pickle-Files/old-new.pkl')
test = load_clean_dataset('Test-Data/prepared-geo-data-test.pkl')

# prepare old tokenizer
old_tokenizer = create_tokenizer(dataset[:, 0])
old_vocab_size = len(old_tokenizer.word_index) + 1
old_length = max_length(dataset[:, 0])

# prepare new tokenizer
new_tokenizer = create_tokenizer(dataset[:, 1])
new_vocab_size = len(new_tokenizer.word_index) + 1
new_length = max_length(dataset[:, 1])

# prepare data
testX = encode_sequences(old_tokenizer, old_length, test[:, 0])

# load model
model_name = 'Translation-Model/translation-model-combine-neloc.h5'
model = load_model(model_name)

# test on some test sequences
print('test')
translations = evaluate_model(model, new_tokenizer, testX, test)
save_clean_data(translations, 'Test-Data/translated-geo-data-test.pkl')
