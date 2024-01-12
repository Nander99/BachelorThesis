from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
 
# load a clean dataset
def load_clean_dataset(filename):
    return load(open(filename, 'rb'))
 
# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
# max length
def max_length(lines):
    return max(len(line.split()) for line in lines)
 
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X
 
# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y
 
# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


# Function to load and preprocess data in batches
def data_generator(data, old_tokenizer, new_tokenizer, old_length, new_length, batch_size, vocab_size):
    num_samples = len(data)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_data = data[offset:offset+batch_size]
            X = encode_sequences(old_tokenizer, old_length, batch_data[:, 0])
            Y = encode_sequences(new_tokenizer, new_length, batch_data[:, 1])
            Y = encode_output(Y, vocab_size)
            yield X, Y


# load datasets
dataset = load_clean_dataset('Translation-Pickle-Files/old-new.pkl')
train = load_clean_dataset('Translation-Pickle-Files/old-new-train.pkl')
val = load_clean_dataset('Translation-Pickle-Files/old-new-val.pkl')
 
# prepare old tokenizer
old_tokenizer = create_tokenizer(dataset[:, 0])
old_vocab_size = len(old_tokenizer.word_index) + 1
old_length = max_length(dataset[:, 0])
print('Old Vocabulary Size: %d' % old_vocab_size)
print('Old Max Length: %d' % (old_length))
# prepare New tokenizer
new_tokenizer = create_tokenizer(dataset[:, 1])
new_vocab_size = len(new_tokenizer.word_index) + 1
new_length = max_length(dataset[:, 1])
print('New Vocabulary Size: %d' % new_vocab_size)
print('New Max Length: %d' % (new_length))
 
# define model
model = define_model(old_vocab_size, new_vocab_size, old_length, new_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='Pictures/model.png', show_shapes=True)
# # Fit model using the generator
batch_size = 32
steps_per_epoch = len(train) // batch_size
validation_steps = len(val) // batch_size

filename = 'Translation-Model/translation-model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(
    data_generator(train, old_tokenizer, new_tokenizer, old_length, new_length, batch_size, new_vocab_size),
    epochs=32,
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(val, old_tokenizer, new_tokenizer, old_length, new_length, batch_size, new_vocab_size),
    validation_steps=validation_steps,
    callbacks=[checkpoint],
    verbose=2
)


