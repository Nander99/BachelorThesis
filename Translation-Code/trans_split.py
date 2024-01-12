from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

# load a clean dataset
def load_clean_dataset(filename):
	return load(open(filename, 'rb'))

# save a list of clean words to file
def save_clean_dataset(words, filename):
	dump(words, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_dataset('Translation-Pickle-Files/old-new.pkl')

# random shuffle
shuffle(raw_dataset)
dataset = raw_dataset

# split into train, test, and validation
nr_train = int(len(dataset) * 0.7)
nr_val = int(len(dataset) * 0.2)
nr_test = len(dataset) - nr_train - nr_val
print(nr_train, nr_val, nr_test)

# split into train, validation, and test sets
train, val, test = dataset[:nr_train], dataset[nr_train:(nr_train + nr_val)], dataset[(nr_train + nr_val):]

# save
save_clean_dataset(train, 'Translation-Pickle-Files/old-new-train.pkl')
save_clean_dataset(val, 'Translation-Pickle-Files/old-new-val.pkl')
save_clean_dataset(test, 'Translation-Pickle-Files/old-new-test.pkl')
