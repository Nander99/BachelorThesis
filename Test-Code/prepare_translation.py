import pickle
from pickle import dump
from numpy import array

# save a list of clean words to file
def save_clean_data(words, filename):
    dump(words, open(filename, 'wb'))
    print('Saved: %s' % filename)

translation_data = []

# Open the pickle file for reading in binary mode
with open('Geo-Identificator-Files/geo_data_test.pkl', 'rb') as file:
    # Load the contents of the pickle file
    data = pickle.load(file)
    for line in data:
        translation_line = [line[0], line[2], line[3]]
        translation_data.append(translation_line)


save_clean_data(array(translation_data), 'Test-Data/prepared-geo-data-test.pkl')
