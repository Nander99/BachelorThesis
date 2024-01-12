from pickle import load
from pickle import dump
from numpy import array


# load a clean dataset
def load_clean_dataset(filename):
	return load(open(filename, 'rb'))

# save a list of clean words to file
def save_clean_dataset(words, filename):
	dump(words, open(filename, 'wb'))
	print('Saved: %s' % filename)


def main():
    data = load_clean_dataset('Geo-Identificator-Files/geo_data.pkl')
    neloc_list = [[item[0], item[2], item[-1]] for item in data if item[1] == 'neloc']
    save_clean_dataset(array(neloc_list), 'Test-Data/neloc-data.pkl')


if __name__ == "__main__":
    main()

