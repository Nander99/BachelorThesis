import pickle

# Open the pickle file for reading in binary mode
with open('Geo-Identificator-Files\geo_data_train.pkl', 'rb') as file:
    # Load the contents of the pickle file
    data = pickle.load(file)

# Open a text file for writing
with open('output.txt', 'w', encoding='utf-8') as txt_file:
    # Write each item in the data to the text file
    for item in data:
        txt_file.write(str(item) + '\n')
