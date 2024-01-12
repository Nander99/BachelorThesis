import os
import pandas as pd


def read_letters():
    # read all letters in folder and create a list of all letters
    data_list = []
    folder_path = 'brieven'
    exclude_list = {'<doc>', '<file>', '</doc>', '</file>'}

    for filename in os.listdir(folder_path):
        data_file = os.path.join(folder_path, filename)
        with open(data_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines:
                line = line.strip().lower()
                if line not in exclude_list:
                    parts = line.split('\t')

                    if len(parts) < 3:
                        continue

                    word = parts[0]
                    pos_tag = parts[1]
                    id_or_lemma = parts[-1]

                    if id_or_lemma.isdigit():
                        id = id_or_lemma
                        lemma = ' '.join(parts[2:-1])
                    else:
                        id = '0'
                        lemma = ' '.join(parts[2:])

                    data_list.append([word, pos_tag, lemma, id])

    return data_list



data_list = read_letters()
# create pandas dataframe
df = pd.DataFrame(data_list, columns=['Woord', 'Pos-Tag', 'Lemma', 'ID'])

# filter ids not equal to 0
not_zero_ids = df[df['ID'] != '0']

# do calculations
amounts = {
    'Aantal Items': len(df),
    'Aantal Locaties': df[df['Pos-Tag'] == 'neloc']['Woord'].count(),
    'Aantal Unieke Locaties': df[df['Pos-Tag'] == 'neloc']['Woord'].nunique(),
    'Aantal ID\'s': not_zero_ids['ID'].count(),
    'Aantal Unieke ID\'s': not_zero_ids['ID'].nunique()
}

# make a dataframe 
amounts_df = pd.DataFrame.from_dict(amounts, orient='index', columns=['Aantal'])

# show results
print(amounts_df)
