import re
import os
from pickle import dump


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


def concatenate_first_elements(list_of_lists):
    return [' '.join(sublist[0] for sublist in list_of_lists)] + [sublist[1:] for sublist in list_of_lists]


def combine_neloc(data_list):
    # Remove special characters from words within lists
    cleaned_lists = [[re.sub(r'[^\w\s]', '', word1), pos_tag, re.sub(r'[^\w\s]', '', word2), id] for [word1, pos_tag, word2, id] in data_list]
    result_list = []
    current_group = []
    for sublist in cleaned_lists:
        tag = sublist[1]
        geo_id = sublist[-1]
        if not current_group or geo_id == current_group[-1][-1] and tag == 'neloc' and geo_id != '0':
            current_group.append(sublist)
        else:
            result_list.append(current_group)
            current_group = [sublist]

    # Append the last group
    if current_group:
        result_list.append(current_group)

    result = [concatenate_first_elements(sublist) for sublist in result_list]
    new_list = [sublist[:2] for sublist in result]
    flattened_list_of_lists = [item[:1] + item[1] for item in new_list]

    return flattened_list_of_lists
    


def clean_data(data_list, tags_list):
    # Remove special characters from words within lists
    cleaned_lists = [[re.sub(r'[^\w\s]', '', word1), pos_tag, re.sub(r'[^\w\s]', '', lemma), geo_id] for [word1, pos_tag, lemma, geo_id] in data_list]
    # Filter out lists where the last word is in stop words, word2 is "unresolved" or contains "_"
    dutch_stop_words = ["aan","aangaande","aangezien","achte","achter","achterna","af","afgelopen","al","aldaar","aldus","alhoewel","alias","alle","allebei","alleen","alles","als","alsnog","altijd","altoos","ander","andere","anders","anderszins","beetje","behalve","behoudens","beide","beiden","ben","beneden","bent","bepaald","betreffende","bij","bijna","bijv","binnen","binnenin","blijkbaar","blijken","boven","bovenal","bovendien","bovengenoemd","bovenstaand","bovenvermeld","buiten","bv","daar","daardoor","daarheen","daarin","daarna","daarnet","daarom","daarop","daaruit","daarvanlangs","dan","dat","de","deden","deed","der","derde","derhalve","dertig","deze","dhr","die","dikwijls","dit","doch","doe","doen","doet","door","doorgaand","drie","duizend","dus","echter","een","eens","eer","eerdat","eerder","eerlang","eerst","eerste","eigen","eigenlijk","elk","elke","en","enig","enige","enigszins","enkel","er","erdoor","erg","ergens","etc","etcetera","even","eveneens","evenwel","gauw","ge","gedurende","geen","gehad","gekund","geleden","gelijk","gemoeten","gemogen","genoeg","geweest","gewoon","gewoonweg","haar","haarzelf","had","hadden","hare","heb","hebben","hebt","hedden","heeft","heel","hem","hemzelf","hen","het","hetzelfde","hier","hierbeneden","hierboven","hierin","hierna","hierom","hij","hijzelf","hoe","hoewel","honderd","hun","hunne","ieder","iedere","iedereen","iemand","iets","ik","ikzelf","in","inderdaad","inmiddels","intussen","inzake","is","ja","je","jezelf","jij","jijzelf","jou","jouw","jouwe","juist","jullie","kan","klaar","kon","konden","krachtens","kun","kunnen","kunt","laatst","later","liever","lijken","lijkt","maak","maakt","maakte","maakten","maar","mag","maken","me","meer","meest","meestal","men","met","mevr","mezelf","mij","mijn","mijnent","mijner","mijzelf","minder","miss","misschien","missen","mits","mocht","mochten","moest","moesten","moet","moeten","mogen","mr","mrs","mw","na","naar","nadat","nam","namelijk","nee","neem","negen","nemen","nergens","net","niemand","niet","niets","niks","noch","nochtans","nog","nogal","nooit","nu","nv","of","ofschoon","om","omdat","omhoog","omlaag","omstreeks","omtrent","omver","ondanks","onder","ondertussen","ongeveer","ons","onszelf","onze","onzeker","ooit","ook","op","opnieuw","opzij","over","overal","overeind","overige","overigens","paar","pas","per","precies","recent","redelijk","reeds","rond","rondom","samen","sedert","sinds","sindsdien","slechts","sommige","spoedig","steeds","tamelijk","te","tegen","tegenover","tenzij","terwijl","thans","tien","tiende","tijdens","tja","toch","toe","toen","toenmaals","toenmalig","tot","totdat","tussen","twee","tweede","u","uit","uitgezonderd","uw","vaak","vaakwat","van","vanaf","vandaan","vanuit","vanwege","veel","veeleer","veertig","verder","verscheidene","verschillende","vervolgens","via","vier","vierde","vijf","vijfde","vijftig","vol","volgend","volgens","voor","vooraf","vooral","vooralsnog","voorbij","voordat","voordezen","voordien","voorheen","voorop","voorts","vooruit","vrij","vroeg","waar","waarom","waarschijnlijk","wanneer","want","waren","was","wat","we","wederom","weer","weg","wegens","weinig","wel","weldra","welk","welke","werd","werden","werder","wezen","whatever","wie","wiens","wier","wij","wijzelf","wil","wilden","willen","word","worden","wordt","zal","ze","zei","zeker","zelf","zelfde","zelfs","zes","zeven","zich","zichzelf","zij","zijn","zijne","zijzelf","zo","zoals","zodat","zodra","zonder","zou","zouden","zowat","zulk","zulke","zullen","zult"]
    useless_words = ["unresolved", 'ul', 'ue']
    filtered_list = [list_item for list_item in cleaned_lists if list_item[-2] not in dutch_stop_words and list_item[0] and list_item[-2] not in useless_words and "_" not in list_item[-2] and list_item[1] in tags_list]

    return filtered_list


# save a list of clean words to file
def save_clean_data(data, filename):
    dump(data, open(filename, 'wb'))
    print('Saved: %s' % filename)


data_list = read_letters()
# All potential usefull tags ("num", "neorg", "res", "prn", "foreign", "neother", "con", "art", "adj", "hs", "neloc", "int", "adv", "adp", "vrn", "neper", "vrb", "nou")
tags_list = ["neorg", "res", "prn", "foreign", "neother", "hs", "neloc", "vrn", "neper", "nou"]
combine_neloc_data = combine_neloc(data_list)
geo_data = clean_data(combine_neloc_data, tags_list)
save_clean_data(geo_data, 'Geo-Identificator-Files/geo_data.pkl')
