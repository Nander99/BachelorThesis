import os
from collections import Counter
from tabulate import tabulate
import matplotlib.pyplot as plt

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

                    data_list.append((word, pos_tag, lemma))

    return data_list

# Call the read_letters function
letters = read_letters()

# Extract second items from each tuple
tags = [letter[1] for letter in letters]

# Use Counter to count occurrences of each tag
tag_counts = Counter(tags)

# Filter tags with count >= 100
filtered_tag_counts = {tag: count for tag, count in tag_counts.items()  if count >= 100}

# Sort the filtered tag counts in descending order
sorted_tag_counts = sorted(filtered_tag_counts.items(), key=lambda x: x[1], reverse=True)

# Create a list of lists for tabulate
table_data = [["Tag", "Count"]]
table_data.extend(sorted_tag_counts)

# Print the table using tabulate
table_str = tabulate(table_data, headers="firstrow", tablefmt="grid")

# Create a plot with a table
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

# Save the figure as an image
fig.savefig('tag_counts_table.png', bbox_inches='tight', pad_inches=0.1)

# Display the table
print(table_str)

# Extract the data for plotting
tags, counts = zip(*sorted_tag_counts)

# Plot a bar chart
plt.figure(figsize=(10, 6))
plt.bar(tags, counts, color='skyblue', edgecolor='black')
plt.title('Aantal POS Tags')
plt.xlabel('POS Tags')
plt.ylabel('Aantal')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('tag_counts_plot.png')
plt.show()
