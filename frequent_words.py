import csv

# Read the CSV file into a list of dictionaries
data = []
with open("metrics.csv", mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Sort the data by the 'Frequency' column in descending order
data.sort(key=lambda x: int(x['Frequency']), reverse=True)

# Calculate the average frequency
total_frequency = sum(int(row['Frequency']) for row in data)
average_frequency = total_frequency / len(data)

for row in data:
    row["deviation"] = (int(row["Frequency"]) - average_frequency) ** 2

# Find the 10 most frequent words
most_frequent_words = data[:10]

# Find the 10 least frequent words
least_frequent_words = data[-10:]

# Find the 10 averagely frequent words
averagely_frequent_words = sorted(data, key=lambda n: n["deviation"])[:10]
# Print the results
print("10 Most Frequent Words:")
for word in most_frequent_words:
    print(word)

print("\n10 Averagely Frequent Words:")
for word in averagely_frequent_words:
    print(word)

print("\n10 Least Frequent Words:")
for word in least_frequent_words:
    print(word)
