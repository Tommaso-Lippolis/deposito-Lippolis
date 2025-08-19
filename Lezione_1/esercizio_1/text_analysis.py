
def read_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return "file loaded correctly"
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return "file not found"


def count_rows(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        count = 0
        for line in file:
            count += 1
        return count


def count_words(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        count = 0
        for line in file:
            words = line.split()
            count += len(words)
        return count
    

def top_5_occurrences(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        word_count = {}
        for line in file:
            words = line.split()
            for word in words:
                if word.isalnum():
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
        sorted_words = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
        return sorted_words[:5]


file_path = '18_agosto/esercizio_1/prova.txt'

print(read_file(file_path))
rows=count_rows(file_path)
words=count_words(file_path)
top_5=top_5_occurrences(file_path)

print(f"Number of rows: {rows}")
print(f"Number of words: {words}")
print("Top 5 occurrences:", top_5)

with open('18_agosto/esercizio_1/output.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(f"Number of rows: {rows}\n")
    output_file.write(f"Number of words: {words}\n")
    output_file.write("Top 5 occurrences:\n")
    for word, count in top_5:
        output_file.write(f"{word}: {count}\n")

