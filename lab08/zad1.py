# b

import nltk
from nltk.tokenize import word_tokenize

with open('article.txt', 'r') as file:
    article = file.read()

tokens = word_tokenize(article)
print("Liczba słów po tokenizacji:", len(tokens))


# c

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
print(stop_words)
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Liczba słów po usunięciu stop-words:", len(filtered_tokens))


# d

additional_stopwords = {'could', 'would', 'also'}
stop_words.update(additional_stopwords)
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Liczba słów po dodaniu dodatkowych stop-words i ich usunięciu:", len(filtered_tokens))


# e

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("Liczba słów po lematyzacji:", len(lemmatized_tokens))

# f

from collections import Counter
import matplotlib.pyplot as plt

word_counts = Counter(lemmatized_tokens)

most_common_words = word_counts.most_common(10)

# Wykres słupkowy
words, counts = zip(*most_common_words)
plt.bar(words, counts)
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.title('10 najczęściej występujących słów')
plt.show()

# g

# from wordcloud import WordCloud

# # Tworzenie chmury tagów
# wordcloud = WordCloud().generate(article)

# # Wyświetlanie chmury tagów
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()