import torch
from collections import Counter

# Загрузка текста
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Токенизация (простая, на основе слов)
words = text.split()
vocab = Counter(words)
vocab_size = len(vocab)
word_to_idx = {word: i for i, (word, _) in enumerate(vocab.items())}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Преобразование текста в индексы
data = [word_to_idx[word] for word in words]

# Создание последовательностей
sequence_length = 20
sequences = [data[i:i+sequence_length]
             for i in range(0, len(data) - sequence_length)]
targets = [data[i+1:i+sequence_length+1]
           for i in range(0, len(data) - sequence_length)]
