import torch
import torch.nn as nn
import math
from collections import Counter
import time
from datetime import datetime  # Для работы с датой и временем


class time_func:
    """Выводит время выполнения блока между созданием класса и end_time(self)"""

    def __init__(self):
        self.time_begin = time.time()
        self.time_end = self.time_begin
        self.time_rez = ''

    def end_time(self):
        self.time_end = time.time()
        execution_time = self.time_end - self.time_begin
        h_exe, remainder = divmod(execution_time, 3600)
        m_exe, s_exe = divmod(remainder, 60)
        self.time_rez = f'Время выполнения блока программы: {int(h_exe)} часов {int(m_exe)} минут {s_exe:.2f} секунд'
        return self.time_rez


# Указываем текущую дату и время

print('Загрузка исходного текста')
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Токенизация (простая, на основе слов)
words = text.split()
vocab = Counter(words)
vocab_size = len(vocab)
word_to_idx = {word: i for i, (word, _) in enumerate(vocab.items())}
idx_to_word = {i: word for word, i in word_to_idx.items()}
data = [word_to_idx[word] for word in words]
sequence_length = 20
sequences = [data[i:i+sequence_length]
             for i in range(0, len(data) - sequence_length)]
targets = [data[i+1:i+sequence_length+1]
           for i in range(0, len(data) - sequence_length)]


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, n_heads=4, n_layers=2, dropout=0.1):
        super(MiniGPT, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(1000, embed_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                embed_size, n_heads, dim_feedforward=512, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(
            0).repeat(batch_size, 1)
        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, x)  # Само-внимание
        return self.fc_out(x)


print('Обучение модели')
t1 = time_func()
model = MiniGPT(vocab_size=vocab_size, embed_size=128, n_heads=4, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
sequences = torch.tensor(sequences, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

model.train()
for epoch in range(10):
    total_loss = 0
    for i in range(0, len(sequences), 32):
        batch_seq = sequences[i:i+32]
        batch_tgt = targets[i:i+32]
        optimizer.zero_grad()
        output = model(batch_seq)
        loss = criterion(output.view(-1, vocab_size), batch_tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / (len(sequences) // 32)}')

# Вывод информации о модели


def print_model_info(model, vocab_size):
    # Подсчет общего количества параметров
    total_params = sum(p.numel() for p in model.parameters())
    print("\nИнформация о модели MiniGPT:")
    print(f"Размер словаря: {vocab_size} слов")
    print(f"Размер эмбеддингов: {model.embed_size}")
    print(f"Количество слоёв трансформера: {model.layers.__len__()}")
    print(f"Количество голов механизма внимания: {model.layers[0].nhead}")
    print(
        f"Размер скрытого слоя в прямом распространении: {model.layers[0].dim_feedforward}")
    print(f"Максимальная длина последовательности: 1000 токенов")
    print(f"Общее количество параметров: {total_params:,}")
    print(f"Сложность модели: Упрощённая трансформерная модель с {model.layers.__len__()} слоями и {total_params:,} параметрами, "
          f"подходит для небольших текстовых корпусов, но ограничена по сравнению с полноразмерными GPT-моделями.")


def generate_text(model, seed_text, max_length=50):
    model.eval()
    words = seed_text.split()
    input_seq = torch.tensor([[word_to_idx[word]
                             for word in words]], dtype=torch.long)
    for _ in range(max_length):
        output = model(input_seq)
        probs = torch.softmax(output[:, -1, :], dim=-1)
        next_word_idx = torch.argmax(probs, dim=-1).item()
        input_seq = torch.cat(
            [input_seq, torch.tensor([[next_word_idx]])], dim=1)
        words.append(idx_to_word[next_word_idx])
    return ' '.join(words)


# Вызов функции после обучения
print_model_info(model, vocab_size)

t_end = t1.end_time()
print(f"Обучение заняло {t_end}")
print('Использование miniGPT')
seed = "Холмс увлечение расследования"
generated = generate_text(model, seed)
print(generated)
