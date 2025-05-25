import torch
import torch.nn as nn
import math
from collections import Counter
import time
from datetime import datetime
import os


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
print(f'Текущая дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# Загрузка исходного текста
print('Загрузка исходного текста')
try:
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Ошибка: Файл data.txt не найден. Убедитесь, что файл существует.")
    exit(1)

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

# Определение модели MiniGPT

# self, vocab_size, embed_size = 128, n_heads = 4, n_layers = 2, dropout = 0.1, dim_feedforward = 512


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=512, n_heads=4, n_layers=4, dropout=0.1, dim_feedforward=512):
        super(MiniGPT, self).__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(1000, embed_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                embed_size, n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
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

# Функция для вывода информации о модели


def print_model_info(model, vocab_size):
    total_params = sum(p.numel() for p in model.parameters())
    print("\nИнформация о модели MiniGPT:")
    print(f"Размер словаря: {vocab_size} слов")
    print(f"Размер эмбеддингов: {model.embed_size}")
    print(f"Количество слоёв трансформера: {model.layers.__len__()}")
    print(f"Количество голов механизма внимания: {model.n_heads}")
    print(
        f"Размер скрытого слоя в прямом распространении: {model.dim_feedforward}")
    print(f"Максимальная длина последовательности: 1000 токенов")
    print(f"Общее количество параметров: {total_params:,}")
    print(f"Сложность модели: Упрощённая трансформерная модель с {model.layers.__len__()} слоями и {total_params:,} параметрами, "
          f"подходит для небольших текстовых корпусов, но ограничена по сравнению с полноразмерными GPT-моделями.")

# Функция для генерации текста


def generate_text(model, seed_text, max_length=50, word_to_idx=None, idx_to_word=None):
    model.eval()
    words = seed_text.split()
    input_indices = []
    for word in words:
        if word not in word_to_idx:
            print(
                f"Ошибка: Слово '{word}' отсутствует в словаре. Заменяем на случайное слово.")
            input_indices.append(list(word_to_idx.values())[0])
        else:
            input_indices.append(word_to_idx[word])

    input_seq = torch.tensor([input_indices], dtype=torch.long)
    for _ in range(max_length):
        output = model(input_seq)
        probs = torch.softmax(output[:, -1, :], dim=-1)
        # Используем выбор по вероятностям вместо argmax
        next_word_idx = torch.multinomial(probs, 1).item()
        input_seq = torch.cat(
            [input_seq, torch.tensor([[next_word_idx]])], dim=1)
        words.append(idx_to_word[next_word_idx])
    return ' '.join(words)


# Путь для сохранения модели
MODEL_PATH = 'minigpt_model.pth'

# Проверка наличия сохранённой модели и выбор действия
# vocab_size = vocab_size, embed_size = 128,n_heads = 4, n_layers = 2, dim_feedforward = 51

"""
Размер эмбеддингов: 512 (можно настроить через параметр embed_size).
Количество слоёв: 4 трансформерных декодерных слоя (параметр n_layers).
Количество голов внимания: 4 (параметр n_heads).
Скрытый слой в прямом распространении: 512 (параметр dim_feedforward).
Dropout: 0.1 для регуляризации.
"""


def load_or_train_model():
    model = MiniGPT(vocab_size=vocab_size, embed_size=512,
                    n_heads=4, n_layers=4, dim_feedforward=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    if os.path.exists(MODEL_PATH):
        print(f"Обнаружена сохранённая модель по пути: {MODEL_PATH}")
        user_choice = input(
            "Хотите использовать сохранённую модель? (y/n): ").strip().lower()
        if user_choice == 'y' or user_choice == 'Y':
            try:
                model.load_state_dict(torch.load(MODEL_PATH))
                print("Сохранённая модель загружена.")
                return model, optimizer, criterion, sequences_tensor, targets_tensor
            except Exception as e:
                print(
                    f"Ошибка при загрузке модели: {e}. Начинаем обучение новой модели.")
        else:
            print("Выбрано обучение новой модели.")
    else:
        print("Сохранённая модель не найдена.")

    # Обучение модели, только если выбрано обучение или модель не загружена
    print('Обучение модели')
    t1 = time_func()
    model.train()
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(sequences), 32):
            batch_seq = sequences_tensor[i:i+32]
            batch_tgt = targets_tensor[i:i+32]
            optimizer.zero_grad()
            output = model(batch_seq)
            loss = criterion(output.view(-1, vocab_size), batch_tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f'Эпоха {epoch+1} из всего {epochs}, Loss: {total_loss / (len(sequences) // 32)}')

    # Сохранение модели
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена по пути: {MODEL_PATH}")

    t_end = t1.end_time()
    print(f"Обучение заняло {t_end}")
    return model, optimizer, criterion, sequences_tensor, targets_tensor


# Вызов функции для загрузки или обучения модели
model, optimizer, criterion, sequences, targets = load_or_train_model()

# Вывод информации о модели
try:
    print_model_info(model, vocab_size)
except Exception as e:
    print(f"Ошибка вывода информации о модели: {e}")

# Генерация текста
print('Использование miniGPT')
seed = "Холмс и Ватсон"
try:
    seed = "Холмс и Ватсон"
    generated = generate_text(model, seed, max_length=50,
                              word_to_idx=word_to_idx, idx_to_word=idx_to_word)
    print(f"Сгенерированный текст: {generated}")
except Exception as e:
    print(f"Ошибка при генерации текста: {e}")
