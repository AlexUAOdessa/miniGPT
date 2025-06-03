import torch
import torch.nn as nn
import time
import os
import json
import re
from collections import Counter
from datetime import datetime


class time_func:
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


def preprocess_text(file_path, min_freq=2):
    print('Предварительная обработка текста')
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    vocab = Counter(words)
    vocab = {word: count for word, count in vocab.items() if count >= min_freq}
    vocab['<UNK>'] = 1
    vocab_size = len(vocab)
    word_to_idx = {word: i for i, word in enumerate(vocab.keys())}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    data = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump({'word_to_idx': word_to_idx,
                  'idx_to_word': idx_to_word}, f, ensure_ascii=False)
    return data, vocab_size, word_to_idx, idx_to_word


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=512, n_heads=4, n_layers=4, dropout=0.1, dim_feedforward=512):
        super(MiniGPT, self).__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(1000, embed_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(
            0).repeat(batch_size, 1)
        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        mask = torch.triu(torch.ones(seq_len, seq_len,
                          device=x.device) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).repeat(batch_size * self.n_heads, 1, 1)
        assert mask.size() == (batch_size * self.n_heads, seq_len, seq_len), \
            f"Mask size {mask.size()} does not match expected {(batch_size * self.n_heads, seq_len, seq_len)}"
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.fc_out(x)


def print_model_info(model, vocab_size):
    total_params = sum(p.numel() for p in model.parameters())
    print("\nИнформация о модели MiniGPT:")
    print(f"Размер словаря: {vocab_size} слов")
    print(f"Размер эмбеддингов: {model.embed_size}")
    print(f"Количество слоёв трансформера: {len(model.layers)}")
    print(f"Количество голов механизма внимания: {model.n_heads}")
    print(
        f"Размер скрытого слоя в прямом распространении: {model.dim_feedforward}")
    print(f"Максимальная длина последовательности: 1000 токенов")
    print(f"Общее количество параметров: {total_params:,}")


def generate_text(model, seed_text, max_length=50, word_to_idx=None, idx_to_word=None, sequence_length=20):
    model.eval()
    words = seed_text.lower().split()
    input_indices = [word_to_idx.get(
        word, word_to_idx['<UNK>']) for word in words]
    input_indices = input_indices[-sequence_length:]
    input_seq = torch.tensor([input_indices], dtype=torch.long).to(device)

    generated_words = words.copy()
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_seq)
            probs = torch.softmax(output[:, -1, :], dim=-1)
            next_word_idx = torch.argmax(probs, dim=-1).item()
            input_indices.append(next_word_idx)
            input_indices = input_indices[-sequence_length:]
            input_seq = torch.tensor(
                [input_indices], dtype=torch.long).to(device)
            generated_words.append(idx_to_word.get(next_word_idx, '<UNK>'))
    return ' '.join(generated_words)


MODEL_CONFIGS = {
    'small': {'embed_size': 512, 'n_heads': 8, 'n_layers': 6, 'dim_feedforward': 2048, 'dropout': 0.1},
    'medium': {'embed_size': 1024, 'n_heads': 8, 'n_layers': 8, 'dim_feedforward': 4096, 'dropout': 0.1},
    'large': {'embed_size': 2048, 'n_heads': 8, 'n_layers': 10, 'dim_feedforward': 8192, 'dropout': 0.1}
}


def load_or_train_model(model_name, vocab_size, sequences, targets):
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['medium'])
    model_path = f'minigpt_model_{model_name}.pth'
    config_path = f'minigpt_config_{model_name}.json'

    model = MiniGPT(
        vocab_size=vocab_size,
        embed_size=config['embed_size'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    sequences_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)

    if os.path.exists(model_path):
        print(f"Обнаружена сохранённая модель: {model_path}")
        user_choice = input(
            "Использовать сохранённую модель? (y/n): ").strip().lower()
        if user_choice == 'y':
            try:
                model.load_state_dict(torch.load(
                    model_path, map_location=device))
                print("Модель загружена.")
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        saved_config = json.load(f)
                    print(f"Конфигурация модели: {saved_config}")
                return model, optimizer, criterion, sequences_tensor, targets_tensor
            except Exception as e:
                print(
                    f"Ошибка при загрузке модели: {e}. Обучаем новую модель.")
        else:
            print("Выбрано обучение новой модели.")
    else:
        print("Сохранённая модель не найдена.")

    print('Обучение модели')
    t1 = time_func()
    model.train()
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(sequences), 32):
            batch_seq = sequences_tensor[i:i+32]
            batch_tgt = targets_tensor[i:i+32]
            assert batch_seq.size() == (min(32, len(sequences)-i), sequence_length), \
                f"Batch size mismatch: got {batch_seq.size()}, expected ({min(32, len(sequences)-i)}, {sequence_length})"
            optimizer.zero_grad()
            output = model(batch_seq)
            loss = criterion(output.view(-1, vocab_size), batch_tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(
            f'Эпоха {epoch+1}/{epochs}, Loss: {total_loss / (len(sequences) // 32)}')

    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)
    print(f"Модель сохранена: {model_path}")
    print(f"Конфигурация сохранена: {config_path}")

    t_end = t1.end_time()
    print(f"Обучение заняло {t_end}")
    return model, optimizer, criterion, sequences_tensor, targets_tensor


print(f'Текущая дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    data, vocab_size, word_to_idx, idx_to_word = preprocess_text('data.txt')
except FileNotFoundError:
    print("Ошибка: Файл data.txt не найден.")
    exit(1)

sequence_length = 32
sequences = [data[i:i+sequence_length]
             for i in range(0, len(data) - sequence_length)]
targets = [data[i+1:i+sequence_length+1]
           for i in range(0, len(data) - sequence_length)]

print("Доступные модели:", ', '.join(MODEL_CONFIGS.keys()))
model_name = input("Выберите модель (small/medium/large): ").strip().lower()
model, optimizer, criterion, sequences_tensor, targets_tensor = load_or_train_model(
    model_name, vocab_size, sequences, targets)

print_model_info(model, vocab_size)

print('\nИспользование miniGPT')
print("Введите Enter для генерации текста с начальным текстом 'Холмс и Ватсон'.")
print("Введите 'exit', 'Exit' или 'EXIT' и нажмите Enter для выхода.")
seed = "Рим гладиатор"

while True:
    user_input = input("Ваш ввод: ").strip()
    if user_input.lower() == 'exit':
        print("Выход из программы.")
        break
    if user_input == "":
        user_input = seed
    try:
        generated = generate_text(model, user_input, max_length=50, word_to_idx=word_to_idx,
                                  idx_to_word=idx_to_word, sequence_length=sequence_length)
        print(f"Сгенерированный текст: {generated}\n")
    except Exception as e:
        print(f"Ошибка при генерации текста: {e}")
