from transformers import BertTokenizer
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    """
    Класс для конвертирования Pandas.DataFrame в torch.Dataset
    """
    def __init__(self, df):
        """
        Токенизирует текст датасета
        :param df: исходный датафрейм для токенизации
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = list(df['toxic'])
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
                      for text in df['comment']
                      ]

    def classes(self):
        """
        Метод получения всех строк класса датасета
        :return: классы датасета списком
        """
        return self.labels

    def __len__(self):
        """
        Метод получения длины датасета
        :return: длина датасета
        """
        return len(self.labels)

    def get_batch_labels(self, idx):
        """
        Метод получения опредленного количества строк классов
        :param idx: набор индексов строк
        :return: набор строк классов
        """
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        """
        Метод получения опредленного количества строк текста
        :param idx: набор индексов строк
        :return: набор строк текста
        """
        return self.texts[idx]

    def __getitem__(self, idx):
        """
        Метод получения опредленного количества строк текста и классов
        :param idx: набор индексов строк
        :return: набор строк текста и классов
        """
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y