from transformers import BertTokenizer
import torch


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
