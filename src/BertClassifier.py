from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    Класс модели-классификатора Bert
    """
    def __init__(self, dropout=0.5):
        """
        Метод инициализации модели с добавлением трёх
        дополнительных слоев: Dropout, Linear, ReLU

        :param dropout: дропаут для борьбы с переобучением и улучшения модели
        """
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()



