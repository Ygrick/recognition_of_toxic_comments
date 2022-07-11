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

    def forward(self, input_id, mask):
        """
        Метод для прямого распространения ошибки (переопределенный)
        :param input_id: входной номер
        :param mask: маска
        :return:
        """
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


