import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from train_and_eval_model import training, evaluate
from BertClassifier import BertClassifier
from test_model import predict_bert
from lemmatize import lemmatize


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print('Обучить модель - 0')
print('Загрузить модель - 1')
input_chose = int(input())

if input_chose == 0:
    # uploading dataset
    df = pd.read_csv('/../data/data_train.csv').drop('Unnamed: 0', axis=1)
    df_test = pd.read_csv('/../data/data_test_public.csv').drop('Unnamed: 0', axis=1)

    # cleaning dataset
    df_test = df.dropna().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    # lemmatizing dataset
    df_test['comment'] = df_test['comment'].apply(lambda x: ' '.join(lemmatize(x)))
    df['comment'] = df['comment'].apply(lambda x: ' '.join(lemmatize(x)))

    # split dataset
    df_train, df_val = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df))])
    print(len(df_train), len(df_val), len(df_test))

    # initialization of model
    EPOCHS = 7
    model = BertClassifier()
    LR = 1e-6

    training(model, df_train, df_val, LR, EPOCHS)
    torch.save(model, 'src/../models/model_toxic_comment.pt')
    evaluate(model, df_test)

elif input_chose == 1:
    model = torch.load('src/../models/model_toxic_comment.pt', map_location='cpu')
    model.eval()

else:
    pass

# manual testing of the model
while True:
    print("Введите предложение: ")
    predict_bert(input(), model, tokenizer)
