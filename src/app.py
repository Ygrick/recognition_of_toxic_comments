import pandas as pd
import numpy as np
from BertClassifier import BertClassifier
from lemmatize import lemmatize
from MyModel import MyModel


print('Обучить модель - 0')
print('Загрузить модель - 1')
input_chose = int(input())

# training model
if input_chose == 0:
    # uploading dataset
    df = pd.read_csv('src/../data/data_train.csv').drop('Unnamed: 0', axis=1)
    df_test = pd.read_csv('src/../data/data_test_public.csv').drop('Unnamed: 0', axis=1)

    # cleaning dataset
    df_test = df.dropna().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    df_test['toxic'] = pd.to_numeric(df_test['toxic'], downcast='integer')
    df['toxic'] = pd.to_numeric(df['toxic'], downcast='integer')

    # lemmatizing dataset
    df_test['comment'] = df_test['comment'].apply(lambda x: ' '.join(lemmatize(x)))
    df['comment'] = df['comment'].apply(lambda x: ' '.join(lemmatize(x)))

    # split dataset
    df_train, df_val = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df))])
    print(len(df_train), len(df_val), len(df_test))

    # initialization of model
    model = MyModel()
    Bert = BertClassifier()
    model.initialization(Bert)
    model.training(df_train, df_val, lr=1e-6, epochs=7)
    model.saving()
    model.evaluate(df_test)

    # для создания датафрейма: [text, class_true, class_prediction, probabilities] (инференс модели)
    # df_inference = inference(df_test.rename(columns={"comment": "text", "toxic": "class_true"}), model)

# uploading model
elif input_chose == 1:
    model = MyModel()
    model.loading()

# manual testing of the model
while True:
    print("Введите предложение: ")
    input_text = input()
    model.predict_bert(' '.join(lemmatize(input_text)))
