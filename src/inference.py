import torch
import math
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer


def inference(df, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    labels = list(df['class_true'])
    texts = [
        tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        for text in df['text']
    ]

    dict_of_output = {'class_prediction': [],
                      'probabilities': []
                      }
    with torch.no_grad():
        for text in tqdm(texts):
            mask = text['attention_mask'].to('cpu')
            input_id = text['input_ids'].squeeze(1).to('cpu')
            model.to('cpu')

            output = model(input_id, mask)

            label_of_edu = output.argmax(dim=1).item()  # 1 - toxic, 0 - no toxic
            sigmoid = 1 / (1 + math.exp(-output[0][label_of_edu]))

            dict_of_output['class_prediction'].append(label_of_edu)
            dict_of_output['probabilities'].append(round(sigmoid, 4))

    return pd.concat([df, pd.DataFrame(data=dict_of_output)], axis=1)