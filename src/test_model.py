import torch
import math


def predict_bert(text, model, tokenizer):
    test_text = tokenizer(text, padding='max_length', max_length = 50, truncation=True, return_tensors="pt")
    model.eval()

    with torch.no_grad():
        mask = test_text['attention_mask'].to('cpu')
        input_id = test_text['input_ids'].squeeze(1).to('cpu')
        model.to('cpu')

        output = model(input_id, mask)
        label_of_edu = output.argmax(dim=1).item() # 1 - toxic, 0 - no toxic
        sigmoid = 1 / (1 + math.exp(-output[0][label_of_edu]))

        print(f'Вероятность: {sigmoid: .3f}')
        print(f'КЛАСС: {label_of_edu}')

