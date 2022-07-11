from Dataset import Dataset
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


def training(model, train_data, val_data, learning_rate, epochs):
    model.train()
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in tqdm(val_dataloader):
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} '
            f'| Train Loss: {total_loss_train / len(train_data): .3f} '
            f'| Train Accuracy: {total_acc_train / len(train_data): .3f} '
            f'| Val Loss: {total_loss_val / len(val_data): .3f} '
            f'| Val Accuracy: {total_acc_val / len(val_data): .3f}')


def f1_score(tp, fp, tn, fn):
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(f'accuracy = {accuracy}')
    precision = tp/(tp+fp)
    print(f'precision = {precision}')
    recall = tp/(tp+fn)
    print(f'recall = {recall}')
    f1 = 2*precision*recall/(precision+recall)
    print(f'f1_score = {f1}')


def evaluate(model, test_data):
    test = Dataset(test_data)
    print(f'len_dataset: {len(test_data)}')
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
    total_acc_test = 0
    model.eval()

    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader):
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            predict = output.argmax(dim=1)
            acc = (predict == test_label).sum().item()
            total_acc_test += acc

            true_positive += ((predict and test_label) + 0).item()
            true_negative += ((not predict and not test_label) + 0)
            false_negative += ((not predict and test_label) + 0)
            false_positive += ((predict and not test_label) + 0)

    print(f'count tp, tn, fn, fp: {true_positive + true_negative + false_negative + false_positive}')
    f1_score(true_positive, false_positive, true_negative, false_negative)
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
