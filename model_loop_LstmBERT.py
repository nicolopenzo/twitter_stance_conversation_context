import numpy as np
import torch as torch
import torch.nn as nn
# standard loops for training, evaluation and testing, or to simply get predictions

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0.0

    for data in data_loader:

        post_text = data['post_text']
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        len_seq = data['len_seq'].to(device)
        pad_vector_seq = data['pad_vector_seq'].to(device)
        seq_real_size = data['seq_size'].to(device)
        labels = data['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            len_seq=len_seq,
            pad_vector_seq=pad_vector_seq,
            seq_real_size=seq_real_size
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels.flatten().long())
        correct_predictions += torch.sum(preds == labels.flatten().long())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return correct_predictions/n_examples, np.mean(losses)

def train_step(model, data, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0.0

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    len_seq = data['len_seq'].to(device)
    pad_vector_seq = data['pad_vector_seq'].to(device)
    seq_real_size = data['seq_size'].to(device)
    labels = data['labels'].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        len_seq=len_seq,
        pad_vector_seq=pad_vector_seq,
        seq_real_size=seq_real_size
    )

    preds = torch.argmax(outputs, dim=1)
    loss = loss_fn(outputs, labels.flatten().long())
    correct_predictions += torch.sum(preds == labels.flatten().long())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return correct_predictions/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0.0
    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            len_seq = data['len_seq'].to(device)
            pad_vector_seq = data['pad_vector_seq'].to(device)
            seq_real_size = data['seq_size'].to(device)
            labels = data['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                len_seq=len_seq,
                pad_vector_seq=pad_vector_seq,
                seq_real_size=seq_real_size
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels.flatten().long())

            correct_predictions += torch.sum(preds == labels.flatten().long())
            losses.append(loss.item())

    return correct_predictions/n_examples, np.mean(losses)


def get_predictions(model, data_loader, device):
    model = model.eval()

    tweet_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            len_seq = data['len_seq'].to(device)
            pad_vector_seq = data['pad_vector_seq'].to(device)
            seq_real_size = data['seq_size'].to(device)
            labels = data['labels'].to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                len_seq=len_seq,
                pad_vector_seq=pad_vector_seq,
                seq_real_size=seq_real_size
            )

            _, preds = torch.max(outputs, dim=1)
            #tweet_texts.extend(post_text)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels.flatten().long())

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return tweet_texts, predictions, prediction_probs, real_values
