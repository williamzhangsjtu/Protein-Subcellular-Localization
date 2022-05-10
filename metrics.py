import torch
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import pandas as pd

@torch.no_grad()
def evaluate(model, dataloader, DEVICE, threshold=0.5):
    model.eval()
    sample_score, gt, sample_num = {}, {}, {}
    for data, targets, samples, _ in dataloader:
        data = data.to(DEVICE, torch.float)
        outputs = torch.sigmoid(model(data).cpu()).numpy()
        for idx, output in enumerate(outputs):
            sample = samples[idx]
            sample_score[sample] = sample_score.setdefault(
                sample, np.zeros(10)) + output
            sample_num[sample] = sample_num.setdefault(sample, 0) + 1
            gt[sample] = targets[idx]
    preds, labels = [], []
    for k, v in sample_score.items():
        pred = ((v / sample_num[k]) > threshold).astype(np.int8)
        pred[np.argmax(v)] = 1
        preds.append(pred)
        labels.append(gt[k].numpy().astype(np.int8))
    f1_macro = f1_score(np.stack(labels), np.stack(preds), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(preds), average='micro')
    return f1_macro, f1_micro


@torch.no_grad()
def evaluate_multiple(model, dataloader, DEVICE, threshold=0.5):
    model.eval()
    sample_score, gt, sample_num = {}, {}, {}
    for data, targets, samples, _ in dataloader:
        data = data.to(DEVICE, torch.float)
        outputs = torch.sigmoid(model(data).cpu()).numpy()
        for idx, output in enumerate(outputs):
            sample = samples[idx]
            sample_score[sample] = sample_score.setdefault(
                sample, np.zeros(10)) + output
            sample_num[sample] = sample_num.setdefault(sample, 0) + 1
            gt[sample] = targets[idx]
    preds, labels = [], []
    for k, v in sample_score.items():
        pred = ((v / sample_num[k]) > threshold).astype(np.int8)
        pred[np.argmax(v)] = 1
        preds.append(pred)
        labels.append(gt[k].numpy().astype(np.int8))
    f1_macro = f1_score(np.stack(labels), np.stack(preds), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(preds), average='micro')
    return f1_macro, f1_micro

@torch.no_grad()
def test(model, dataloader, DEVICE, threshold=0.5):
    model.eval()
    sample_score, gt, sample_num = {}, {}, {}
    for data, targets, samples, _ in tqdm(dataloader, desc='Test', ncols=75):
        data = data.to(DEVICE, torch.float)
        outputs = torch.sigmoid(model(data).cpu()).numpy()
        for idx, output in enumerate(outputs):
            sample = samples[idx]
            sample_score[sample] = sample_score.setdefault(
                sample, np.zeros(10)) + output
            sample_num[sample] = sample_num.setdefault(sample, 0) + 1
            gt[sample] = targets[idx]
    preds, labels, names = [], [], []
    
    for k, v in sample_score.items():
        pred = ((v / sample_num[k]) > threshold).astype(np.int8)
        pred[np.argmax(v)] = 1
        preds.append(pred)
        labels.append(gt[k].numpy().astype(np.int8))
        names.append(k)
    f1_macro = f1_score(np.stack(labels), np.stack(preds), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(preds), average='micro')
    preds = [np.where(p == 1)[0] for p in preds]
    labels = [np.where(l == 1)[0] for l in labels]
    df = pd.DataFrame(list(zip(preds, labels)), index=names, columns=['Pred', 'Label'])
    return df, f1_macro, f1_micro

@torch.no_grad()
def test_multiple(model, dataloader, DEVICE, threshold=0.5):
    model.eval()
    sample_score, gt, sample_num = {}, {}, {}
    for data, targets, samples, _ in tqdm(dataloader, desc='Test', ncols=75):
        data = data.to(DEVICE, torch.float)
        outputs = torch.sigmoid(model(data).cpu()).numpy()
        for idx, output in enumerate(outputs):
            sample = samples[idx]
            sample_score[sample] = sample_score.setdefault(
                sample, np.zeros(10)) + output
            sample_num[sample] = sample_num.setdefault(sample, 0) + 1
            gt[sample] = targets[idx]
    preds, labels, names = [], [], []
    
    for k, v in sample_score.items():
        pred = ((v / sample_num[k]) > threshold).astype(np.int8)
        pred[np.argmax(v)] = 1
        preds.append(pred)
        labels.append(gt[k].numpy().astype(np.int8))
        names.append(k)
    f1_macro = f1_score(np.stack(labels), np.stack(preds), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(preds), average='micro')
    preds = [np.where(p == 1)[0] for p in preds]
    labels = [np.where(l == 1)[0] for l in labels]
    df = pd.DataFrame(list(zip(preds, labels)), index=names, columns=['Pred', 'Label'])
    return df, f1_macro, f1_micro