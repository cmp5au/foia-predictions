# -*- coding: utf-8 -*-
"""Text_Classification_With_BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KT8gH2tlAtVba50wxoTJ0neIAHLSfIl6
"""

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/drive')

df = pd.read_csv('/drive/MyDrive/body_multiclass_target.csv')
df['request_agency'] = pd.read_csv('/drive/MyDrive/model_features.csv')['request_agency']

df.dropna(axis=0, inplace=True)
new_labels = {'done' : 'Completed',
              'partial' : 'Completed',
              'no_docs' : 'Redacted',
              'fix' : 'Rejected',
              'rejected' : 'Rejected'}
df.loc[:, 'target'] = df.target.apply(lambda x: new_labels[x])

df['target'].value_counts()

possible_labels = df.target.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict

df['label'] = df.target.replace(label_dict)

df.head()

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val, agency_train, agency_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  df.request_agency.values,
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

df.groupby(['target', 'label', 'data_type']).count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    truncation=True, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].body.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    truncation=True, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

len(dataset_train), len(dataset_val)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False,
                                                      attention_probs_dropout_prob=0.05,
                                                      hidden_dropout_prob=0.05)

# for param in model.bert.bert.parameters():
#     param.requires_grad = False

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 1

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)

epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

def evaluate(dataloader_val, predicting=False):

    model.eval()
    
    loss_val_total = 0
    predictions, hidden_states, true_vals = [], [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        if predicting:
          hidden_state = outputs[2]
          hidden_states.append(hidden_state)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals



for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'/drive/MyDrive/triclass_d05_finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')



for param in model.bert.parameters():
    param.requires_grad = False

count=0
for param in model.parameters():
  count+=1
  if param.requires_grad: print(param)

print('number of parameters is', count)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=True)

model.to(device)

model.load_state_dict(torch.load('/drive/MyDrive/triclass_d10_finetuned_BERT_epoch_3.model', map_location=torch.device('cpu')))

torch.cuda.empty_cache()

_, predictions, true_vals, hidden_states = evaluate(dataloader_validation, predicting=True)

accuracy_per_class(predictions, true_vals)

class_preds = np.argmax(predictions, axis=1)
for i in range(3):
  print(f"Class {i}:", (class_preds == i).sum())

final_layer_output = []
idxs = np.random.choice(range(X_val.shape[0]), size=1000, replace=False)
model.to(torch.device('cpu'))

for i, batch in enumerate(dataloader_validation):
  if i in idxs:
    # batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids':      batch[0],
              'attention_mask': batch[1],
              'labels':         batch[2],
              }
    
    with torch.no_grad():
      final_layer_output.append(model(**inputs)[2][-1])

for tensor in outputs[2]:
  print(tensor.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# lr_clf = LogisticRegression()
# rf_clf = RandomForestClassifier()

final_layer_output = np.concatenate(final_layer_output)
X_output = np.concatenate(((final_layer_output).reshape(-1, 256 * 768),
                           np.array(agency_val)[idxs].reshape(-1, 1)), axis=1)

print(X_output.shape)

lr_clf.fit(X_output, y_train[idxs])
rf_clf.fit(X_output, y_train[idxs])

lr_prob_preds = lr_clf.predict_proba(X_output)
rf_prob_preds = rf_clf.predict_proba(X_output)

lr_preds = lr_clf.predict(X_output)
rf_preds = rf_clf.predict(X_output)

lr_val_prob_preds = lr_clf.predict_proba(X_output)
rf_val_prob_preds = rf_clf.predict_proba(X_output)

lr_val_preds = lr_clf.predict(X_output)
rf_val_preds = rf_clf.predict(X_output)

from sklearn.metrics import accuracy_score, f1_score

print(accuracy_score(y_train[idxs], lr_preds), f1_score(y_train[idxs], lr_preds, average='weighted'))
print(accuracy_score(y_train[idxs], rf_preds), f1_score(y_train[idxs], rf_preds, average='weighted'))

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

colors = np.array(['g' if y == 2 else 'k' if y == 1 else 'r'
                    for y in y_train])

tsne = TSNE(n_components=2)
tsne_3d = TSNE(n_components=3)

scatter = tsne.fit_transform(lr_prob_preds)

fig, ax = plt.subplots()

xs = scatter[:, 0]
ys = scatter[:, 1]

ax.scatter(xs, ys, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

scatter_3d = tsne_3d.fit_transform(lr_prob_preds)

xs_3d = scatter_3d[:, 0]
ys_3d = scatter_3d[:, 1]
zs_3d = scatter_3d[:, 2]

ax = plt.figure().gca(projection='3d')

ax.scatter(xs_3d, ys_3d, zs_3d, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

tsne = TSNE(n_components=2)
tsne_3d = TSNE(n_components=3)

scatter = tsne.fit_transform(rf_prob_preds)

fig, ax = plt.subplots()

xs = scatter[:, 0]
ys = scatter[:, 1]

ax.scatter(xs, ys, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

scatter_3d = tsne_3d.fit_transform(rf_prob_preds)

xs_3d = scatter_3d[:, 0]
ys_3d = scatter_3d[:, 1]
zs_3d = scatter_3d[:, 2]

ax = plt.figure().gca(projection='3d')

ax.scatter(xs_3d, ys_3d, zs_3d, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

colors = np.array(['g' if y == 2 else 'k' if y == 1 else 'r'
                    for y in y_val])

tsne = TSNE(n_components=2)
tsne_3d = TSNE(n_components=3)

scatter = tsne.fit_transform(lr_prob_preds)

fig, ax = plt.subplots()

xs = scatter[:, 0]
ys = scatter[:, 1]

ax.scatter(xs, ys, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

scatter_3d = tsne_3d.fit_transform(lr_prob_preds)

xs_3d = scatter_3d[:, 0]
ys_3d = scatter_3d[:, 1]
zs_3d = scatter_3d[:, 2]

ax = plt.figure().gca(projection='3d')

ax.scatter(xs_3d, ys_3d, zs_3d, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

colors = np.array(['g' if y == 2 else 'k' if y == 1 else 'r'
                    for y in y_val])

tsne = TSNE(n_components=2)
tsne_3d = TSNE(n_components=3)

scatter = tsne.fit_transform(rf_prob_preds)

fig, ax = plt.subplots()

xs = scatter[:, 0]
ys = scatter[:, 1]

ax.scatter(xs, ys, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

scatter_3d = tsne_3d.fit_transform(rf_prob_preds)

xs_3d = scatter_3d[:, 0]
ys_3d = scatter_3d[:, 1]
zs_3d = scatter_3d[:, 2]

ax = plt.figure().gca(projection='3d')

ax.scatter(xs_3d, ys_3d, zs_3d, c=colors[idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

print(accuracy_score(true_vals[idxs], lr_preds), f1_score(true_vals[idxs], lr_preds, average='weighted'))
print(accuracy_score(true_vals[idxs], rf_preds), f1_score(true_vals[idxs], rf_preds, average='weighted'))

model.to(device)
_, predictions, true_vals = evaluate(dataloader_validation)

class_preds = np.argmax(predictions, axis=1)

print(accuracy_score(true_vals, class_preds), f1_score(true_vals, class_preds, average='weighted'))

true_vals[:5]

last_idxs = np.random.choice(range(len(true_vals)), size=1000, replace=False)

colors = np.array(['g' if y == 2 else 'k' if y == 1 else 'r'
                    for y in true_vals])

tsne = TSNE(n_components=2)
tsne_3d = TSNE(n_components=3)

scatter = tsne.fit_transform(class_preds.reshape(-1, 1)[last_idxs])

fig, ax = plt.subplots()

xs = scatter[:, 0]
ys = scatter[:, 1]

ax.scatter(xs, ys, c=colors[last_idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")

scatter_3d = tsne_3d.fit_transform(class_preds.reshape(-1, 1)[last_idxs])

xs_3d = scatter_3d[:, 0]
ys_3d = scatter_3d[:, 1]
zs_3d = scatter_3d[:, 2]

ax = plt.figure().gca(projection='3d')

ax.scatter(xs_3d, ys_3d, zs_3d, c=colors[last_idxs])
ax.set_title("BERT Sequence Classifier visualized with t-SNE")
