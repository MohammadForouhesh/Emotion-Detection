import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd

import gc
import logging
import argparse
import warnings
from src.metrics import ir_metrics
from src.model import LSTM, CNN
from src.preprocessing.Preprocessing import correction, preprocess
from src.utils.Run import run

gc.enable()
warnings.filterwarnings("ignore", category=DeprecationWarning)

N_EPOCH = 300

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

file = open('persian', 'r')
sw_persian = list(file.read().splitlines())

emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def preparation(args) -> (pd.DataFrame, pd.DataFrame):
    train_df = pd.read_csv(args.train_path)
    
    test_df = pd.read_csv(args.test_path)
    df = train_df.append(test_df)
    df['category_id'] = df['label'].factorize()[0]
    
    category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    
    df.label = list(correction(df.label))
    df['preprocessed'] = df.text.apply(preprocess)
    
    train_df = df[:5999]
    test_df = df[5999:]
    return df[:5999], df[5999:], category_to_id


def main():
    parser = argparse.ArgumentParser(description='Exa Emotion Detection')
    parser.add_argument('--train_path', dest='train_path', type=str, default='dataset/Emotion.csv',
                        help='Raw dataset file address.')
    parser.add_argument('--augment', dest='augment', type=int, default=None,
                        help='augment the dataset to learn better.')
    parser.add_argument('--model_name', dest='model_name', type=str, default='lstm',
                        help="supported models in this implementation are CNN and LSTM.")
    parser.add_argument('--preprocess', dest='preprocess', type=bool, default=True,
                        help="whether or not preprocessing the training set.")
    parser.add_argument('--epoch', dest='epoch', type=int, default=100,
                        help="number of epochs in the training")
    parser.add_argument('--test_path', dest='test_path', type=str, default='dataset/EmotionTest.csv',
                        help="address to test dataset.")
    
    args = parser.parse_args()
    
    train_df, test_df, category_to_id = preparation(args)
    
    sentences = list(train_df.preprocessed)
    sentence_embeddings = emb_model.encode(sentences)
    
    inputs = torch.from_numpy(sentence_embeddings).to(device)
    target = torch.cuda.LongTensor(train_df.category_id)
    
    train_ds = TensorDataset(inputs, target)
    
    if args.model_name is 'lstm':
        bach_size = 5
        train_dl = DataLoader(train_ds, bach_size, shuffle=True)
        
        lstm_model = LSTM(output_size=len(category_to_id)).to(device)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(device)
        optimizer = torch.optim.AdamW(lstm_model.parameters(), amsgrad=True)
        
        trained_lstm = run(model=lstm_model, iterator=train_dl, optimizer=optimizer,
                           loss_function=loss_function, n_epoch=args.epoch, if_lstm=True)
        
        ir_metrics(model=lstm_model)
    
    elif args.model_name is 'cnn':
        bach_size = 1
        train_dl = DataLoader(train_ds, bach_size, shuffle=True)
        
        cnn_model = CNN(output_dim=len(category_to_id)).to(device)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(device)
        optimizer = torch.optim.AdamW(cnn_model.parameters(), amsgrad=True)
        
        trained_cnn = run(model=cnn_model, iterator=train_dl, optimizer=optimizer,
                          loss_function=loss_function)

        ir_metrics(model=cnn_model)