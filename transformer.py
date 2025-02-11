import torch
import numpy as np
import pandas as pd
import re
import gensim.downloader as api
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, TensorDataset
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from collections import Counter
from torch import optim
import torch.nn.functional as F

'''
********************************************all function for the calculations are here*******************************************
'''
def attn(Q,K,V):
    D,l = Q.shape
    d_model = torch.tensor(D)
    qk_transpose = Q @ K.T
    scaling = qk_transpose/torch.sqrt(d_model)
    softmax_apply = torch.softmax(scaling, dim = -1)
    attention = softmax_apply @ V
    return attention 
    
def attention(X):
    N,D = X.shape
    d_model = torch.tensor(D)
    W_q = torch.rand(D,D)
    W_k = torch.rand(D,D)
    W_v = torch.rand(D,D)
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    qk_transpose = Q @ K.T
    scaling = qk_transpose/torch.sqrt(d_model)
    #print(scaling.shape)
    softmax_apply = torch.softmax(scaling, dim = -1)
    attention = softmax_apply @ V
    return attention 
    
def multi_head_attention(X,H,D_v):
    N,D = X.shape
    d_model = torch.tensor(D)
    W_o = torch.rand(H * D_v, D)
    heads = []
    for h in range(H):
        W_q = torch.rand(D,D)
        W_k = torch.rand(D,D)
        W_v = torch.rand(D,D_v)
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
        H_h= attn(Q,K,V)
        heads.append(H_h)
    H = torch.cat(heads, dim = 1)
    Y = H @ W_o
    mask = torch.triu(torch.ones_like(Y), diagonal=1).bool() 
    Y[mask] = large_negative_number
    return Y

def add_norm(Y,X):
    Z = Y + X
    layer_norm = nn.LayerNorm(normalized_shape=Z.size()[1:])
    z = layer_norm(Z)
    return z
    
def decoder_model(X, H, D_v):
    Y =  multi_head_attention(X,H,D_v)
    Z = add_norm(X,Y)
    m,n = X.shape
    MLP = nn.Linear(n,n)
    z_mlp = MLP(Z)
    X_tilda = add_norm(z_mlp, Z)
    return X_tilda


'''
***************************************multi head attention & decdoer block of the transformer**********************************
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, H, D_v, large_negative_number=-1e9):
        super(MultiHeadAttention, self).__init__()
        self.H = H  
        self.D_v = D_v  
        self.d_model = d_model
        self.large_negative_number = large_negative_number
        self.W_q = nn.Parameter(torch.randn(H, d_model, d_model, dtype=torch.float32))  
        self.W_k = nn.Parameter(torch.randn(H, d_model, d_model, dtype=torch.float32))
        self.W_v = nn.Parameter(torch.randn(H, d_model, D_v, dtype=torch.float32))
        self.W_o = nn.Parameter(torch.randn(H * D_v, d_model, dtype=torch.float32))
    def forward(self, X):
        B, N, D = X.shape
        assert D == self.d_model, "Input feature dimension must match d_model."
        heads = []
        for h in range(self.H):
            Q = X[:,] @ self.W_q[h]
            K = X[:,] @ self.W_k[h]
            V = X[:,]@ self.W_v[h]
            # print(Q.shape)
            # print(K.shape)
            K_T = K.transpose(-1, -2)
            scaling = Q[:,] @ K_T[:,] / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
            attention_weights = torch.softmax(scaling, dim=-1)
            H_h = attention_weights @ V
            heads.append(H_h)
        H_concat = torch.cat(heads, dim=-1)
        Y = H_concat @ self.W_o
        mask = torch.triu(torch.ones_like(Y), diagonal=1).bool()
        Y[mask] = self.large_negative_number
        return Y


class DecoderLayer(nn.Module):
    def __init__(self, d_model, H, D_v, large_negative_number=-1e9):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, H, D_v, large_negative_number)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Linear(d_model, d_model)

    def forward(self, X):
        Y = self.multi_head_attention(X)
        Z = self.layer_norm1(X + Y)
        z_mlp = self.feed_forward(Z)
        X_tilda = self.layer_norm2(Z + z_mlp)
        return X_tilda

'''
****************************************formating of the sentences and the tokenization of the data set******************************
'''
data_file_path = input("please give the path of the data file: ")

df = pd.read_csv(data_file_path)
text = df["Text"]
summary = df["Summary"]
text = df["Text"].apply(lambda x: '<sos>' + x + '<eos>')

def tokenization(text):
    return re.findall(r'\b\w+\b', text.lower())

combined_texts = text.tolist() 
combined_tokens  = []

for text in combined_texts:
    combined_tokens.append(tokenization(text))
tokens = []

for token_list in combined_tokens:
    for token in token_list:
        tokens.append(token)
        
token_counts  = Counter(tokens)
vocabulary = list(token_counts.keys())
vocabulary.append('<unk>')

def sentence_truncation(sentence_box, max_size):
    final_data = []
    for i in range(len(sentence_box)):
        if len(sentence_box[i]) <= max_size:
            final_data.append(sentence_box[i])
    return final_data 
    
final_data = sentence_truncation(combined_texts, max_size = 64)

embedding_dim = 50  
word2vec_model = Word2Vec(sentences=final_data, vector_size=embedding_dim, window=5, min_count=1, workers=4)
word_vectors = word2vec_model.wv
vocab = word_vectors.key_to_index.keys()


'''
****************************************** Sentence to embedding generation **********************************************
'''


def sentence_to_embedding(sentence, max_length, embedding_dim):
    """
    Convert a sentence (list of tokens) into a fixed-size embedding matrix with padding.
    """
    embedding = []
    for token in sentence:
        if token in word_vectors:
            embedding.append(word_vectors[token])
        else:
            embedding.append(np.zeros(embedding_dim))  
    
    if len(embedding) > max_length:
        embedding = embedding[:max_length]  
    else:
        padding = [np.zeros(embedding_dim)] * (max_length - len(embedding))  
        embedding.extend(padding)
    
    return np.array(embedding)


'''
*******************************************************Model initialization ***********************************************
'''
    
max_length = 64
embedded_data = [sentence_to_embedding(sentence, max_length, embedding_dim) for sentence in final_data]
X = np.array(embedded_data) 
w = np.random.randint(0, 2, size=len(embedded_data))
X_train, X_test, a, b = train_test_split(X, w, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)

if not isinstance(X, torch.Tensor):
    X = torch.tensor(X)
batch_size = 189
dataset = TensorDataset(X)  
total_size = len(dataset)
train_size = int(0.80 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
Decoder_model = DecoderLayer(d_model = 50, H=3 , D_v = 20, large_negative_number=-1e9 )


'''
********************************************************loss function**************************************************
'''


D = 50
vocab_size = len(vocabulary)  
num_epochs = int(input("please give the number of iterations you want to run the code for: "))
learning_rate = 1e-4
large_negative_number = -1e9
vocab_projection = nn.Linear(D, vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(Decoder_model.parameters()) + list(vocab_projection.parameters()), lr=learning_rate)


loss_fn = nn.CrossEntropyLoss()

'''
**************************************************** Training the transformer *********************************************
'''

for epoch in range(num_epochs):
    Decoder_model.train()
    total_loss = 0.0
    
    for batch_X, in train_loader:
        batch_X = batch_X.to("cpu", dtype=torch.float32) 
        #print(decoder_output.shape)
        decoder_output = Decoder_model(batch_X)
        vocab_logits = vocab_projection(decoder_output)
        vocab_logits_softmax = torch.softmax(vocab_logits, dim = -1)
        max_values, locations = torch.max(vocab_logits_softmax, dim=-1)
        
        targets = torch.argmax(batch_X, dim=-1) 
        # logits = vocab_logits[:, :-1] 
        # logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)/torch.max(targets)
        #targets = batch_X.to(torch.long)
        #logits = vocab_logits.view(-1, vocab_logits.size(-1))
        max_values = max_values.view(-1)
        loss = loss_fn(max_values, targets)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Training completed.")

