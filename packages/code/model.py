import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
from pathlib import Path
from dataset import data_split

import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader, random_split
import numpy as np

from tqdm import tqdm
import datetime
import os

import matplotlib.pyplot as plt


"""
Le Modele Transformer est implémenté selon l'article All you need is attention
LINK: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

Je m'aide aussi de l'article et de la vidéo suivante:
ARTICLE: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
VIDEO: https://www.youtube.com/watch?v=ISNdQcPhsts&list=WL&index=7&t=1064s
"""

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, sq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.sq_len = sq_len
        self.dropout = nn.Dropout(dropout)

        #positional encoding matrix
        pe = torch.zeros(sq_len, d_model)

        #position of words [0,1,2,...,sq_len-1] as a 1D tensor (1, sq_len)
        pos = torch.arange(0,  sq_len).unsqueeze(1)

        #division term, where i is [0,2,...,2*(d_model-1)]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))

        #calculating positional encodings
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)

        #we calculated positional encodings for one batch.
        #(1, sq_len, d_model)
        pe = pe.unsqueeze(0) 

        #Not learned parameter, want it saved
        self.register_buffer('pe', pe)

    def forward(self, x):
        #we don't want the model to learn this
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.beta * (x-mean)/(std+self.eps) + self.gamma
    
class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_W1b1 = nn.Linear(d_model, d_ff)
        self.linear_W2b2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        #We have our tensor x (batch, sq_len, d_model)*linear_W1b1 = (batch, sq_len, d_ff)
        #and then (batch, sq_len, d_ff)*linear_W2b2 = (batch, sq_len, d_model)
        return self.linear_W2b2(self.dropout(torch.relu(self.linear_W1b1(x))))
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        #We need d_model to be divisible by number of heads
        assert d_model % heads == 0, "d_model not divisible by heads"

        self.d_k = d_model //  heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]

        score = (query @ key.transpose(-1,-2))/math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(3)
            score.masked_fill(mask==0, float('-inf'))

        score = score.softmax(dim=-1)

        if dropout is not None:
            score = dropout(score)

        #score used for visualization
        return (score @ value), score


    def forward(self, q, k, v, mask):
        #We keep the same dimensions as inputs q, k, v here since Wq, Wk, Wv are (d_model,d_model)
        query = self.Wq(q) 
        key = self.Wk(k)
        value = self.Wv(v)

        #We divide each matrices into "heads:int" pieces to give to each heads
        #(batch, sq_len, d_model) -> (batch, sq_len, heads, d_k) -> (batch, heads, sq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1,2)

        x, self.score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        #change back heads and sq_len and concatenate
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.heads*self.d_k)

        return self.Wo(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        #Organize list of modules
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    #We send the x to the multihead attention and also to the normalization layer.
    #We then combine the two in the normalization layer
    def forward(self, x, mask):
        combined_ = self.residual_connections[0](x, lambda x: self.self_attention(x,x,x,mask))
        combined_ = self.residual_connections[1](combined_, self.feed_forward)
        return combined_
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerRegressor(nn.Module):
    def __init__(self, encoder: Encoder, intput_embed: InputEmbedding, positional_encod: PositionalEncoding):
        super().__init__()
        self.encoder = encoder
        self.input_embed = intput_embed
        self.positional_encod = positional_encod
        self.regression_head = nn.Linear(intput_embed.d_model,1)
        
    def forward(self, x, mask):
        x = self.input_embed(x)
        x = self.positional_encod(x)
        x = self.encoder(x, mask)
        x = torch.mean(x,dim=1)
        return self.regression_head(x)

    

def build_transformer(vocab_size: int, sq_len: int, d_model: int, N: int, heads: int, dropout: float, d_ff: int):
        #Embedding layers
        embed = InputEmbedding(d_model=d_model, vocab_size=vocab_size)
        
        #Positional encoding layer
        pos_encod = PositionalEncoding(d_model=d_model, sq_len=sq_len, dropout=dropout)

        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttention(d_model=d_model, heads=heads, dropout=dropout)
            feed_forward_block = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout=dropout)
            encoder_blocks.append(encoder_block)
        
        #We build the encoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))

        #We build the transformer
        transformerRegressor = TransformerRegressor(encoder=encoder, intput_embed=embed, positional_encod=pos_encod)    

        for p in transformerRegressor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformerRegressor


def train_model(config, train_dataloader, valid_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    
    model = build_transformer(config['vocab_size'], config['sq_len'], config['d_model'], config['N'], config['heads'], config['dropout'], config['d_ff'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-7)
    loss_function = nn.MSELoss()

    n_epochs = config['num_epochs']
    
    #Eventually, we could try scheduling. right now, it seems a constant 1e-5 lr is good enough.
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=0)

    model = model.to(device)

    early_stop = 20
    best_valid_loss = 1e9
    counter = 0
    valid_loss = []
    train_loss = []
    last_epoch = 0
    interval=25
    
    ######
    #Change 2p5M by data size used for training.
    ######
    current_time = "2p5M-" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    log_dir = os.path.join(config['exp_name'], current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = tb.SummaryWriter(log_dir=log_dir)
    
    for epoch in range(n_epochs):
        model.train()
        
        batch_iterator = tqdm(train_dataloader, desc=f"Training epoch {epoch+1:02d}")
        running_loss_MSE=0
        for batch in batch_iterator:
            sq, mfe, mask = batch
            sq, mfe, mask = sq.to(device), mfe.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(sq, mask)
            loss = loss_function(outputs, mfe)

            loss.backward()
            optimizer.step()
            #scheduler.step()
            running_loss_MSE += loss.item()
            avg_train_mse_loss = running_loss_MSE/len(train_dataloader)

        train_loss.append(avg_train_mse_loss)   
        
        #########################
        #Validation after 1 epoch
        #########################
        model.eval()
        predicted_values = []
        actual_values = []
        loss_function = nn.MSELoss()
        running_loss = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                sq, mfe, mask = batch
                sq, mfe, mask = sq.to(device), mfe.to(device), mask.to(device)
                outputs = model(sq, mask)
                loss = loss_function(outputs, mfe)
                running_loss += loss.item()
                predicted_values.extend(outputs.cpu().numpy())
                actual_values.extend(mfe.cpu().numpy())
        avg_valid_loss = running_loss / len(valid_dataloader)
        valid_loss.append(avg_valid_loss)
        
        last_epoch = epoch
        
        if epoch % interval == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
        
        #Early stopping and checkpoint best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            counter = 0
            torch.save(model.state_dict(), 'packages/model/best-model_checkpoint/best_model.pth')
        else:
            counter += 1
            
        if counter >= early_stop:
            print(f"Stopped early at epoch {epoch+1}")
            break
            
        #This is overall MSE loss. Since data is skewed, would be nice to get MSE by interval of sequences.
        print(f'Epoch {epoch+1}/{n_epochs}, MSE Train Loss: {avg_train_mse_loss}, MSE valid loss: {avg_valid_loss}')
        
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_values, predicted_values, alpha=0.5)
    plt.xlabel('Actual MFE')
    plt.ylabel('Predicted MFE')
    plt.title('Predicted vs Actual MFE')
    plt.grid(True)

    writer.add_figure('Predicted vs Actual MFE', plt.gcf())
    writer.add_scalar("Loss/valid", avg_valid_loss, len(valid_dataloader))
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(valid_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('Training and Validation Losses Over Epochs', plt.gcf())
    
    writer.add_text('Configuration', f"Last epoch: {last_epoch}, Learning Rate: {config['learning_rate']}, Batch Size: {config['batch_size']}, \
                    dropout: {config['dropout']}, d_model: {config['d_model']}, d_ff: {config['d_ff']}, Encoder Layers: {config['N']}, heads: {config['heads']}, \
                    max_len: {config['max_len']}")

    writer.close()

def test_model(config, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    
    model = build_transformer(config['vocab_size'], config['sq_len'], config['d_model'], config['N'], config['heads'], config['dropout'], config['d_ff'])
    model.load_state_dict(torch.load('packages/model/best-model_checkpoint/best_model.pth'))
    
    model.eval()
    predictions = []
    actual_values = []

    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            sq, mfe, mask = batch
            sq, mfe, mask = sq.to(device), mfe.to(device), mask.to(device)
            outputs = model(sq, mask)
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(mfe.cpu().numpy())

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(actual_values, predictions)
    mae = mean_absolute_error(actual_values, predictions)

    print(f'MSE: {mse}, MAE: {mae}')