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
from sklearn.metrics import mean_squared_error, mean_absolute_error


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        #positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        #position of words [0,1,2,...,max_len-1] as a 1D tensor (1, max_len)
        pos = torch.arange(0,  max_len).unsqueeze(1)

        #division term, where i is [0,2,...,2*(d_model-1)]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))

        #calculating positional encodings
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)

        #we calculated positional encodings for one batch.
        #(1, max_len, d_model)
        pe = pe.unsqueeze(0) 

        #Not learned parameter, want it saved
        self.register_buffer('pe', pe)

    def forward(self, x):
        #we don't want the model to learn this
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    
    def __init__(self,config):
        super(Transformer,self).__init__()
        self.d_model = config["d_model"]
        self.vocab_size = config["vocab_size"]
        self.heads = config["heads"]
        self.N = config["N"]
        self.d_ff = config["d_ff"]
        self.dropout = config["dropout"]
        self.max_len = config["max_len"]
        self.device = config["device"]
        
        self.model_type = "Transformer"
        
        self.positional_encoder = PositionalEncoding(
            d_model = self.d_model,
            max_len = self.max_len,
            dropout = self.dropout
        )
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.heads,
            dim_feedforward  = self.d_ff,
            dropout  = self.dropout, 
            batch_first = True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.N)
        
        self.out = nn.Linear(self.d_model, 1)
        
        self.init_weights()
        
        self.to(self.device)
                                     
    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = torch.mean(x,dim=1)
        x = self.out(x)
        return x
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        
def train_model(config, train_dataloader, valid_dataloader, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    
    ################
    #Setting up variables
    ################
    early_stop = 10
    best_valid_loss = 1e9
    counter = 0
    train_loss = np.zeros(len(train_dataloader.dataset))
    valid_loss = np.zeros(len(valid_dataloader.dataset))
    last_epoch = 0
    interval=10
    start_epoch = 0
    n_epochs = config['num_epochs']
    
    predicted_values_train = np.zeros(len(train_dataloader.dataset))
    actual_values_train = np.zeros(len(train_dataloader.dataset))
    predicted_values_valid = np.zeros(len(valid_dataloader.dataset))
    actual_values_valid = np.zeros(len(valid_dataloader.dataset))
    
    ################
    #Model, Optim and Loss
    ################
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], amsgrad=True)
    loss_function = nn.MSELoss()
    #Eventually, we could try scheduling. right now, it seems a constant 1e-5 lr is good enough.
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=0)
    
    ################
    #Tensorboard output folder and init
    ################
    current_time = "TV-2p5M-" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    log_dir = os.path.join(config['exp_name'], current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = tb.SummaryWriter(log_dir=log_dir)

    ################
    #Check for existing checkpoint
    ################
    checkpoint_path = f"packages/model/model_checkpoint/{config['prefix']}model_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  
        except FileNotFoundError:
            print("No checkpoint found.")   
    else:
        print("No checkpoint found. Continue from scratch.")   
        
    ################
    #Loop for EPOCHS.
    ################       
    for epoch in range(start_epoch, n_epochs):
        ################
        #TRAINING
        ################ 
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training epoch {epoch+1:02d}")
        
        #iterator used in an attempt to save memory. using nparray instead of lists.
        it=0
        for batch in batch_iterator:
            sq, mfe, mask = batch
            sq, mfe, mask = sq.to(device), mfe.to(device), mask.to(device)
                        
            optimizer.zero_grad()
            outputs = model(sq, mask)
            loss = loss_function(outputs, mfe)

            loss.backward()
            optimizer.step()
            #scheduler.step()
            pred = outputs.detach().cpu().numpy()
            act = mfe.detach().cpu().numpy()
            predicted_values_train[it*len(pred):(it+1)*len(pred)] = pred.flatten()
            actual_values_train[it*len(act):(it+1)*len(act)] = act.flatten()
            it+=1

        mse_train = mean_squared_error(actual_values_train, predicted_values_train)
        train_loss[epoch] = mse_train
        
        #########################
        #VALIDATION PER EPOCH
        #########################
        model.eval()
        #running_loss = 0
        batch_iterator_valid = tqdm(valid_dataloader, desc=f"Validating epoch {epoch+1:02d}")
        
        it=0
        with torch.no_grad():
            for batch in batch_iterator_valid:
                sq, mfe, mask = batch
                sq, mask = sq.to(device), mask.to(device)
                
                outputs = model(sq, mask)
                #loss = loss_function(outputs, mfe)
                #running_loss += loss.item()
                pred = outputs.detach().cpu().numpy()
                act = mfe.detach().cpu().numpy()
                predicted_values_valid[it*len(pred):(it+1)*len(pred)] = pred.flatten()
                actual_values_valid[it*len(act):(it+1)*len(act)] = act.flatten()
                
                it+=1
        
        mse_valid = mean_squared_error(actual_values_valid, predicted_values_valid)
        valid_loss[epoch] = mse_valid
        
        ################
        #TRACKING per interval.
        ################ 
        last_epoch = epoch
        
        if epoch % interval == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            plot_predvsactual(writer, actual_values_train, predicted_values_train, epoch, mse_train, "Train")        
            plot_predvsactual(writer, actual_values_valid, predicted_values_valid, epoch, mse_valid, "Valid")
                 
        ################
        #EARLY STOP AND CHECKPOINT
        ################ 
        if mse_valid < best_valid_loss:
            best_valid_loss = mse_valid
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"packages/model/model_checkpoint/{config['prefix']}model_checkpoint.pth")
        else:
            counter += 1
            
        if counter >= early_stop:
            print(f"Stopped early at epoch {epoch+1}")
            plot_predvsactual(writer, actual_values_train, predicted_values_train, epoch, mse_train, "Train")
            plot_predvsactual(writer, actual_values_valid, predicted_values_valid, epoch, mse_valid, "Valid")
            break
            
        
        ################
        #Output LOSS for visualization in terminal
        ################ 
        print(f'Epoch {epoch+1}/{n_epochs}, MSE Train Loss: {mse_train}, MSE valid loss: {mse_valid}')
        
        
    ################
    #TEST LOOP
    ################     
    model.eval()
    predicted_values_test = np.array(len(test_dataloader.dataset))
    actual_values_test = np.array(len(test_dataloader.dataset))

    it=0
    with torch.no_grad():
        for batch in test_dataloader:
            sq, mfe, mask = batch
            sq, mfe, mask = sq.to(device), mask.to(device)
            
            outputs = model(sq, mask)
            pred = outputs.detach().cpu().numpy()
            act = mfe.detach().cpu().numpy()
            predicted_values_test[it*len(pred):(it+1)*len(pred)] = pred.flatten()
            actual_values_test[it*len(act):(it+1)*len(act)] = act.flatten()
            
            it+=1
            
    mse = mean_squared_error(actual_values_test, predicted_values_test)
    mae = mean_absolute_error(actual_values_test, predicted_values_test)
    print(f'MSE: {mse}, MAE: {mae}')
    
    
    ################
    #TRACKING.
    ################ 
    plot_predvsactual(writer, actual_values_test, predicted_values_test, 0, mse, "Test")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(valid_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('Training and Validation Losses Over Epochs', plt.gcf())
    
    writer.add_text('Configuration', f"Last epoch: {last_epoch+1}, Learning Rate: {config['learning_rate']}, Batch Size: {config['batch_size']}, \
                    dropout: {config['dropout']}, d_model: {config['d_model']}, d_ff: {config['d_ff']}, Encoder Layers: {config['N']}, heads: {config['heads']}, \
                    max_len: {config['max_len']}")
    writer.close()
        
    ################
    #SAVING final model.
    ################     
    torch.save(model.state_dict(), f"packages/model/best_model/{current_time}best_model.pth")
    
def plot_predvsactual(writer, actual, pred, epoch, mse, data_origin: str):
        #Plot predicted vs actual for train data every 20 epoch
        plt.figure(figsize=(8, 8))
        plt.scatter(actual, pred, alpha=0.5)
        plt.xlabel('Actual MFE')
        plt.ylabel('Predicted MFE')
        plt.title('Predicted vs Actual MFE')
        plt.grid(True)

        writer.add_figure(f'{data_origin} data: Predicted vs Actual MFE after {epoch} epochs', plt.gcf())
        writer.add_scalar("Loss/train", mse, epoch)
            