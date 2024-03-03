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
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss


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
        self.src_vocab_size = config["src_vocab_size"]
        self.target_vocab_size = config["target_vocab_size"]
        self.heads = config["heads"]
        self.layers_encoder = config["layers_encoder"]
        self.layers_decoder = config["layers_decoder"]
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
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.layers_encoder)
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout, 
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, self.layers_decoder)
        
        self.mfe_out = nn.Linear(self.d_model, 1)
        self.decod_out = nn.Linear(self.d_model, self.target_vocab_size)
        
        self.init_weights()
        
        self.to(self.device)
                                     
    def forward(self, src, tgt, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoder(tgt)
        decoder_output = self.transformer_decoder(tgt, encoder_output)

        mfe = torch.mean(encoder_output, dim=1)
        mfe = self.mfe_out(mfe)

        decoded = self.decoder_out(decoder_output)

        return mfe, decoded
    
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
    early_stop = 5
    best_valid_loss = 1e9
    counter = 0
    train_loss_mfe = []
    train_loss_ce = []
    valid_loss_mfe = []
    valid_loss_ce = []
    last_epoch = 0
    best_epoch = 0
    interval=10
    start_epoch = 0
    n_epochs = config['num_epochs']
    
    predicted_values_train = np.zeros(len(train_dataloader.dataset))
    actual_values_train = np.zeros(len(train_dataloader.dataset))
    predicted_structures_train = np.zeros(len(train_dataloader.dataset))
    actual_structures_train = np.zeros(len(train_dataloader.dataset))
    
    predicted_values_valid = np.zeros(len(valid_dataloader.dataset))
    actual_values_valid = np.zeros(len(valid_dataloader.dataset))
    predicted_structures_valid = np.zeros(len(valid_dataloader.dataset))
    actual_structures_valid= np.zeros(len(valid_dataloader.dataset))
    
    ################
    #Model, Optim and Loss
    ################
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], amsgrad=True)
    loss_function_mfe = nn.MSELoss()
    loss_function_structure = nn.CrossEntropyLoss()
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0)
    
    
    ################
    #Tensorboard output folder and init
    ################
    foldername = "run_2p5M_" + datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S") + "_lr_" + str(config['learning_rate']) + "_batchsize_" + str(config['batch_size'])
    log_dir = os.path.join(config['exp_name'], foldername)
    os.makedirs(log_dir, exist_ok=True)
    writer = tb.SummaryWriter(log_dir=log_dir)

    ################
    #Check for existing checkpoint
    ################
    """
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
    """
        
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
            sq, mfe, structure, mask = batch
            sq, mfe, structure, mask = sq.to(device), mfe.to(device), structure.to(device), mask.to(device)
                        
            optimizer.zero_grad()
            mfes, structures = model(sq, mask)
            loss_mfe = loss_function_mfe(mfes, mfe)
            loss_ce = loss_function_structure(structures, structure)
            
            #Minimizing total loss
            total_loss = loss_mfe + loss_ce
            
            total_loss.backward()
            optimizer.step()
            #scheduler.step()
            pred = mfes.detach().cpu().numpy()
            act = mfe.detach().cpu().numpy()
            pred_structures = structures.detach().cpu().numpy()
            act_structures = structure.detach().cpu().numpy()
            
            predicted_values_train[it*len(pred):(it+1)*len(pred)] = pred.flatten()
            actual_values_train[it*len(act):(it+1)*len(act)] = act.flatten()
            
            predicted_structures_train[it*len(pred_structures):(it+1)*len(pred_structures)] = pred_structures
            actual_structures_train[it*len(act_structures):(it+1)*len(act_structures)] = act_structures
            it+=1

        mse_train = mean_squared_error(actual_values_train, predicted_values_train)
        ce_train = log_loss(actual_structures_train, predicted_structures_train)
        train_loss_mse.append(mse_train)
        train_loss_ce.append(ce_train)
        
        #########################
        #VALIDATION PER EPOCH
        #########################
        model.eval()
        #running_loss = 0
        batch_iterator_valid = tqdm(valid_dataloader, desc=f"Validating epoch {epoch+1:02d}")
        
        it=0
        with torch.no_grad():
            for batch in batch_iterator_valid:
                sq, mfe, structure, mask = batch
                sq, structure, mask = sq.to(device), structure.to(device), mask.to(device)
                
                mfes, structures = model(sq, mask)
                #loss = loss_function_mfe(outputs, mfe)
                #running_loss += loss.item()
                pred = mfes.detach().cpu().numpy()
                act = mfe.detach().cpu().numpy()
                pred_structures = structures.detach().cpu().numpy()
                act_structures = structure.detach().cpu().numpy()
                
                predicted_values_valid[it*len(pred):(it+1)*len(pred)] = pred.flatten()
                actual_values_valid[it*len(act):(it+1)*len(act)] = act.flatten()
                
                predicted_structures_valid[it*len(pred_structures):(it+1)*len(pred_structures)] = pred_structures
                actual_structures_valid[it*len(act_structures):(it+1)*len(act_structures)] = act_structures 
                it+=1
        
        mse_valid = mean_squared_error(actual_values_valid, predicted_values_valid)
        ce_valid = log_loss(actual_structures_valid, predicted_structures_valid)
        valid_loss_mse.append(mse_valid)
        valid_loss_ce.append(ce_valid)
        
        
        
        writer.add_scalar('MSE-Loss/Train', mse_train, epoch+1)
        writer.add_scalar('MSE-Loss/Valid', mse_valid, epoch+1)
        writer.add_scalar('CE-Loss/Train', ce_train, epoch+1)
        writer.add_scalar('CE-Loss/Valid', ce_valid, epoch+1)
        
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
            best_epoch = epoch
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
        print(f'Epoch {epoch+1}/{n_epochs}, MSE Train Loss: {mse_train}, CE Train Loss: {ce_train}, MSE valid loss: {mse_valid}, CE valid loss: {ce_valid}')
        
 
    ################
    #TRACKING.
    ################ 
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_mse, label='Training Loss', color='blue')
    plt.plot(valid_loss_mse, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('MSE Training and Validation Losses Over Epochs', plt.gcf())
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_ce, label='Training Loss', color='blue')
    plt.plot(valid_loss_ce, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('CE Loss')
    plt.title('CE Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('CE Training and Validation Losses Over Epochs', plt.gcf())
    
    #scheduler_name = scheduler.__class__.__name__
    #scheduler_params = scheduler.state_dict()
    #scheduler_info = f"{scheduler_name} with params: {scheduler_params}"
    
    writer.add_text('Configuration', f"Last epoch: {last_epoch+1}, Best epoch: {best_epoch+1}, Learning Rate: {config['learning_rate']}, Scheduler: None, Batch Size: {config['batch_size']}, \
                    dropout: {config['dropout']}, d_model: {config['d_model']}, d_ff: {config['d_ff']}, Encoder Layers: {config['layers_encoder']}, heads: {config['heads']}, \
                    max_len: {config['max_len']}")
    
            
    ################
    #TEST LOOP
    ################     
    model.eval()
    predicted_values_test = np.zeros(len(test_dataloader.dataset))
    actual_values_test = np.zeros(len(test_dataloader.dataset))
    predicted_structures_test = np.zeros(len(test_dataloader.dataset))
    actual_structures_test = np.zeros(len(test_dataloader.dataset))

    it=0
    with torch.no_grad():
        for batch in test_dataloader:
            sq, mfe, structure, mask = batch
            sq, structure, mask = sq.to(device), structure.to(device), mask.to(device)
            
            mfes, structures = model(sq, mask)
            pred = mfes.detach().cpu().numpy()
            act = mfe.detach().cpu().numpy()
            pred_structures = structures.detach().cpu().numpy()
            act_structures = structure.detach().cpu().numpy()
                
            predicted_values_test[it*len(pred):(it+1)*len(pred)] = pred.flatten()
            actual_values_test[it*len(act):(it+1)*len(act)] = act.flatten()
                
            predicted_structures_test[it*len(pred_structures):(it+1)*len(pred_structures)] = pred_structures
            actual_structures_test[it*len(act_structures):(it+1)*len(act_structures)] = act_structures 
            it+=1
        
    mse_test = mean_squared_error(actual_values_test, predicted_values_test)
    ce_test = log_loss(actual_structures_test, predicted_structures_test)
    print(f'MSE: {mse_test}, CE: {ce_test}')
        
    plot_predvsactual(writer, actual_values_test, predicted_values_test, 0, mse_test, "Test")
    
    ###
    #Closing writer
    ###
    writer.close()
        
    ################
    #SAVING final model.
    ################     
    torch.save(model.state_dict(), f"packages/model/best_model/{current_time}best_model.pth")
    
    ################
    #Hyperparam tune based on minimizing sum of both losses.
    ################  
    return mse_valid + ce_valid
        
def plot_predvsactual(writer, actual, pred, epoch, mse, data_origin: str):
        #Plot predicted vs actual for train data every 20 epoch
        plt.figure(figsize=(8, 8))
        plt.scatter(actual, pred, alpha=0.5)
        plt.xlabel('Actual MFE')
        plt.ylabel('Predicted MFE')
        plt.title('Predicted vs Actual MFE')
        plt.grid(True)
        
        if data_origin != "Test":
            plt.text(x=0.05, y=0.95, s=f'Epoch: {epoch}', transform=plt.gca().transAxes, fontsize=12, ha='left', va='top')
        plt.text(x=0.05, y=0.85, s=f'MSE: {mse:.4f}', transform=plt.gca().transAxes, fontsize=12, ha='left', va='top')

        writer.add_figure(f'{data_origin} data: Predicted vs Actual MFE', plt.gcf())            