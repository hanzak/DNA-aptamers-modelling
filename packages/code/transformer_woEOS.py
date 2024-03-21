import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle
from pathlib import Path
from dataset import data_split

import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm
import datetime
import os

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, confusion_matrix

seed = 22  
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    
class NegativeReLU(nn.Module):
    def forward(self, x):
        return -F.relu(-x)

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
                        
        self.embedding = nn.Embedding(self.src_vocab_size, self.d_model)
        
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
        
        self.mfe_out = nn.Sequential(nn.Linear(self.d_model, 1), NegativeReLU())
        self.decoder_out = nn.Linear(self.d_model, self.target_vocab_size)
        self.num_hairpins_out = nn.Sequential(nn.Linear(self.d_model, 1), nn.ReLU())
        
        self.init_weights()
        
        self.to(self.device)
                                     
    def forward(self, src, target, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):        
        batchsize = src.size(0)
        
        device = src.device
                
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)
        
        encoder_output = self.transformer_encoder(src,mask=src_mask, src_key_padding_mask=src_padding_mask)

        target = self.embedding(target) * math.sqrt(self.d_model)
        target = self.positional_encoder(target)
                
        decoder_output = self.transformer_decoder(target, encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        decoded = self.decoder_out(decoder_output)
        decoded_probs = F.softmax(decoded, dim=-1)  
        
        mean_decoder_out = torch.mean(decoder_output, dim=1)
        #mean_encoder_out = torch.mean(encoder_output, dim=1)
        mfe = self.mfe_out(mean_decoder_out)
        num_hairpins_pred = self.num_hairpins_out(mean_decoder_out)  

        return mfe, decoded, decoded_probs, num_hairpins_pred 
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        
def train_model(config, train_dataloader, valid_dataloader, model_checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    
    ################
    #Setting up variables
    ################
    early_stop = 3
    best_valid_loss = 1e9
    counter = 0
    train_loss_mse_mfe = []
    train_loss_mse_num_hairpins = []
    train_loss_ce = []
    valid_loss_mse_mfe = []
    valid_loss_mse_num_hairpins = []
    valid_loss_ce = []
    list_train_loss_total = []
    list_valid_loss_total = []
    last_epoch = 0
    best_epoch = 0
    interval=10
    start_epoch = 0
    n_epochs = config['num_epochs']
    normalize_epochs = 3
    print_prediction=True
    reverse_mapping = {1: '(', 2: ')', 3: '.'}
    
    
    predicted_mfes_train = np.zeros(len(train_dataloader.dataset))
    actual_mfes_train = np.zeros(len(train_dataloader.dataset))
    predicted_num_hairpins_train = np.zeros(len(train_dataloader.dataset))
    actual_num_hairpins_train = np.zeros(len(train_dataloader.dataset))
    
    
    predicted_mfes_valid = np.zeros(len(valid_dataloader.dataset))
    actual_mfes_valid = np.zeros(len(valid_dataloader.dataset))
    predicted_num_hairpins_valid = np.zeros(len(valid_dataloader.dataset))
    actual_num_hairpins_valid = np.zeros(len(valid_dataloader.dataset))
    
    
    ################
    #Model, Optim and Loss
    ################
    model = Transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    ce_weights = torch.tensor([0.0, 2.0, 2.0, 1.0, 0.0])
    ce_weights = ce_weights.to(device)
    loss_function_mse = nn.MSELoss()
    CELoss = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=0)
    
    ################
    #Tensorboard output folder and init
    ################
    start_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    foldername = config['data_size'] + "_woEOS_lr_" + str(config['learning_rate']) + "_batchsize_" + str(config['batch_size']) + "_dropout_" + str(config['dropout'])
    log_dir = os.path.join(config['exp_name']+config['data_size'], foldername)
    os.makedirs(log_dir, exist_ok=True)
    writer = tb.SummaryWriter(log_dir=log_dir)

    ################
    #Check for existing checkpoint
    ################
    """
    checkpoint_path = f"packages/model/model_checkpoint/2p5M/11-03-2024_011243_model_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  
            writer = tb.SummaryWriter(log_dir="packages/model/runs/tmodel/2p5M/run_11-03-2024_011243_lr_0.0002_batchsize_128_dropout_0.1")
        except FileNotFoundError:
            print("No checkpoint found.")   
    else:
        print("No checkpoint found. Continue from scratch.")   
    """
        
    ################
    #Loop for EPOCHS.
    ################ 
    
    first_epoch = True
    
    baseline_mse_mfe = 0
    baseline_mse_num_hairpins = 0
    baseline_ce = 0
    normalized=False
        
    for epoch in range(start_epoch, n_epochs):
        
        print_prediction=True
    
        ################
        #TRAINING
        ################ 
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training epoch {epoch+1:02d}")
        
        print(len(train_dataloader.dataset))
        
        #iterator used in an attempt to save memory. using nparray instead of lists.
        it=0
        running_ce=0
        for i,batch in enumerate(batch_iterator):
            sq, mfe, target, num_hairpins = batch
            
            #when adding custom loss, need to convert this to float for some reasons.
            mfe=mfe.float()
            num_hairpins=num_hairpins.float()

            sq, mfe, target, num_hairpins = sq.to(device), mfe.to(device), target.to(device), num_hairpins.to(device)
            
            target_input = target[:, :-1]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(sq, target_input)
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)
            
            mfes, predicted_structures, structures_prob, predicted_num_hairpins = model(sq,target_input,src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        
            optimizer.zero_grad()
            
            predicted_indices = torch.argmax(structures_prob, dim=-1)
            
            target_out = target[:, 1:]
            
            loss_mfe = loss_function_mse(mfes, mfe)
            loss_num_hairpins = loss_function_mse(predicted_num_hairpins, num_hairpins)
            ce_loss = CELoss(predicted_structures.contiguous().view(-1, 5), target_out.contiguous().view(-1))
                        
            
            if epoch <= normalize_epochs:                                        
                baseline_mse_mfe += loss_mfe.item()
                baseline_mse_num_hairpins += loss_num_hairpins.item()
                baseline_ce += ce_loss.item()

                total_loss = loss_mfe + ce_loss + loss_num_hairpins 
            else:
                if not normalized:
                    num_batches = len(train_dataloader) * normalize_epochs
                    baseline_mse_mfe /= num_batches
                    baseline_mse_num_hairpins /= num_batches
                    baseline_ce /= num_batches
                    normalized = True
                    
                normalized_mse_mfe = loss_mfe / baseline_mse_mfe
                normalized_mse_num_hairpins = loss_num_hairpins / baseline_mse_num_hairpins
                normalized_ce = ce_loss / baseline_ce
                            
                total_loss = normalized_mse_mfe + normalized_mse_num_hairpins + normalized_ce
            
            running_ce += ce_loss.item()
                                
            total_loss.backward()
            optimizer.step()
            
            pred_mfe = mfes.detach().cpu().numpy()
            pred_num_hairpins = predicted_num_hairpins.detach().cpu().numpy()
                        
            predicted_mfes_train[it*len(pred_mfe):(it+1)*len(pred_mfe)] = pred_mfe.flatten()
            predicted_num_hairpins_train[it*len(pred_num_hairpins):(it+1)*len(pred_num_hairpins)] = pred_num_hairpins.flatten()
            
            if first_epoch:
                act_mfe = mfe.detach().cpu().numpy()
                act_num_hairpins = num_hairpins.detach().cpu().numpy()
                actual_mfes_train[it*len(act_mfe):(it+1)*len(act_mfe)] = act_mfe.flatten()
                actual_num_hairpins_train[it*len(act_num_hairpins):(it+1)*len(act_num_hairpins)] = act_num_hairpins.flatten()
            
            it+=1

        mse_train_mfe = mean_squared_error(actual_mfes_train, predicted_mfes_train)
        mse_train_num_hairpins = mean_squared_error(actual_num_hairpins_train, predicted_num_hairpins_train)
        
        ce_train = running_ce/len(list(train_dataloader))
        total_train_loss = ce_train + mse_train_mfe + mse_train_num_hairpins
        train_loss_mse_mfe.append(mse_train_mfe)
        train_loss_mse_num_hairpins.append(mse_train_num_hairpins)
        train_loss_ce.append(ce_train)
        list_train_loss_total.append(total_train_loss)
        
        #########################
        #VALIDATION PER EPOCH
        #########################
        model.eval()
        batch_iterator_valid = tqdm(valid_dataloader, desc=f"Validating epoch {epoch+1:02d}")
        
        it=0
        running_ce=0
        with torch.no_grad():
            for batch in batch_iterator_valid:                
                sq, mfe, target, num_hairpins = batch                
                sq, mfe, target, num_hairpins = sq.to(device), mfe.to(device), target.to(device), num_hairpins.to(device)
                
                target_input = target[:, :-1]
            
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(sq, target_input)
            
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)
            
                mfes, predicted_structures, structures_prob, predicted_num_hairpins = model(sq,target_input,src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    
                predicted_indices = torch.argmax(structures_prob, dim=-1)
            
                target_out = target[:, 1:]
                
                if print_prediction:
                    predicted = predicted_indices[0:10].tolist()
                    target_structure = target_out[0:10].tolist()
                    decoded_predicted_structures = [''.join(reverse_mapping[symbol] for symbol in structure if symbol in reverse_mapping) for structure in predicted]
                    target_structure_10 = [''.join(reverse_mapping[symbol] for symbol in structure if symbol in reverse_mapping) for structure in target_structure]
                    
                    print("Predicted structures: ")
                    for j,decoded_structure in enumerate(decoded_predicted_structures):
                        print(decoded_structure)
                        print(mfes[j])
                        
                    print("Target structures: ")
                    for j,target_ in enumerate(target_structure_10):
                        print(target_)
                        print(mfe[j])
                        
                    print_prediction=False
                
                        
                ce_loss = CELoss(predicted_structures.contiguous().view(-1, 5), target_out.contiguous().view(-1))
                running_ce += ce_loss.item()

                pred_mfe = mfes.detach().cpu().numpy()
                act_mfe = mfe.detach().cpu().numpy()

                pred_num_hairpins = predicted_num_hairpins.detach().cpu().numpy()
                act_num_hairpins = num_hairpins.detach().cpu().numpy()

                predicted_mfes_valid[it*len(pred_mfe):(it+1)*len(pred_mfe)] = pred_mfe.flatten()
                predicted_num_hairpins_valid[it*len(pred_num_hairpins):(it+1)*len(pred_num_hairpins)] = pred_num_hairpins.flatten()

                if first_epoch:
                    actual_mfes_valid[it*len(act_mfe):(it+1)*len(act_mfe)] = act_mfe.flatten()
                    actual_num_hairpins_valid[it*len(act_num_hairpins):(it+1)*len(act_num_hairpins)] = act_num_hairpins.flatten()

                it+=1

        
        first_epoch = False
        
        mse_valid_mfe = mean_squared_error(actual_mfes_valid, predicted_mfes_valid)
        mse_valid_num_hairpins = mean_squared_error(actual_num_hairpins_valid, predicted_num_hairpins_valid)

        ce_valid = running_ce/len(list(valid_dataloader))
        total_valid_loss = ce_valid + mse_valid_mfe + mse_valid_num_hairpins 
        valid_loss_mse_mfe.append(mse_valid_mfe)
        valid_loss_mse_num_hairpins.append(mse_valid_num_hairpins)
        valid_loss_ce.append(ce_valid)
        list_valid_loss_total.append(total_valid_loss)
        
        writer.add_scalar('MSE-mfe-Loss/Train', mse_train_mfe, epoch+1)
        writer.add_scalar('MSE-mfe-Loss/Valid', mse_valid_mfe, epoch+1)
        writer.add_scalar('MSE-num_hairpins-Loss/Train', mse_train_num_hairpins, epoch+1)
        writer.add_scalar('MSE-num_hairpins-Loss/Valid', mse_valid_num_hairpins, epoch+1)
        writer.add_scalar('CE-Loss/Train', ce_train, epoch+1)
        writer.add_scalar('CE-Loss/Valid', ce_valid, epoch+1)
        writer.add_scalar('Total-Loss/Train', total_train_loss, epoch+1)
        writer.add_scalar('Total-Loss/Valid', total_valid_loss, epoch+1)
        
        ################
        #TRACKING per interval.
        ################ 
        last_epoch = epoch
        
        if epoch % interval == 0 and normalized:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            plot_predvsactual(writer, actual_mfes_train, predicted_mfes_train, epoch, mse_train_mfe, "Train", "mfe")        
            plot_predvsactual(writer, actual_mfes_valid, predicted_mfes_valid, epoch, mse_valid_mfe, "Valid", "mfe")
            plot_predvsactual(writer, actual_num_hairpins_train, predicted_num_hairpins_train, epoch, mse_train_num_hairpins, "Train", "num_hairpins")        
            plot_predvsactual(writer, actual_num_hairpins_valid, predicted_num_hairpins_valid, epoch, mse_valid_num_hairpins, "Valid", "num_hairpins")
                 
        ################
        #EARLY STOP AND CHECKPOINT
        ################ 
        if total_valid_loss < best_valid_loss and normalized:
            best_predicted_mfes_valid = predicted_mfes_valid
            best_predicted_num_hairpins_valid = predicted_num_hairpins_valid
            best_epoch = epoch
            best_valid_loss = total_valid_loss
            best_mse_valid_mfe = mse_valid_mfe
            best_mse_valid_num_hairpins = mse_valid_num_hairpins
            counter = 0
            best_model_path = model_checkpoint_path
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_model_path)
        elif normalized:
            counter += 1
        
        
        if counter > early_stop:
            print(f"Stopped early at epoch {epoch+1}")
            plot_predvsactual(writer, actual_mfes_valid, best_predicted_mfes_valid, epoch, best_mse_valid_mfe, "Valid", "mfe")
            plot_predvsactual(writer, actual_num_hairpins_valid, best_predicted_num_hairpins_valid, epoch, best_mse_valid_num_hairpins, "Valid", "num_hairpins")
            break
        

        ################
        #Output LOSS for visualization in terminal
        ################ 
        print(f'Epoch {epoch+1}/{n_epochs}, MSE-mfe Train Loss: {mse_train_mfe}, MSE-num_hairpins Train Loss: {mse_train_num_hairpins}, CE Train Loss: {ce_train}, \n MSE valid loss: {mse_valid_mfe}, MSE-num_hairpins Valid Loss: {mse_valid_num_hairpins}, CE valid loss: {ce_valid}')

 
    ################
    #TRACKING.
    ################ 
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_mse_mfe, label='Training Loss', color='blue')
    plt.plot(valid_loss_mse_mfe, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE-mfe Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('MSE-mfe Training and Validation Losses Over Epochs', plt.gcf())
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_mse_num_hairpins, label='Training Loss', color='blue')
    plt.plot(valid_loss_mse_num_hairpins, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE-num_hairpins Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('MSE-num_hairpins Training and Validation Losses Over Epochs', plt.gcf())
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_ce, label='Training Loss', color='blue')
    plt.plot(valid_loss_ce, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('CE Loss')
    plt.title('CE Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('CE Training and Validation Losses Over Epochs', plt.gcf())
    
    plt.figure(figsize=(10, 5))
    plt.plot(list_train_loss_total, label='Total training loss', color='blue')
    plt.plot(list_valid_loss_total, label='Total valid loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('total loss')
    plt.title('Total loss for Training and Validation Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('Total loss for Training and Validation Over Epochs', plt.gcf())
    
    writer.add_text('Configuration', f"Last epoch: {last_epoch+1}, Best epoch: {best_epoch+1}, Learning Rate: {config['learning_rate']}, Scheduler: None, Batch Size: {config['batch_size']}, \
                    dropout: {config['dropout']}, d_model: {config['d_model']}, d_ff: {config['d_ff']}, Encoder Layers: {config['layers_encoder']}, Decoder Layers: {config['layers_decoder']}, heads: {config['heads']}, \
                    max_len: {config['max_len']}")
     
    ################
    #Closing writer
    ################
    writer.close() 
    
    ################
    #Hyperparam tune based on minimizing sum of all losses.
    ################  
    return best_valid_loss, best_model_path
        
def plot_predvsactual(writer, actual, pred, epoch, mse, data_origin: str, measurement: str):
        #Plot predicted vs actual for train data every 20 epoch
        plt.figure(figsize=(8, 8))
        plt.scatter(actual, pred, alpha=0.5)
        plt.xlabel(f'Actual {measurement}')
        plt.ylabel(f'Predicted {measurement}')
        plt.title(f'Predicted vs Actual {measurement}')
        plt.grid(True)
        
        if data_origin != "Test":
            plt.text(x=0.05, y=0.95, s=f'Epoch: {epoch}', transform=plt.gca().transAxes, fontsize=12, ha='left', va='top')
        plt.text(x=0.05, y=0.85, s=f'MSE: {mse:.4f}', transform=plt.gca().transAxes, fontsize=12, ha='left', va='top')

        writer.add_figure(f'{data_origin} data: Predicted vs Actual {measurement}', plt.gcf(), epoch)
        
        
def evaluate_model(config, test_dataloader, model_path): 
    model = Transformer(config)  
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()  
    device = config['device']
    
    ce_weights = torch.tensor([0.0, 2.0, 2.0, 1.0, 0.0])
    ce_weights = ce_weights.to(device)
    loss_function_mse = nn.MSELoss()
    CELoss = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=0)   
    reverse_mapping = {1: '(', 2: ')', 3: '.', 4: '@'}
    
    predicted_mfes_test = np.zeros(len(test_dataloader.dataset))
    actual_mfes_test = np.zeros(len(test_dataloader.dataset))
    predicted_num_hairpins_test = np.zeros(len(test_dataloader.dataset))
    actual_num_hairpins_test = np.zeros(len(test_dataloader.dataset))

    it=0
    running_ce=0
    print_prediction=True
    with torch.no_grad():
        for batch in test_dataloader:            
            sq, mfe, target, num_hairpins = batch                            
            sq, mfe, target, num_hairpins = sq.to(device), mfe.to(device), target.to(device), num_hairpins.to(device)
            
            target_input = target[:, :-1]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(sq, target_input)
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)
            
            mfes, predicted_structures, structures_prob, predicted_num_hairpins = model(sq,target_input,src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            predicted_indices = torch.argmax(structures_prob, dim=-1)
            
            if print_prediction:
                predicted = predicted_indices[0:10].tolist()
                target_structure = target[0:10].tolist()
                decoded_predicted_structures = [''.join(reverse_mapping[symbol] for symbol in structure if symbol in reverse_mapping) for structure in predicted]
                target_structure_10 = [''.join(reverse_mapping[symbol] for symbol in structure if symbol in reverse_mapping) for structure in target_structure]
                    
                print("Predicted structures: ")
                for j,decoded_structure in enumerate(decoded_predicted_structures):
                    print(decoded_structure)
                    print(mfes[j])
                        
                print("Target structures: ")
                for j,target_ in enumerate(target_structure_10):
                    print(target_)
                    print(mfe[j])
                        
                print_prediction=False
                
            target_out = target[:, 1:]
                      
            ce_loss = CELoss(predicted_structures.contiguous().view(-1, 5), target_out.contiguous().view(-1))
            running_ce += ce_loss.item()

            pred_mfe = mfes.detach().cpu().numpy()
            act_mfe = mfe.detach().cpu().numpy()

            pred_num_hairpins = predicted_num_hairpins.detach().cpu().numpy()
            act_num_hairpins = num_hairpins.detach().cpu().numpy()

            predicted_mfes_test[it*len(pred_mfe):(it+1)*len(pred_mfe)] = pred_mfe.flatten()
            predicted_num_hairpins_test[it*len(pred_num_hairpins):(it+1)*len(pred_num_hairpins)] = pred_num_hairpins.flatten()

            actual_mfes_test[it*len(act_mfe):(it+1)*len(act_mfe)] = act_mfe.flatten()
            actual_num_hairpins_test[it*len(act_num_hairpins):(it+1)*len(act_num_hairpins)] = act_num_hairpins.flatten()

            it+=1
        
    mse_test_mfe = mean_squared_error(actual_mfes_test, predicted_mfes_test)
    ce_test = running_ce/len(test_dataloader)
    mse_test_num_hairpins = mean_squared_error(actual_num_hairpins_test, predicted_num_hairpins_test)
    
    total_test_loss = mse_test_mfe + ce_test + mse_test_num_hairpins
    
    print(f'MSE-mfe: {mse_test_mfe}, CE: {ce_test}, MSE-num_hairpins: {mse_test_num_hairpins}')
    
    #plot_predvsactual(writer, actual_mfes_test, predicted_values_test, 0, mse_test_mfe, "Test", "mfe")
    #plot_predvsactual(writer, actual_num_hairpins_test, predicted_num_hairpins_test, 0, mse_test_num_hairpins, "Test", "num_hairpins")
    
    return predicted_mfes_test, actual_mfes_test


    
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz)) == 1)).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def subsequent_mask(size):
    """Create a boolean mask for subsequent positions (size x size)."""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool)

    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    
    """
    IGNORE_VALUE = -1e9  
    src_padding_mask = src_padding_mask.float()
    tgt_padding_mask = tgt_padding_mask.float()
    
    src_padding_mask[src_padding_mask == 1] = IGNORE_VALUE
    tgt_padding_mask[tgt_padding_mask == 1] = IGNORE_VALUE
    """
        
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def create_target_mask(tgt, pad_token, device):
    batch_size = tgt.size(0)
    seq_len = tgt.size(1)
    tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    tgt_padding_mask = (tgt == pad_token)

    return tgt_mask.to(device), tgt_padding_mask.to(device)

def generate_sequences(model, src, actual_lengths, config):
    device = config['device']
    src = src.to(device)
    actual_lengths = actual_lengths.to(device)
    
    batch_size, max_len = src.size()  
    tgt = torch.full((batch_size, 1), 4, dtype=torch.long).to(device)  
    active = torch.ones(batch_size, dtype=torch.bool).to(device)  
    
    src_mask, no, src_padding_mask, no = create_mask(src,tgt)
    src_mask, src_padding_mask = src_mask.to(device), src_padding_mask.to(device)

    for _ in range(max_len - 1): 
        tgt_mask, tgt_padding_mask = create_target_mask(tgt, 0, device)
        mfes, predicted_structures, structures_prob, predicted_num_hairpins = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        next_tokens = structures_prob[:, -1, :].argmax(dim=-1) 

        tgt = torch.cat([tgt, torch.where(active.unsqueeze(-1), next_tokens.unsqueeze(-1), torch.full_like(next_tokens.unsqueeze(-1), 0))], dim=1)

        active &= (tgt.size(1) < actual_lengths)

        if not active.any():
            break

    return tgt

def evaluate_on_test(model_path, test_dataloader, config):
    batch_iterator_test = tqdm(test_dataloader, desc=f"Testing")
    running_ce=0
    ce_weights = torch.tensor([0.0, 2.0, 2.0, 1.0, 0.0])
    ce_weights = ce_weights.to(config['device'])
    CELoss = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=0)  
    interval=0
    result = []
    with torch.no_grad():
        for batch in batch_iterator_test:            
            sq, mfe, target, num_hairpins = batch
            actual_lengths = get_actual_lengths(sq, 0)
            model = Transformer(config)  
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
            model.eval()  
            predicted_struct = generate_sequences(model, sq, actual_lengths, config).tolist()
            for p in predicted_struct:
                result.append(p)
            
    return result
            

def get_actual_lengths(sequence_tensor, pad_token):
    non_padding_mask = sequence_tensor != pad_token

    actual_lengths = non_padding_mask.sum(dim=1)

    return actual_lengths
            
            

"""
def generate_secondary_structures_batch(model, batch, eos_token_id, pad_token_id):
        model.eval()
        with torch.no_grad():
            sq, mfe, target, num_hairpins, mask = batch     
            sq, mfe, target, num_hairpins, mask = sq.to(device), mfe.to(device), target.to(device), num_hairpins.to(device), mask.to(device)
            mfes, predicted_structures, structures_prob, predicted_num_hairpins = model(sq, mask)
            
            predictions = torch.argmax(structures_prob, dim=-1)
            predicted_structures = []

            for i in range(predictions.shape[0]):
                predicted_structure = []
                for token_id in predictions[i]:
                    if token_id.item() == eos_token_id:
                        break
                    elif token_id.item() != pad_token_id:
                        predicted_structure.append(token_id.item())
                predicted_structures.append(predicted_structure)

        return predicted_structures
"""