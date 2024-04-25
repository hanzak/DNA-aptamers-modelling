import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, confusion_matrix
import random

from tqdm import tqdm
import datetime
import os

"""
We set a seed for reproducibility
"""
seed = 22  
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PositionalEncoding(nn.Module):
    """
    Absolute Positional Encoding class. Followed on this video: https://www.youtube.com/watch?v=ISNdQcPhsts&t=4723s
    """

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
    """
    Defines a Transformer with only an encoder
    """
    
    def __init__(self,config):
        super(Transformer,self).__init__()
        self.d_model = config["d_model"]
        self.src_vocab_size = config["src_vocab_size"]
        self.heads = config["heads"]
        self.layers_encoder = config["layers_encoder"]
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
                        
        self.embedding1 = nn.Embedding(self.src_vocab_size, self.d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.heads,
            dim_feedforward  = self.d_ff,
            dropout  = self.dropout, 
            batch_first = True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.layers_encoder)
                
        self.mfe_out = nn.Linear(self.d_model, 1)
        self.num_hairpins_out = nn.Linear(self.d_model, 1)
        
        self.init_weights()
        
        self.to(self.device)
        
    def forward(self, src1, src_mask, src_padding_mask):      
        """
        Forward pass of the transformer

        Parameters:
        src1: Sequence input to the encoder
        src_mask: Source mask  (actually masks nothing in this context)
        src_padding_mask: Mask padding tokens
        """           
        src1 = self.embedding1(src1) * math.sqrt(self.d_model)
        src1 = self.positional_encoder(src1)
                
        src = src1
        
        encoder_output = self.transformer_encoder(src,mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        ######
        # We take the average while disregarding padding tokens 
        non_padded_count = torch.sum(~src_padding_mask, dim=1, keepdim=True)
        sum_encoder_out = torch.sum(encoder_output * ~src_padding_mask.unsqueeze(-1), dim=1)
        mean_encoder_out = sum_encoder_out / non_padded_count
        ######

        mfe = self.mfe_out(mean_encoder_out)
        num_hairpins_pred = self.num_hairpins_out(mean_encoder_out)  

        return mfe, num_hairpins_pred 
    
    
    
    def init_weights(self):
        """
        Initialize weights
        """
        nn.init.xavier_uniform_(self.embedding1.weight)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        
def train_model(config, train_dataloader, valid_dataloader, model_checkpoint_path):
    """
    Function that performs both Training and validation.

    Parameters:
    config: the configuration dictionnary
    train_dataloader (Dataloader): dataloader containing training set
    valid_dataloader (Dataloader): dataloader containing validation set
    model_checkpoint_path (string): path to save model checkpoint

    Returns:
    best_valid_loss (float): The best total validation loss
    best_model_path (string): Path to the best model
    """

    #Gets cuda device if available
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    
    ID = random.randint(1,100000)
    
    ################
    #Setting up variables
    ################
    early_stop = 4
    best_valid_loss = 1e9
    counter = 0
    train_loss_mse_mfe = []
    train_loss_mse_num_hairpins = []
    valid_loss_mse_mfe = []
    valid_loss_mse_num_hairpins = []
    list_train_loss_total = []
    list_valid_loss_total = []
    last_epoch = 0
    best_epoch = 0
    interval=10
    start_epoch = 0
    n_epochs = config['num_epochs']
    normalize_epochs = 3
    
    ##
    # We utilize numpy arrays in an effort to save memory when training with larger datasets.
    ##
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
    loss_function_mse = nn.MSELoss()
    
    #Get the total number of parameters of the current model
    print(get_n_params(model))
    
    ################
    #Tensorboard output folder and init
    ################
    start_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    foldername = str(ID) + "_" + str({config['data_size']}) + "_mix_no-decoder_complex_lr_" + str(config['learning_rate']) + "_batchsize_" + str(config['batch_size']) + "_dropout_" + str(config['dropout']) + "_" + str(start_time)
    log_dir = os.path.join(config['exp_name']+config['data_size'], foldername)
    os.makedirs(log_dir, exist_ok=True)
    writer = tb.SummaryWriter(log_dir=log_dir)
        
    ################
    #Loop for EPOCHS.
    ################ 
    first_epoch = True
    
    #Initilize baseline losses for normalization
    baseline_mse_mfe = 0
    baseline_mse_num_hairpins = 0
    normalized=False
        
    for epoch in range(start_epoch, n_epochs):
            
        ################
        #TRAINING
        ################ 
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Training epoch {epoch+1:02d}")
        
        print(len(train_dataloader.dataset))
        
        #iterator used in an attempt to save memory. using nparray instead of lists.
        it=0
        for i,batch in enumerate(batch_iterator):
            sq, mfe, stuct, num_hairpins = batch
                        
            #This is necessary for some obscure reason
            mfe=mfe.float()
            num_hairpins=num_hairpins.float()
            
            optimizer.zero_grad()
            
            sq, mfe, num_hairpins = sq.to(device), mfe.to(device), num_hairpins.to(device)
            src_mask, src_padding_mask = create_mask(sq)
            src_mask, src_padding_mask = src_mask.to(device), src_padding_mask.to(device)
            mfes, predicted_num_hairpins = model(sq, src_mask, src_padding_mask)
                                    
            loss_mfe = loss_function_mse(mfes, mfe)
            loss_num_hairpins = loss_function_mse(predicted_num_hairpins, num_hairpins)                        
            
            ###
            # Normalization of loss takes place here. We first gather baseline loss over |normalize_epochs| number of times.
            ###
            if epoch <= normalize_epochs:                                        
                baseline_mse_mfe += loss_mfe.item()
                baseline_mse_num_hairpins += loss_num_hairpins.item()

                total_loss = loss_mfe + loss_num_hairpins 
            else:
                if not normalized:
                    num_batches = len(train_dataloader) * normalize_epochs
                    baseline_mse_mfe /= num_batches
                    baseline_mse_num_hairpins /= num_batches
                    normalized = True
                    
                normalized_mse_mfe = loss_mfe / baseline_mse_mfe
                normalized_mse_num_hairpins = loss_num_hairpins / baseline_mse_num_hairpins
                            
                #We minimize the total normalized loss
                total_loss = normalized_mse_mfe + normalized_mse_num_hairpins
            
            
            #We minimize the total loss for the first |normalize_epochs|
            total_loss = loss_mfe + loss_num_hairpins 
                                            
            total_loss.backward()
            optimizer.step()
            
            ###
            #Gather and keep track of predictions in numpy arrays.
            #This is done to calculate the actual MSE loss, and not estimate it.
            ###
            pred_mfe = mfes.detach().cpu().numpy()
            pred_num_hairpins = predicted_num_hairpins.detach().cpu().numpy()
                        
            predicted_mfes_train[it*len(pred_mfe):(it+1)*len(pred_mfe)] = pred_mfe.flatten()
            predicted_num_hairpins_train[it*len(pred_num_hairpins):(it+1)*len(pred_num_hairpins)] = pred_num_hairpins.flatten()
            
            if first_epoch:
                act_mfe = mfe.detach().cpu().numpy()
                act_num_hairpins = num_hairpins.detach().cpu().numpy()
                actual_mfes_train[it*len(act_mfe):(it+1)*len(act_mfe)] = act_mfe.flatten()
                actual_num_hairpins_train[it*len(act_num_hairpins):(it+1)*len(act_num_hairpins)] = act_num_hairpins.flatten()

            #Iteration step   
            it+=1

        #Calculate the actual MSE for MFE and number of hairpins after training
        mse_train_mfe = mean_squared_error(actual_mfes_train, predicted_mfes_train)
        mse_train_num_hairpins = mean_squared_error(actual_num_hairpins_train, predicted_num_hairpins_train)
        
        #Keep track of losses
        total_train_loss = mse_train_mfe + mse_train_num_hairpins
        train_loss_mse_mfe.append(mse_train_mfe)
        train_loss_mse_num_hairpins.append(mse_train_num_hairpins)
        list_train_loss_total.append(total_train_loss)
        
        #########################
        #VALIDATION PER EPOCH
        #########################
        model.eval()
        batch_iterator_valid = tqdm(valid_dataloader, desc=f"Validating epoch {epoch+1:02d}")
        
        it=0
        with torch.no_grad():
            for batch in batch_iterator_valid:                
                sq, mfe, stuct, num_hairpins = batch
            
                #This is necessary for some obscure reason
                mfe=mfe.float()
                num_hairpins=num_hairpins.float()

                sq, mfe, num_hairpins = sq.to(device), mfe.to(device), num_hairpins.to(device)
                src_mask, src_padding_mask = create_mask(sq)
                src_mask, src_padding_mask = src_mask.to(device), src_padding_mask.to(device)
                mfes, predicted_num_hairpins = model(sq, src_mask, src_padding_mask)

                pred_mfe = mfes.detach().cpu().numpy()
                act_mfe = mfe.detach().cpu().numpy()

                pred_num_hairpins = predicted_num_hairpins.detach().cpu().numpy()
                act_num_hairpins = num_hairpins.detach().cpu().numpy()

                ###
                #Gather and keep track of predictions in numpy arrays.
                #This is done to calculate the actual MSE loss, and not estimate it.
                ###
                predicted_mfes_valid[it*len(pred_mfe):(it+1)*len(pred_mfe)] = pred_mfe.flatten()
                predicted_num_hairpins_valid[it*len(pred_num_hairpins):(it+1)*len(pred_num_hairpins)] = pred_num_hairpins.flatten()

                if first_epoch:
                    actual_mfes_valid[it*len(act_mfe):(it+1)*len(act_mfe)] = act_mfe.flatten()
                    actual_num_hairpins_valid[it*len(act_num_hairpins):(it+1)*len(act_num_hairpins)] = act_num_hairpins.flatten()

                it+=1

        
        first_epoch = False
        
        #Calculate actual MSE for MFE and number of hairpins after validation
        mse_valid_mfe = mean_squared_error(actual_mfes_valid, predicted_mfes_valid)
        mse_valid_num_hairpins = mean_squared_error(actual_num_hairpins_valid, predicted_num_hairpins_valid)

        #Keep track of losses
        total_valid_loss = mse_valid_mfe + mse_valid_num_hairpins 
        valid_loss_mse_mfe.append(mse_valid_mfe)
        valid_loss_mse_num_hairpins.append(mse_valid_num_hairpins)
        list_valid_loss_total.append(total_valid_loss)
        
        #Write the losses in tensorboard
        writer.add_scalar('MSE-mfe-Loss/Train', mse_train_mfe, epoch+1)
        writer.add_scalar('MSE-mfe-Loss/Valid', mse_valid_mfe, epoch+1)
        writer.add_scalar('MSE-num_hairpins-Loss/Train', mse_train_num_hairpins, epoch+1)
        writer.add_scalar('MSE-num_hairpins-Loss/Valid', mse_valid_num_hairpins, epoch+1)
        writer.add_scalar('Total-Loss/Train', total_train_loss, epoch+1)
        writer.add_scalar('Total-Loss/Valid', total_valid_loss, epoch+1)
        
        ################
        #TRACKING per interval.
        ################ 
        last_epoch = epoch
        
        #We track at each interval
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
        if total_valid_loss < best_valid_loss:
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
        else:
            counter += 1
        
        #If we do not see improvement after some time, stop early.
        if counter > early_stop:
            print(f"Stopped early at epoch {epoch+1}")
            plot_predvsactual(writer, actual_mfes_valid, best_predicted_mfes_valid, epoch, best_mse_valid_mfe, "Valid", "mfe")
            plot_predvsactual(writer, actual_num_hairpins_valid, best_predicted_num_hairpins_valid, epoch, best_mse_valid_num_hairpins, "Valid", "num_hairpins")
            break
        

        ################
        #Output LOSS for visualization in terminal
        ################ 
        print(f'Epoch {epoch+1}/{n_epochs}, MSE-mfe Train Loss: {mse_train_mfe}, MSE-num_hairpins Train Loss: {mse_train_num_hairpins}, \n MSE valid loss: {mse_valid_mfe}, MSE-num_hairpins Valid Loss: {mse_valid_num_hairpins}')

 
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
    plt.plot(list_train_loss_total, label='Total training loss', color='blue')
    plt.plot(list_valid_loss_total, label='Total valid loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('total loss')
    plt.title('Total loss for Training and Validation Over Epochs')
    plt.legend()
    plt.grid(True)
    
    writer.add_figure('Total loss for Training and Validation Over Epochs', plt.gcf())
    
    writer.add_text('Configuration', f"Last epoch: {last_epoch+1}, Best epoch: {best_epoch+1}, Learning Rate: {config['learning_rate']}, Scheduler: None, Batch Size: {config['batch_size']}, \
                    dropout: {config['dropout']}, d_model: {config['d_model']}, d_ff: {config['d_ff']}, Encoder Layers: {config['layers_encoder']}, heads: {config['heads']}, \
                    max_len: {config['max_len']}")
     
    ################
    #Closing writer
    ################
    writer.close() 
    
    
    ################
    #Hyperparam based on minimizing sum of all losses.
    ################  
    return best_valid_loss, best_model_path
        
def plot_predvsactual(writer, actual, pred, epoch, mse, data_origin: str, measurement: str):
    """
    Plots and add predicted vs actual plots to tensorboard. Utilized to plot various interactions inside loop.

    Args:
        writer (Tensorboard writer): A tensorboard writer
        actual (np.array): Actual values 
        pred (array): Predicted values 
        epoch (int): Epoch value at time of plotting
        mse (float): MSE value
        data_origin (str): Name of the dataset
        measurement (str): The measurement being compared
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, pred, alpha=0.5)
    plt.xlabel(f'Actual {measurement}')
    plt.ylabel(f'Predicted {measurement}')
    plt.title(f'Predicted vs Actual {measurement}')
    plt.grid(True)
    
    plt.text(x=0.05, y=0.85, s=f'MSE: {mse:.4f}', transform=plt.gca().transAxes, fontsize=12, ha='left', va='top')

    writer.add_figure(f'{data_origin} data: Predicted vs Actual {measurement}', plt.gcf(), epoch)
        
        
def evaluate_model(test_dataloader, model_path, config): 
    """
    Evalutes the model on test dataset.

    Args:
        test_dataloader (Dataloader): Test dataloader containing the test dataset.
        model_path (string): The path to the model we want to test.
        config (dictionnary): The config dictionnary
    """
    model = Transformer(config)  
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()  
    device = config['device']
        
    predicted_mfes_test = np.zeros(len(test_dataloader.dataset))
    actual_mfes_test = np.zeros(len(test_dataloader.dataset))
    predicted_num_hairpins_test = np.zeros(len(test_dataloader.dataset))
    actual_num_hairpins_test = np.zeros(len(test_dataloader.dataset))

    it=0
    with torch.no_grad():
        for batch in test_dataloader:            
            sq, mfe, struct, num_hairpins = batch
            
            #Again, just don't touch that. Respect the gods.
            mfe=mfe.float()
            num_hairpins=num_hairpins.float()
            
            sq, mfe, num_hairpins = sq.to(device), mfe.to(device), num_hairpins.to(device)
            
            src_mask, src_padding_mask = create_mask(sq)
            src_mask, src_padding_mask = src_mask.to(device), src_padding_mask.to(device)
            mfes, predicted_num_hairpins = model(sq, src_mask, src_padding_mask)
                                                          
            pred_mfe = mfes.detach().cpu().numpy()
            act_mfe = mfe.detach().cpu().numpy()

            pred_num_hairpins = predicted_num_hairpins.detach().cpu().numpy()
            act_num_hairpins = num_hairpins.detach().cpu().numpy()
            
            predicted_mfes_test[it*len(pred_mfe):(it+1)*len(pred_mfe)] = pred_mfe.flatten()
            predicted_num_hairpins_test[it*len(pred_num_hairpins):(it+1)*len(pred_num_hairpins)] = pred_num_hairpins.flatten()

            actual_mfes_test[it*len(act_mfe):(it+1)*len(act_mfe)] = act_mfe.flatten()
            actual_num_hairpins_test[it*len(act_num_hairpins):(it+1)*len(act_num_hairpins)] = act_num_hairpins.flatten()

            it+=1
        
    #Calculates actual MSE loss for MFE and number of hairpins
    mse_test_mfe = mean_squared_error(actual_mfes_test, predicted_mfes_test)
    mse_test_num_hairpins = mean_squared_error(actual_num_hairpins_test, predicted_num_hairpins_test)
        
    #Print to see the results. Result is not saved in file but could be.
    print(f'MSE-mfe: {round(mse_test_mfe,6)}, MSE-num_hairpins: {round(mse_test_num_hairpins,6)}')

def create_mask(src):
    """
    Create masks for inputed sequence

    Args:
        src (string): string sequence, usually padded and ready to input to a Transformer

    Returns:
        src_mask (tensor): Tensor that masks the padding
        src_padding_mask (tensor): Tensor that masks nothing. But could eventually masks something, if it believes hard enough.
    """
    src_seq_len = src.shape[1]

    src_mask = torch.zeros(src_seq_len, src_seq_len).bool().type(torch.bool)

    src_padding_mask = (src == 0).type(torch.bool)
        
    return src_mask, src_padding_mask

def get_n_params(model):
    """
    Gets the number of parameter from the given model

    Args:
        model (nn.Module): pytorch nn.Module Model

    Returns:
        params (int): Number of parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params