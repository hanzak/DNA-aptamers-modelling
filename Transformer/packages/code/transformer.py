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
    
class NegativeReLU(nn.Module):
    """
    nn.Module class that defines a NegativeReLU
    """
    def forward(self, x):
        return -F.relu(-x)

class Transformer(nn.Module):
    """
    Defines a Transformer with Encoder&Decoder
    """
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
        self.sampling_p = 1.00
        self.decay_rate = 0.05
        
        self.model_type = "Transformer"
        
        self.positional_encoder = PositionalEncoding(
            d_model = self.d_model,
            max_len = self.max_len,
            dropout = self.dropout
        )
        
        #We use a different embedding for Encoder and Decoder
        self.embedding_encoder = nn.Embedding(self.src_vocab_size, self.d_model)
        self.embedding_decoder = nn.Embedding(self.target_vocab_size, self.d_model)
        
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
        
    def update_sampling_p(self):
        """
        Updates the sampling proportion
        """
        self.sampling_p *= (1-self.decay_rate)
        
    def mix_embeddings(self, first_decoder_output, tgt_emb, tgt_padding_mask):
        """
        Mixes the embeddings of the target structure and the predicted structure after a first pass through the decoder.

        Args:
            first_decoder_output (Tensor (batch_size, seq_len, target_vocab_size)): Output of after the first decoder pass
            tgt_emb (Tensor (batch_size, seq_len, d_model)): Target tensor
            tgt_padding_mask (Tensor (batch_size, seq_len)): Target padding tensor

        Returns:
            mixed_input (Tensor (batch_size, seq_len, d_model)): Mixed embedding to give as input to the decoder for a second pass.
        """
        softmax_scores = F.softmax(first_decoder_output, dim=-1)
        softmax_scores = softmax_scores * ~tgt_padding_mask.unsqueeze(-1)
        
        # Multiply each score by its corresponding embedding and sum them to get the mixed embedding
        weighted_sum_embeddings = torch.matmul(softmax_scores, self.embedding_decoder.weight)
        weighted_sum_embeddings = weighted_sum_embeddings.reshape(tgt_emb.shape)
        
        mixed_input = (1-self.sampling_p) * weighted_sum_embeddings + (self.sampling_p) * tgt_emb

        return mixed_input
                                     
    def forward(self, src, target, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, training=False):   
        """
        Forward pass of the transformer

        Parameters:
        src: Sequence input to the encoder
        target: Target input to the decoder
        src_mask: Source mask  (actually masks nothing in this context)
        src_mask: Target mask. Avoids looking into future tokens.
        src_padding_mask: Mask padding tokens in sequence
        tgt_padding_mask: Mask padding tokens in target structure
        memory_key_padding_mask: Mask for the output of the encoder given to the decoder. Usually put src_padding_mask.
        training: Boolean value to determine if the model is training. Defaults to False.
        """                                
        src = self.embedding_encoder(src) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)
        
        encoder_output = self.transformer_encoder(src,mask=src_mask, src_key_padding_mask=src_padding_mask)

        target = self.embedding_decoder(target) * math.sqrt(self.d_model)
        target = self.positional_encoder(target)
                
        real_output = self.transformer_decoder(target, encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
        
        #If we are training, performed scheduled sampling (Mix embeddings)
        if training:
            decoded = self.decoder_out(real_output)
            mixed_input = self.mix_embeddings(decoded, target, tgt_padding_mask)
            real_output = self.transformer_decoder(mixed_input, encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)
              
        decoded = self.decoder_out(real_output)
        decoded_probs = F.softmax(decoded, dim=-1)  
        
        non_padded_count = torch.sum(~tgt_padding_mask, dim=1, keepdim=True)
        sum_decoder_out = torch.sum(real_output * ~tgt_padding_mask.unsqueeze(-1), dim=1)
        mean_decoder_out = sum_decoder_out / non_padded_count
        
        mfe = self.mfe_out(mean_decoder_out)
        num_hairpins_pred = self.num_hairpins_out(mean_decoder_out)  

        return mfe, decoded, decoded_probs, num_hairpins_pred 

    
    def init_weights(self):
        """
        Initialize weights
        """
        nn.init.xavier_uniform_(self.embedding_decoder.weight)
        nn.init.xavier_uniform_(self.embedding_encoder.weight)
        
        pad_index = 0 
        sos_index = 1  
        with torch.no_grad():
            self.embedding_decoder.weight[pad_index].zero_()
            self.embedding_decoder.weight[sos_index].zero_()
            self.embedding_encoder.weight[pad_index].zero_()
            self.embedding_encoder.weight[sos_index].zero_()
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
def train_model(config, train_dataloader, valid_dataloader, model_checkpoint_path):
    """
    Performs training and validation of the model.

    Args:
        config (dictionnary): Configuration dictionnary
        train_dataloader (Dataloader): Dataloader containing the training dataset
        valid_dataloader (Dataloader): Dataloader containing the validation dataset
        model_checkpoint_path (string): Path to save the model checkpoint

    Returns:
    best_valid_loss (float): The best total validation loss
    best_model_path (string): Path to the best model
    """
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
    reverse_mapping = {2: '(', 3: ')', 4: '.'}
    
    
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

    #We use weighted cross entropy loss. 
    #We give weights of 0 to both PAD and SOS tokens.
    #We give weights of 2 to both "(" and ")" tokens.
    #We give weights of 1 to both "." and EOS tokens
    ce_weights = torch.tensor([0.0, 0.0, 2.0, 2.0, 1.0, 1.0])
    ce_weights = ce_weights.to(device)
    loss_function_mse = nn.MSELoss()
    CELoss = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=0)
    
    print(f"Total number of parameters is: {get_n_params(model)}")
    
    ################
    #Tensorboard output folder and init
    ################
    start_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
    foldername = config['data_size'] + "_" + str(config['learning_rate']) + "_batchsize_" + str(config['batch_size']) + "_dropout_" + str(config['dropout']) + "_" + str(start_time)
    
    #Folders are created in Transformer/packages/model/runs under the folder with the same name as the data_size used for training
    log_dir = os.path.join(config['exp_name']+config['data_size'], foldername)
    os.makedirs(log_dir, exist_ok=True)
    writer = tb.SummaryWriter(log_dir=log_dir)
        
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
            
            optimizer.zero_grad()
            
            #We need to do this for some reasons.
            mfe=mfe.float()
            num_hairpins=num_hairpins.float()

            sq, mfe, target, num_hairpins = sq.to(device), mfe.to(device), target.to(device), num_hairpins.to(device)
            
            #We removed the last token before predictions, in accordance with the teacher forcing technique
            target_input = target[:, :-1]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(sq, target_input)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)
            mfes, predicted_structures, structures_prob, predicted_num_hairpins = model(sq,target_input,src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, training=True)

            predicted_indices = torch.argmax(structures_prob, dim=-1)
            
            #We removed the first token before loss calculations, in accordance with the teacher forcing technique
            target_out = target[:, 1:]
            
            loss_mfe = loss_function_mse(mfes, mfe)
            loss_num_hairpins = loss_function_mse(predicted_num_hairpins, num_hairpins)
            ce_loss = CELoss(predicted_structures.contiguous().view(-1, 6), target_out.contiguous().view(-1))

            ####
            #We normalize the losses here
            #We first gather a baseline loss for each losses.
            #We then use the baseline to normalize each loss, and minimize the total normalized loss.
            ####
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
                            
                #We minimize the total normalized loss
                total_loss = normalized_mse_mfe + normalized_mse_num_hairpins + normalized_ce

            running_ce += ce_loss.item()
                                
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
        
        #Update the sampling proportion after each epoch
        model.update_sampling_p()
        
        #Calculate the actual MSE for MFE and number of hairpins after training
        mse_train_mfe = mean_squared_error(actual_mfes_train, predicted_mfes_train)
        mse_train_num_hairpins = mean_squared_error(actual_num_hairpins_train, predicted_num_hairpins_train)
        
        #Keep track of losses
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
                
                #We removed the last token before predictions, in accordance with the teacher forcing technique
                target_input = target[:, :-1]
            
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(sq, target_input)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)
                mfes, predicted_structures, structures_prob, predicted_num_hairpins = model(sq,target_input,src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    
                predicted_indices = torch.argmax(structures_prob, dim=-1)
            
                #We removed the first token before loss calculations, in accordance with the teacher forcing technique
                target_out = target[:, 1:]     
                
                #Just to visualize secondary structure predictions during training
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
                        
                ce_loss = CELoss(predicted_structures.contiguous().view(-1, 6), target_out.contiguous().view(-1))
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
        
        #Calculate actual MSE for MFE and number of hairpins after validation
        mse_valid_mfe = mean_squared_error(actual_mfes_valid, predicted_mfes_valid)
        mse_valid_num_hairpins = mean_squared_error(actual_num_hairpins_valid, predicted_num_hairpins_valid)

        #Keep track of losses
        ce_valid = running_ce/len(list(valid_dataloader))
        total_valid_loss = ce_valid + mse_valid_mfe + mse_valid_num_hairpins 
        valid_loss_mse_mfe.append(mse_valid_mfe)
        valid_loss_mse_num_hairpins.append(mse_valid_num_hairpins)
        valid_loss_ce.append(ce_valid)
        list_valid_loss_total.append(total_valid_loss)
        
        #Write the losses in tensorboard
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
        if total_valid_loss < best_valid_loss and normalized:
            best_predicted_mfes_valid = predicted_mfes_valid
            best_predicted_num_hairpins_valid = predicted_num_hairpins_valid
            best_epoch = epoch
            best_valid_loss = total_valid_loss
            best_mse_valid_mfe = mse_valid_mfe
            best_mse_valid_num_hairpins = mse_valid_num_hairpins
            counter = 0
            best_model_path = model_checkpoint_path

            #Modele is saved in Transformer/packages/model/model_checkpoint under the folder with the same name as the data_size used
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_model_path)
        elif normalized:
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
    
    if data_origin != "Test":
        plt.text(x=0.05, y=0.95, s=f'Epoch: {epoch}', transform=plt.gca().transAxes, fontsize=12, ha='left', va='top')
    plt.text(x=0.05, y=0.85, s=f'MSE: {mse:.4f}', transform=plt.gca().transAxes, fontsize=12, ha='left', va='top')

    writer.add_figure(f'{data_origin} data: Predicted vs Actual {measurement}', plt.gcf(), epoch)

    
def generate_square_subsequent_mask(sz):
    """
    Generates square triangular matrix for masking
    """
    mask = (torch.triu(torch.ones((sz, sz)) == 1)).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def subsequent_mask(size):
    """
    Create a boolean mask for subsequent positions (size x size)
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


def create_mask(src, tgt):
    """
    Creates masking tensors for source and target

    Args:
        src (Tensor): Sequences
        tgt (Tensor): Targets
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool)

    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
        
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def create_target_mask(tgt, pad_token, device):
    """
    Creates masks for target structures.
    This function is used in greedy decode function.

    Args:
        tgt (Tesnor): Target structures
        pad_token (int): The pad token
        device (Device): The device

    Returns:
        tgt_mask: Triangular mask
        tgt_padding_mask: Padding mask
    """
    seq_len = tgt.size(1)
    tgt_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    tgt_padding_mask = (tgt == pad_token)

    return tgt_mask.to(device), tgt_padding_mask.to(device)

def generate_sequences(model, src, actual_lengths, config):
    """
    Greedy decode algorithm to generate sequences during inference

    Args:
        model (nn.Module): The model
        src (Tensor): Sequences
        actual_lengths (int): Actual lengths of the sequences in src
        config (dictionnary): Configuration 

    Returns:
        tgt (Tensor): Predicted structures
        mfes (Tensor): Predicted mfes
        predicted_num_hairpins (Tensor): Predicted number of hairpins
    """
    device = config['device']
    actual_lengths = actual_lengths.to(device)
    
    batch_size, max_len = src.size()  
    tgt = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)  

    #Tells us which sequence is still being generated
    active = torch.ones(batch_size, dtype=torch.bool).to(device)  
        
    src_mask, no, src_padding_mask, no = create_mask(src,tgt)
    src_mask, src_padding_mask = src_mask.to(device), src_padding_mask.to(device)

    #We loop a maximum of max_len+1 times (maximum sequence length + 1)
    for _ in range(max_len+1): 
        tgt_mask, tgt_padding_mask = create_target_mask(tgt, 0, device)
        mfes, logits, structures_prob, predicted_num_hairpins = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        next_tokens = structures_prob[:, -1, :].argmax(dim=-1) 
        
        tgt = torch.cat([tgt, torch.where(active.unsqueeze(-1), next_tokens.unsqueeze(-1), torch.full_like(next_tokens.unsqueeze(-1), 0))], dim=1)

        #If we have reached the actual length of the sequence, we stop generating (instead of checking for EOS which is also possible).
        #We do +2 to take into account the SOS and EOS in the predicted structure tgt.
        active &= (tgt.size(1) < actual_lengths+2)
        
        #If we have predicted a structure for all sequences, we break.
        if not active.any():
            break

    return tgt, mfes, predicted_num_hairpins


def generate_sequences_sim(model, src, target_out, actual_lengths, config):
    """
    This function is similar to the greedy decode, except here we simulate predictions at each step.
    This means we intentionnally give it a good or bad prediction every step.
    This function is only used to study the problems of greedy decode during inference.

    Args:
        model (nn.Module): The model
        target_out (Tensor): The real target structures
        src (Tensor): Sequences
        actual_lengths (int): Actual lengths of the sequences in src
        config (dictionnary): Configuration 

    Returns:
        _type_: _description_
    """

    #We specify a % of error. 
    random_error = 0.0
    device = config['device']
    actual_lengths = actual_lengths.to(device)
    
    batch_size, max_len = src.size() 
    tgt = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)  
    
    src_mask, no, src_padding_mask, no = create_mask(src, tgt)
    src_mask, src_padding_mask = src_mask.to(device), src_padding_mask.to(device)
        
    for step in range(max_len + 1): 
        tgt_mask, tgt_padding_mask = create_target_mask(tgt, 0, device)
        mfes, logits, structures_prob, predicted_num_hairpins = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        #We get the next token from the real target structures.
        #We have a (random chance = random_error) to change the real next token to a wrong one. 
        next_token = target_out[:, (step+1)]
        if torch.rand(1) < random_error:
            for i, t in enumerate(next_token):
                actual_token = t.item()
                if actual_token == 0 or actual_token == 1:
                    break
                random_token = actual_token
                while random_token==actual_token:
                    random_token = np.random.randint(2,5)
                target_out[i, (step+1)] = random_token
                
        tgt = target_out[:, :(step+2)]
                        
    return tgt, mfes, predicted_num_hairpins


def evaluate_model(test_dataloader, model_path, config):
    """
    Evalutes the model on test dataset.

    Args:
        test_dataloader (Dataloader): Test dataloader containing the test dataset.
        model_path (string): The path to the model we want to test.
        config (dictionnary): The config dictionnary

    Returns:
        result (list): List of predicted structures
        mse_test_mfe (float): MSE loss of MFE predictions
        mse_test_num_hairpins (float): MSE loss of number of hairpins predictions
    """
    batch_iterator_test = tqdm(test_dataloader, desc=f"Testing")
    
    predicted_mfes_test = np.zeros(len(test_dataloader.dataset))
    actual_mfes_test = np.zeros(len(test_dataloader.dataset))
    predicted_num_hairpins_test = np.zeros(len(test_dataloader.dataset))
    actual_num_hairpins_test = np.zeros(len(test_dataloader.dataset))
    
    model = Transformer(config)  
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()  
    
    it=0
    result = []
    with torch.no_grad():
        for batch in batch_iterator_test:            
            device = config['device']
            sq, mfe, target, num_hairpins = batch
            sq = sq.to(device)
            target = target.to(device)
            mfe = mfe.to(device)
            num_hairpins = num_hairpins.to(device)
            actual_lengths = get_actual_lengths(sq, 0)

            predicted_struct, mfes, predicted_num_hairpins = generate_sequences(model, sq, actual_lengths, config)
            #predicted_struct, mfes, predicted_num_hairpins = generate_sequences_sim(model, sq, target, actual_lengths, config)

            #We put the predicted structures in a lost called "result" to then calculate precision, recall and F1 
            predicted_struct = predicted_struct.tolist()
            for p in predicted_struct:
                result.append(p)
        
            
            pred_mfe = mfes.detach().cpu().numpy()
            act_mfe = mfe.detach().cpu().numpy()

            pred_num_hairpins = predicted_num_hairpins.detach().cpu().numpy()
            act_num_hairpins = num_hairpins.detach().cpu().numpy()

            predicted_mfes_test[it*len(pred_mfe):(it+1)*len(pred_mfe)] = pred_mfe.flatten()
            predicted_num_hairpins_test[it*len(pred_num_hairpins):(it+1)*len(pred_num_hairpins)] = pred_num_hairpins.flatten()

            actual_mfes_test[it*len(act_mfe):(it+1)*len(act_mfe)] = act_mfe.flatten()
            actual_num_hairpins_test[it*len(act_num_hairpins):(it+1)*len(act_num_hairpins)] = act_num_hairpins.flatten()

            it+=1
            
    #Calculate the real MSE for MFE and number of hairpins
    mse_test_mfe = mean_squared_error(actual_mfes_test, predicted_mfes_test)
    mse_test_num_hairpins = mean_squared_error(actual_num_hairpins_test, predicted_num_hairpins_test)
    
    return result, mse_test_mfe, mse_test_num_hairpins
            

def get_actual_lengths(sequence_tensor, pad_token):
    """
    Gets the actual lengths of sequences, ignoring padding tokens

    Args:
        sequence_tensor (Tensor): The sequences
        pad_token (int): The padding token 

    Returns:
        actual_lengths: the actual lengths of sequences, ignoring padding tokens
    """
    non_padding_mask = sequence_tensor != pad_token

    actual_lengths = non_padding_mask.sum(dim=1)

    return actual_lengths
            
    
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
        