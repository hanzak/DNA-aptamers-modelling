import transformer
import transformer_no_decoder
import config
import dataset 
import pickle
import os
from BucketDataLoader import BucketDataLoader
import numpy as np
import json
from NpEncoder import *
from sklearn.metrics import classification_report

#Initializations
best_validation_loss = float(1e9)
best_model_path = None

#I needed that to write in JSON. dont know if i need it anymore, but better keep it.
np.int = int
    
class Utils():    
    """
    Util class containing function to make testing and running the code easier for external users.
    """
    def check_file(file_path):
        """
        Check if file exists and is not empty

        Args:
            file_path (string): Path to file

        Returns:
            Boolean: True if file exists
        """
        return os.path.isfile(file_path) and os.stat(file_path).st_size > 0
    
    def check_path(file_path):
        """
        Checks if file exists

        Args:
            file_path (string): Path to file

        Returns:
            Boolean: True if file exists
        """
        return os.path.isfile(file_path)


    def get_data_structures(data_name):
        """
        Gets datasets from specified data_name

        Args:
            data_name (string): Name corresponding to the size of the dataset of the data file we want
                All data files are assumed to be named train/valid/test_{data_name}_struct.pkl.
                All data files are assumed to be in data/splits/ folder

        Raises:
            FileNotFoundError: If file not found

        Returns:
            _type_: _description_
        """
        if Utils.check_file(f"data/splits/train_{data_name}_struct.pkl") and Utils.check_file(f"data/splits/valid_{data_name}_struct.pkl") and Utils.check_file(f"data/splits/test_{data_name}_struct.pkl"):
            with open(f"data/splits/train_{data_name}_struct.pkl", 'rb') as f:
                train = pickle.load(f)
            with open(f"data/splits/valid_{data_name}_struct.pkl", 'rb') as f:
                valid = pickle.load(f)
            with open(f"data/splits/test_{data_name}_struct.pkl", 'rb') as f:
                test = pickle.load(f)
        else:
            if Utils.check_file(f"data/data_{data_name}_struct.pkl") == False:
                raise FileNotFoundError(f"data_{data_name}_struct.pkl doesn't exist or is empty")
            with open(f'data/data_{data_name}_struct.pkl', 'rb') as file:
                data_for_transformer = pickle.load(file)

            data_for_transformer = dataset.count_hairpins(data_for_transformer)

            train, valid, test = dataset.data_split(data_for_transformer)

            with open(f"data/splits/train_{data_name}_struct.pkl", 'wb') as f:
                pickle.dump(train, f)
            with open(f"data/splits/valid_{data_name}_struct.pkl", 'wb') as f:
                pickle.dump(valid, f)
            with open(f"data/splits/test_{data_name}_struct.pkl", 'wb') as f:
                pickle.dump(test, f) 
        return train, valid, test


    def calculate_metrics(targets, predictions, mfes, num_hairpins, filename):
        """
        Calculates the relevant metrics from predictions of model

        Args:
            targets (list): List of real structures
            predictions (list): List of predicted structures
            mfes (float): The MSE loss of the MFE
            num_hairpins (num_hairpins): The MSE loss of the number of hairpins
            filename (string): Name of the .json file containing the resulting metrics
        """
        # Extract predicted and target sequences
        predicted_sequences = [item for item in predictions]
        target_sequences = [item for item in targets]

        y_true = np.concatenate(target_sequences)
        y_pred = np.concatenate(predicted_sequences)

        non_padding_mask = (y_true != 0) & (y_true != 1)

        # Apply the mask to filter out padding
        y_true_without_padding = y_true[non_padding_mask]
        y_pred_without_padding = y_pred[non_padding_mask]

        # Define your class labels
        class_labels = ['(', ')', '.', 'EOS']
        class_labels_no_PAD = ['(', ')', '.', 'EOS']

        try:
            report_dict = classification_report(y_true_without_padding, y_pred_without_padding, target_names=class_labels, output_dict=True)
        except:
            report_dict = classification_report(y_true_without_padding, y_pred_without_padding, target_names=class_labels_no_PAD, output_dict=True)

        metrics = {
            "report": report_dict,
            "mse_mfes": mfes,
            "mse_hairpins": num_hairpins
        }

        with open(f'results/{filename}_output_with_metrics.json', 'w') as file:
            json.dump(metrics, file, indent=4)


    def train_model(model, data_size, model_checkpoint_path):
        """
        Calls the relevant functions to train the model

        Args:
            model (string): Name of the model we want to train, "encoder" or "decoder"
            data_size (string): The size of the data we want to train the model on
            model_checkpoint_path (string): The path to save the model checkpoint

        Raises:
            ValueError: Value error if model is not either "encoder" or "decoder"
        """
        if model.strip() not in ('encoder', 'decoder'):
            raise ValueError("Model must be either 'encoder' or 'decoder'")
        if model=="encoder":
            config_ = config.get_config_no_decoder()
        elif model=="decoder":
            config_ = config.get_config()
        config_['data_size'] = data_size

        train, valid, test = Utils.get_data_structures(data_size)

        train_dataloader = BucketDataLoader(train, config_)
        valid_dataloader = BucketDataLoader(valid, config_)

        if model=="encoder":
            transformer_no_decoder.train_model(config_, train_dataloader, valid_dataloader, model_checkpoint_path)
        elif model=="decoder":
            transformer.train_model(config_, train_dataloader, valid_dataloader, model_checkpoint_path)
            
    
    def test_model(model, file_path):
        """
        Calls the relevant functions to test the model

        Args:
            model (string): Name of the model we want to train, "encoder" or "decoder"
            file_path (string): Path to the saved .pth model
        """
        with open(f'data/data_test.pkl', 'rb') as file:
            data = pickle.load(file)
            
        if model=="encoder":
            config_ = config.get_config_no_decoder()
        elif model=="decoder":
            config_ = config.get_config()
        
        new_data = []

        for d in data:
            sq,_,_,_ = d
            if len(sq)>=10 and len(sq) <=50:
                new_data.append(d)
        test_dataloader = BucketDataLoader(new_data, config_)
        
        if config_['type']=="decoder":
            targets = []
            for _, _, tgt, _ in test_dataloader:
                for t in tgt:
                    targets.append(t.tolist())
            result, mfes, num_hairpins = transformer.evaluate_model(test_dataloader, file_path, config_)
            Utils.calculate_metrics(targets, result, mfes, num_hairpins, f'test_{model}')
        else:
            pred_mfes, act_mfes, pred_hairpins, act_hairpins = transformer_no_decoder.evaluate_model(test_dataloader,file_path, config_)

