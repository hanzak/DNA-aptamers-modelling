import transformer
import config
import dataset 
import pickle
import os
from BucketDataLoader import BucketDataLoader

train_path = 'packages/data/train_2p5M.pkl'
valid_path = 'packages/data/valid_2p5M.pkl'
test_path = 'packages/data/test_2p5M.pkl'

def check_file(file_path):
    return os.path.isfile(file_path) and os.stat(file_path).st_size > 0

if check_file(train_path) and check_file(valid_path) and check_file(test_path):
    with open(train_path, 'rb') as f:
        train = pickle.load(f)
    with open(valid_path, 'rb') as f:
        valid = pickle.load(f)
    with open(test_path, 'rb') as f:
        test = pickle.load(f)
else:
    if check_file("data_2p5M.pkl") == False:
        raise FileNotFoundError("data_2p5M.pkl doesn't exist or is empty")
    with open('data_2p5M.pkl', 'rb') as file:
        data_for_transformer = pickle.load(file)
    train, valid, test = dataset.data_split(data_for_transformer, config.get_config())
    
    with open(train_path, 'wb') as f:
        pickle.dump(train, f)
    with open(valid_path, 'wb') as f:
        pickle.dump(valid, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test, f)
        
train_dataloader = BucketDataLoader(train, config.get_config())
valid_dataloader = BucketDataLoader(valid, config.get_config())
test_dataloader = BucketDataLoader(test, config.get_config())

transformer.train_model(config.get_config(), train_dataloader, valid_dataloader, test_dataloader)
