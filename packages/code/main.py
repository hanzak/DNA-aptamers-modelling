import model
import config
from dataset import data_split
import pickle
import os

train_path = '../data/train.pkl'
valid_path = '../data/valid.pkl'
test_path = '../data/test.pkl'

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
    if check_file("../data/data.pkl") == False:
        raise FileNotFoundError("data.pkl doesn't exist or is empty")
    with open('../data/data.pkl', 'rb') as file:
        data_for_transformer = pickle.load(file)
    train, valid, test = data_split(data_for_transformer, config.get_config())
    
    with open(train_path, 'wb') as f:
        pickle.dump(train, f)
    with open(valid_path, 'wb') as f:
        pickle.dump(valid, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test, f)

#model.train_model(config.get_config(), train, valid)
model.test_model(config.get_config(), test)



