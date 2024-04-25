import argparse
from utils import Utils
import sys
import os

class MainClass:
    """
    Main class handling user interactions
    """
    def __init__(self, model, action):
        self.action = action
        self.model = model

    def run(self):
        if self.action == "train":
            datasize = input("Which datasize to use: dummy, 250k, 1p5M, 2p5M, 5M \n Type here:")
            if datasize.strip() not in ['dummy', '250k', '1p5M', '2p5M', '5M']:
                print("Invalid choice")
                return
            print("You chose:", datasize)
            file_name = input(f"Specify 'file_name' for your model checkpoint: Transformer/packages/model/model_checkpoint/{datasize}/file_name.pth \n Type here:").strip()
            is_file = Utils.check_path(f"Transformer/packages/model/model_checkpoint/{datasize}/{file_name}.pth")
            if is_file:
                print("File already exists.")
                return
            folder_path = f"Transformer/packages/model/model_checkpoint/{datasize}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder created: {folder_path}")
            file_path = f"Transformer/packages/model/model_checkpoint/{datasize}/{file_name}.pth"
            Utils.train_model(self.model, datasize, file_path)
        elif self.action == "evaluate":
            datasize = input("Which datasize was used to train the model: dummy, 250k, 1p5M, 2p5M, 5M \n Type here:").strip()
            if datasize not in ['dummy', '250k', '1p5M', '2p5M', '5M']:
                print("Invalid choice")
                return
            print("You chose:", datasize)
            file_name = input(f"Specify 'file_name' : Transformer/packages/model/model_checkpoint/{datasize}/file_name.pth \n Type here:").strip()
            is_file = Utils.check_path(f"Transformer/packages/model/model_checkpoint/{datasize}/{file_name}.pth")
            if not is_file:
                print("Invalid path")
                return
            file_path = f"Transformer/packages/model/model_checkpoint/{datasize}/{file_name}.pth"
            Utils.test_model(self.model, file_path)
            

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Missing required arguments. Usage: main.py {encoder,decoder} {train,evaluate}")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Running program")
    parser.add_argument("action", type=str, choices=["train", "evaluate"], help="Action to take")
    parser.add_argument("model", type=str, choices=["encoder", "decoder"], help="Type of model to use")
    args = parser.parse_args()

    main_obj = MainClass(args.model, args.action)
    main_obj.run()