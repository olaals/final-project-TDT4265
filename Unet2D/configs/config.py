

def get_config():
    config = {
        "model": "model"
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 8,
        "input_width": 500,
        "input_height": 500,
        
        "train_dir": "datasets/TTE/train"
        "val_dir": "datasets/TTE/val",
        "test_dir": "datasets/TTE/test",



    }
    return config





if __name__ == '__main__':
    print(get_config())
