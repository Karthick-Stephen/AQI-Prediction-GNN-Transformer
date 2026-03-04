# config.py

class Config:
    # Model hyperparameters
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100
    hidden_dim = 64
    dropout_rate = 0.2

    # Data settings
    data_path = './data/'
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    # Logging settings
    log_dir = './logs/'
    save_model_dir = './models/'

    # Others
    seed = 42

config = Config()