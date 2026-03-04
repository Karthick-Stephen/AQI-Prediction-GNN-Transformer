import tensorflow as tf
from gnn_module import GNN
from transformer_module import Transformer
from data_loader import load_data
from metrics import evaluate_metrics

class AQIPredictionPipeline:
    def __init__(self):
        self.gnn = GNN()
        self.transformer = Transformer()

    def load_data(self):
        train_data, test_data = load_data()
        return train_data, test_data

    def train(self, train_data):
        features, labels = self.gnn.extract_features(train_data)
        predictions = self.transformer.train(features, labels)
        return predictions

    def evaluate(self, test_data):
        features, labels = self.gnn.extract_features(test_data)
        predictions = self.transformer.predict(features)
        metrics = evaluate_metrics(labels, predictions)
        return metrics

    def run_pipeline(self):
        train_data, test_data = self.load_data()
        self.train(train_data)
        metrics = self.evaluate(test_data)
        print(f"Evaluation Metrics: {metrics}")

if __name__ == '__main__':
    pipeline = AQIPredictionPipeline()
    pipeline.run_pipeline()