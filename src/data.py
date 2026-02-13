from sklearn.datasets import load_wine

def load_data():
    """Load the Wine dataset and return features and labels."""
    wine = load_wine()
    return wine.data, wine.target, wine.feature_names, wine.target_names