from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Order matters: the model is trained on columns in exactly this order.
FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_ratio",
    "proline",
]

CULTIVARS = ["cultivar_A", "cultivar_B", "cultivar_C"]


def load_data():
    """
    Load the Wine recognition dataset.
    Returns:
        X (numpy.ndarray): 178 x 13 matrix of chemical measurements.
        y (numpy.ndarray): Cultivar label (0, 1 or 2) for every sample.
    """
    wine = load_wine()
    return wine.data, wine.target


def split_data(X, y):
    """
    Stratified train/test split so every cultivar is represented in both sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
