import numpy as np, os
from .csv_loader import CSVLoader, SETS
from .trainer import compute_metrics
from abc import ABC, abstractmethod


class TableClassifier(ABC):
    def __init__(self, loader: CSVLoader):
        self.loader  = loader
        self.METRICS = ("f1-score", "accuracy", "precision","recall", "auc")
        
    @abstractmethod
    def predict(self, x):
        pass
        
    @abstractmethod    
    def record_performance(self, stage: str, missing_values: str):
        pass
    
    def get_set(self, set: str):
        set = self.loader.get_set(set)
        return set["x"], set["y"]
        
    def compute_metrics(self, set: str, verbose = False):
        x, y   = self.get_set(set)
        y_prob = self.predict(x)
        if verbose:
            l, c = np.unique(y, return_counts = True)
            print(l, c)
        return compute_metrics(y, y_prob)
        
    def plot_2d(self, set: str):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        x, y        = self.get_set(set)
        transformer = TSNE(n_components = 2)
        x_2d        = transformer.fit_transform(x)
        colors      = ["red" if el else "blue" for el in y]
        plt.scatter(x_2d[:,0], x_2d[:,1], c = colors)
        plt.show()
        
    def init_run_dir(self, run_name: str):
        if not os.path.isdir(run_name):
            os.system(f"mkdir {run_name}")
            os.system(f"mkdir {run_name}/models")
            with open(f"{run_name}/performance.csv", "w+") as f:
                f.write("stage,missing_values,model,set")
                for metric in self.METRICS:
                    f.write(f",{metric}")
                f.write("\n")
