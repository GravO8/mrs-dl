import skopt, pandas as pd, joblib, os
from .table_classifier import TableClassifier


class ClassicClassifier(TableClassifier):
    def __init__(self, model, loader, ranges, metric = "f1", weights: str = None, **kwargs):
        super().__init__(loader)
        self.model  = model
        self.metric = metric
        if weights is None:
            self.hyperparam_tune(ranges, **kwargs)
        else:
            self.model = joblib.load(weights)
        
    def predict(self, x):
        try:
            return self.model.predict_proba(x)[:,1]
        except:
            return self.model.predict(x)
        
    def fit(self, params = None):
        x_train, y_train = self.get_set("train")
        self.model.set_params(**(self.best_params if params is None else params))
        self.model.fit(x_train, y_train)
        
    def hyperparam_tune(self, ranges, init_points = 5, n_iter = 20, cv = 5, verbose = True):
        opt = skopt.BayesSearchCV(
            self.model,
            ranges,
            n_iter  = n_iter,
            cv      = cv,
            scoring = self.metric,
            verbose = 0,
            random_state = 0,
            return_train_score = True)
        x_train, y_train = self.get_set("train")
        opt.fit(x_train, y_train)
        self.cv_results  = pd.DataFrame(opt.cv_results_)
        self.best_params = opt.best_params_
        self.cv          = cv
        self.fit()
        if verbose:
            print("val score:", opt.best_score_)
            
    def grid_hyperparam_tune(self, ranges):
        from sklearn.model_selection import ParameterGrid
        from copy import deepcopy
        x_train, y_train = self.get_set("train")
        best_score = None
        best_model = None
        best_param = None
        for params in ParameterGrid(ranges):
            self.model.set_params(**params)
            self.model.fit(x_train, y_train)
            metrics = self.compute_metrics("val")
            if (best_score is None) or (metrics["f1-score"] > best_score):
                best_score = metrics["f1-score"]
                best_model = deepcopy(self.model)
                best_param = params
        self.model = best_model
        self.best_params = best_param
        
    def record_performance(self, stage: str, missing_values: str, run_name: str = "runs"):
        self.init_run_dir(run_name)
        model_name = self.model.__class__.__name__
        best_row   = self.cv_results[self.cv_results["rank_test_score"] == 1].iloc[0]
        assert best_row["params"] == self.best_params
        with open(f"{run_name}/performance.csv", "a") as f:
            for i in range(self.cv):
                train_score = best_row[f"split{i}_train_score"]
                test_score  = best_row[f"split{i}_test_score"]
                f.write(f"{stage},{missing_values},{model_name},train{i},{train_score},,,,\n")
                f.write(f"{stage},{missing_values},{model_name},test{i},{test_score},,,,\n")
            for set in self.loader.available_sets():
                metrics = self.compute_metrics(set)
                f.write(f"{stage},{missing_values},{model_name},{set}")
                for metric in ("f1-score", "accuracy", "precision","recall", "auc"):
                    f.write(f",{metrics[metric]}")
                f.write("\n")
        joblib.dump(self.model, f"{run_name}/models/{model_name}-{stage}-{missing_values}.joblib")
        with open(f"{run_name}/models/{model_name}-{stage}-{missing_values}-params.txt", "w") as f:
            f.write(str(self.best_params))
    
    def record_pretrained_performance(self, stage: str, missing_values: str, run_name: str = "runs"):
        model_name = self.model.__class__.__name__
        with open(f"{run_name}/performance.csv", "a") as f:
            for set in self.loader.available_sets():
                metrics = self.compute_metrics(set)
                f.write(f"{stage},{missing_values},{model_name},{set}")
                for metric in self.METRICS:
                    f.write(f",{metrics[metric]}")
                f.write("\n")


def knn(loader, **kwargs):
    from sklearn.neighbors import KNeighborsClassifier as KNN
    return ClassicClassifier(model = KNN(), 
                            loader = loader,
                            ranges = {
                                "n_neighbors": [3, 5, 7, 9, 11],
                                "weights": ["uniform", "distance"],
                                "metric": ["euclidean", "manhattan", "chebyshev", "jaccard"]
                            },
                            **kwargs)


def decision_tree(loader, **kwargs):
    from sklearn.tree import DecisionTreeClassifier as DT
    return ClassicClassifier(model = DT(),
                            loader = loader,
                            ranges = {
                                "criterion": ["entropy", "gini"],
                                "max_depth": skopt.space.space.Integer(2, 25),
                                "min_impurity_decrease": skopt.space.space.Real(.001, .025),
                                "random_state": [0]
                           },
                           **kwargs)


def random_forest(loader, **kwargs):
    from sklearn.ensemble import RandomForestClassifier as RF
    return ClassicClassifier(model = RF(),
                            loader = loader,
                            ranges = {
                                "criterion": ["entropy", "gini"],
                                "max_depth": skopt.space.space.Integer(2, 25),
                                "n_estimators": skopt.space.space.Integer(5, 300),
                                "max_features": skopt.space.space.Real(.1, 1),
                                "random_state": [0]
                           },
                           **kwargs)
    
    
def logistic_regression(loader, **kwargs):
    from sklearn.linear_model import LogisticRegression as LR
    return ClassicClassifier(model = LR(),
                            loader = loader,
                            ranges = {
                                "C": skopt.space.space.Real(.001, 3),
                                "tol": skopt.space.space.Real(1e-5, 1e-1),
                                "penalty": ["l1", "l2"],
                                "solver": ["saga"],
                                "max_iter": [100000],
                                "random_state": [0]
                           },
                           **kwargs)
    
    
def gradient_boosting(loader, **kwargs):
    from sklearn.ensemble import GradientBoostingClassifier as GB
    return ClassicClassifier(model = GB(),
                            loader = loader,
                            ranges = {
                                "n_estimators": skopt.space.space.Integer(5, 300),
                                "max_depth": skopt.space.space.Integer(5, 25),
                                "learning_rate": skopt.space.space.Real(.1, .9),
                                "random_state": [0],
                           },
                           **kwargs)
    
    
def svm(loader, **kwargs):
    from sklearn.svm import SVC as SVM
    return ClassicClassifier(model = SVM(),
                            loader = loader,
                            ranges = {
                                "C": skopt.space.space.Real(.001, 3),
                                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                                "degree": skopt.space.space.Integer(2, 5),
                                "gamma": ["scale", "auto"],
                                "random_state": [0],
                            },
                            **kwargs)
    
