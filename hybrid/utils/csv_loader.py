import pandas as pd, os, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from abc import ABC, abstractmethod

SETS = ["train", "val", "test"]
ALL  = "all"


class CSVLoader(ABC):
    def __init__(self, csv_filename: str, keep_cols: list, target_col: str, 
    set_col: str = "set", normalize: bool = True, 
    empty_values_method: str = "amputate", join_train_val: bool = False, 
    join_train_test: bool = False, dirname: str = "", reshuffle: int = False, 
    sets_distr: dict = {"train": 4/6, "val": 1/6, "test": 1/6}, 
    sep = ",", verbose = True, **kwargs):
        self.verbose = verbose
        csv_filename = os.path.join(dirname, csv_filename)
        self.table   = pd.read_csv(csv_filename, sep = sep)
        keep_cols    = self.preprocess(keep_cols, **kwargs)
        self.table   = self.table.apply(lambda x: pd.to_numeric(x, errors = "coerce"))
        self.remove_outliers()
        self.set_sets(keep_cols, target_col, set_col, normalize, empty_values_method, 
        join_train_val, join_train_test, reshuffle, sets_distr)
        
    def set_sets(self, keep_cols: list, target_col: str, set_col: str, 
    normalize: bool, empty_values_method: str, join_train_val: bool, 
    join_train_test: bool, reshuffle: bool, sets_distr: dict):
        self.sets = {}
        self.split(set_col, join_train_val, join_train_test, reshuffle, target_col, sets_distr)
        self.filter(keep_cols, target_col)
        self.empty_values(empty_values_method)
        self.feature_selection()
        if normalize:
            self.normalize()
        if self.verbose:
            for s in self.available_sets():
                print(s, np.unique(self.sets[s]["y"], return_counts = True)[1], self.sets[s]["x"].shape)
            
    def empty_values(self, method):
        if method is None:
            return
        elif method == "amputate":
            self.amputate()
        elif method == "impute":
            self.impute()
        elif method == "impute_mean":
            self.simple_impute("mean")
        elif method == "impute_constant":
            self.simple_impute("constant")
        else:
            assert False
            
    def feature_selection(self):
        pass
        
    @abstractmethod
    def preprocess(self):
        pass
        
    @abstractmethod
    def remove_outliers(self):
        pass
        
    def reshuffle(self, set_col: str, target_col: str, sets: list):
        distr = {}
        labels = None
        for s in sets:
            labels, distr[s] = np.unique(self.table[self.table[set_col] == s][target_col].values, return_counts = True)
        self.table  = self.table.sample(frac = 1, random_state = 4).reset_index(drop = True)
        self.sets_from_distr(set_col, labels, distr, target_col)
        
    def sets_from_distr(self, set_col: str, labels: list, distr: dict, target_col: str):
        '''
        Creates a column with name 'set_col' with a list of strings with sets names 
        (train, val or test) with the distribution set by distr
            set_col    - string with the name of the column that sets which set each patient belongs to
            labels     - list with the different values the target variable can take (including np.nan)
            distr      - dictionary with the number of patients that should go to each set
            target_col - string with the name of the target variable
        '''
        set_col_vals = []
        sets         = {} # {label: list with the set names corresponding}
        for i in range(len(labels)):
            label = labels[i]
            # if np.isnan(label): continue
            sets[label] = []
            for set in distr:
                sets[label].extend( [set] * distr[set][i] )
        for _, row in self.table.iterrows():
            label = row[target_col]
            if np.isnan(label):
                set_col_vals.append(np.nan)
            else:
                if len(sets[label]) == 0:
                    # set_col_vals.append( np.random.choice(SETS) )
                    set_col_vals.append(np.nan)
                else:
                    set_col_vals.append(sets[label].pop(0))
        self.table[set_col] = set_col_vals
                
    def add_all_col(self, target_col: str, sets_distr: dict):
        if ALL in self.table.columns:
            return    
        labels, total_distr = np.unique(self.table[target_col].values, return_counts = True)
        distr = {s: [] for s in sets_distr}
        for set in sets_distr:
            for i in range(len(labels)):
                distr[set].append( int(sets_distr[set] * total_distr[i]) )
        for i in range(len(labels)):
            total     = sum([distr[set][i] for set in distr])
            remainder = total_distr[i] - total
            while remainder > 0:
                for s in distr:
                    distr[set][i] += 1
                    remainder -= 1
                    if remainder <= 0: break
        self.sets_from_distr(ALL, labels, distr, target_col)
        
    def split(self, set_col: str, join_train_val: bool, join_train_test: bool, 
    reshuffle: bool, target_col: str, sets_distr):
        assert self.table is not None
        sets = SETS + []
        if set_col == ALL:
            self.add_all_col(target_col, sets_distr)
        if join_train_val:
            self.table.loc[self.table[set_col] == "val", set_col] = "train"
            del sets[sets.index("val")]
        if join_train_test:
            self.table.loc[self.table[set_col] == "test", set_col] = "train"
            del sets[sets.index("test")]
        if join_train_val and join_train_test:
            self.table.loc[self.table[set_col] != "train", set_col] = "train"
        if reshuffle:
            self.reshuffle(set_col, target_col, sets)
        for s in sets:
            self.sets[s] = self.table[self.table[set_col] == s].copy()
        self.table = None
        
    def filter(self, keep_cols: list, target_col: str):
        assert self.table is None, "CSVLoader.filter: call split first"
        to_keep = []
        for col in keep_cols:
            if col in self.sets["train"].columns:
                to_keep.append(col)
            else:
                print(f"CSVLoader.filter: WARNING variable {col} is omitted")
        for s in self.available_sets():
            x = self.sets[s][to_keep]
            y = self.sets[s][target_col].values
            ids = self.sets[s]["idProcessoLocal"]
            self.sets[s] = {"x": x, "y": y, "patient_ids": ids}
            
    def get_set(self, set: str):
        assert set in self.available_sets(), f"CSVLoader.get_set: Unknown set {set}. Available sets are {self.available_sets()}"
        return self.sets[set]
        
    def normalize(self):
        scaler = StandardScaler()
        scaler.fit(self.sets["train"]["x"])
        for s in self.available_sets():
            self.sets[s]["x"] = pd.DataFrame(scaler.transform(self.sets[s]["x"]), columns = self.sets[s]["x"].columns)

    def impute(self):
        imp = IterativeImputer(max_iter = 40, random_state = 0)
        imp.fit(self.sets["train"]["x"])
        for s in self.available_sets():
            self.sets[s]["x"] = pd.DataFrame(imp.transform(self.sets[s]["x"]), columns = self.sets[s]["x"].columns)
            
    def simple_impute(self, strategy: str):
        if strategy == "constant":
            for s in self.available_sets():
                self.sets[s]["x"].fillna(-1, inplace = True)
        elif strategy == "mean":
            for col in self.sets["train"]["x"].columns:
                mean = self.sets["train"]["x"][col].mean()
                for s in self.available_sets():
                    self.sets[s]["x"][col].fillna(mean, inplace = True)
        
    def amputate(self):
        for s in self.available_sets():
            to_keep = []
            for _, row in self.sets[s]["x"].iterrows():
                if row.isnull().values.any():
                    to_keep.append(False)
                else:
                    to_keep.append(True)
            self.sets[s]["x"] = self.sets[s]["x"].iloc[to_keep]
            self.sets[s]["y"] = self.sets[s]["y"][to_keep]
            self.sets[s]["patient_ids"] = self.sets[s]["patient_ids"][to_keep]
                
    def available_sets(self):
        return [s for s in self.sets]
        
    def get_col(self, set: str, col: str):
        assert set in self.available_sets(), f"CSVLoader.get_col: Unknown set {set}. Available sets are {self.available_sets()}"
        return self.sets[set]["x"][col].values
    
    def set_col(self, set: str, col: str, values):
        assert set in self.available_sets(), f"CSVLoader.update_col: Unknown set {set}. Available sets are {self.available_sets()}"
        self.sets[set]["x"][col] = values
        
    def save_set_splits(self):
        patient_ids = []
        sets = []
        for s in self.available_sets():
            ids = self.sets[s]["patient_ids"]
            patient_ids.extend(ids)
            sets.extend([s] * len(ids))
        out = {"patient_ids": np.array(patient_ids).astype(int), "set": sets}
        pd.DataFrame(out).to_csv("set_splits.csv", index = False)
