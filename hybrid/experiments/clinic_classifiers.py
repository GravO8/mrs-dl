import sys, numpy as np
sys.path.append("..")
from table_loader import TableLoader
from utils.table_classifier import TableClassifier

class ASTRALClassifier(TableClassifier):
    def __init__(self, dataset_filename: str = None, loader = None, **kwargs):
        if dataset_filename is not None:
            loader = TableLoader(csv_filename       = dataset_filename, 
                                keep_cols           = ["age", "totalNIHSS-5", "time_since_onset", "altVis-5", "altCons-5", "gliceAd-4"],
                                target_col          = "binary_rankin",
                                normalize           = False,
                                **kwargs)
        super().__init__(loader)
        self.format_cols()
        
    def format_cols(self):
        for s in self.loader.available_sets():
            self.loader.set_col(s, "age",                   (self.loader.get_col(s, "age") // 5).astype(int))
            self.loader.set_col(s, "totalNIHSS-5",           self.loader.get_col(s, "totalNIHSS-5").astype(int))
            self.loader.set_col(s, "time_since_onset",   (2*(self.loader.get_col(s, "time_since_onset") >= 3)).astype(int))
            self.loader.set_col(s, "altVis-5",           (2*(self.loader.get_col(s, "altVis-5") > 0)).astype(int))
            self.loader.set_col(s, "altCons-5",          (3*(self.loader.get_col(s, "altCons-5") > 0)).astype(int))
            glucose = self.loader.get_col(s, "gliceAd-4")
            self.loader.set_col(s, "gliceAd-4",           ((glucose > 131) | (glucose < 66)).astype(int))
        
    def predict(self, x):
        x = x.sum(axis = 1).values
        a = 0.178
        b = 30
        x = 1/(1 + np.e**(-a*(x-b)))
        return x
        
    def record_performance(self, missing_values: str, run_name: str = "runs"):
        self.init_run_dir(run_name)
        with open(f"{run_name}/performance.csv", "a") as f:
            for set in self.loader.available_sets():
                metrics = self.compute_metrics(set)
                f.write(f"NA,{missing_values},ASTRAL,{set}")
                for metric in self.METRICS:
                    f.write(f",{metrics[metric]}")
                f.write("\n")
            
        

if __name__ == "__main__":
    for missing in ("amputate", "impute", "impute_mean", "impute_constant"):
        astral = ASTRALClassifier("table_data.csv",
                                  dirname             = "../../../data/gravo",
                                  set_col             = "all",
                                  join_train_val      = True,
                                  join_train_test     = True,
                                  filter_out_no_ncct  = False,
                                  reshuffle           = True,
                                  empty_values_method = missing)
        astral.record_performance(missing)
