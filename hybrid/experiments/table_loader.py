import sys, numpy as np
sys.path.append("..")
from stages import *
from utils.csv_loader import CSVLoader
from datetime import datetime


def str_to_datetime(date: str):
    if (date is None) or (date == "None") or (date == "0"):
        return None
    return datetime.fromisoformat(date[:19])
    
def get_age(birthdate, day):
    if (birthdate is None) or (day is None):
        return None
    return day.year - birthdate.year - ((day.month, day.day) < (birthdate.month, birthdate.day))
    
def get_hour_delta(time1, time2):
    if (time1 is None) or (time2 is None) or (time2 < time1):
        return None
    return (time2 - time1).seconds // (60*60)
    

class TableLoader(CSVLoader):
    def preprocess(self, keep_cols, filter_non_witnessed: bool = False, 
    filter_out_no_ncct: bool = True):
        if filter_out_no_ncct:
            self.filter_no_ncct()
        self.add_vars(filter_non_witnessed)
        if isinstance(keep_cols, str):
            keep_cols = self.stage_cols(keep_cols)
        self.convert_boolean_sequences(keep_cols)
        return keep_cols
            
    def stage_cols(self, stage):
        assert stage in STAGES, f"TableLoader.stage_cols: the available are {[s for s in STAGES]}"
        return STAGES[stage]
        
    def convert_boolean_sequences(self, keep_cols):
        boolean_cols = {"ouTerrIsq-7": 4, "ouTerrIsqL-7": 2, "lacAntL-7": 2, 
                        "enfAnt-7": 9, "enfAntL-7": 2, "compRtPA": 5, 
                        "TCCEter": 11, "TCCElac": 7, "RMNter": 11, "RMNlac": 7, 
                        "outProc": 4, "outCom": 8, "ecocarAnormal": 19}
        for col in boolean_cols:
            if f"{col}-1" not in keep_cols: continue
            splitted = [None if v == "None" else v.split(",") for v in self.table[col].values]
            new_cols = {f"{col}-{i+1}":[] for i in range(boolean_cols[col])}
            for s in splitted:
                if (s is None) or (len(s) != boolean_cols[col]):
                    for i in range(boolean_cols[col]):
                        new_cols[f"{col}-{i+1}"].append(np.nan)
                else:
                    for i in range(boolean_cols[col]):
                        new_cols[f"{col}-{i+1}"].append(int(s[i].replace(" ", "").lower() == "true"))
            del self.table[col]
            for col in new_cols:
                if np.isnan(new_cols[col]).any() and (len(np.unique(new_cols[col])) < 3):
                    continue
                self.table[col] = new_cols[col]
        self.table = self.table.copy()
        
    def filter_no_ncct(self):
        self.table = self.table[self.table["NCCT"] == "OK"]
        
    def add_vars(self, filter_non_witnessed: bool):
        '''
        add the variables 'age' and 'time_since_onset'
        'filter_non_witnessed' is a boolean that specifies whether to only consider
        patients whose stroke was witnessed in the 'time_since_onset' computation
        '''
        birthdate   = [str_to_datetime(date) for date in self.table["dataNascimento-1"].values]
        stroke_date = [str_to_datetime(date) for date in self.table["dataAVC-4"].values]
        ncct_time   = [str_to_datetime(date) for date in self.table["data-7"].values]
        age         = [get_age(birthdate[i],stroke_date[i]) for i in range(len(birthdate))]
        onset_time  = [get_hour_delta(stroke_date[i],ncct_time[i]) for i in range(len(birthdate))]
        if filter_non_witnessed:
            onset_time = [onset_time[i] if self.table["instAVCpre-4"].values[i]=="1" else None for i in range(len(birthdate))]
        self.table["age"]              = age
        self.table["time_since_onset"] = onset_time
                
    def remove_outliers(self):
        for col in VALID["intervals"]:
            if col in self.table.columns:
                min, max        = VALID["intervals"][col]
                self.table[col] = [v if np.isnan(v) or (min < v < max) else np.nan for v in self.table[col].values]
        for col in VALID["sets"]:
            if col in self.table.columns:
                set = VALID["sets"][col] + [np.nan]
                self.table[col] = [v if v in set else np.nan for v in self.table[col].values]
        

if __name__ == "__main__":
    loader  = TableLoader("table_data.csv",
                        keep_cols           = STAGE_BASELINE,
                        target_col          = "binary_rankin",
                        normalize           = True,
                        dirname             = "../../../data/gravo",
                        join_train_val      = True,
                        join_train_test     = False,
                        reshuffle           = True,
                        set_col             = "all",
                        filter_out_no_ncct  = False,
                        empty_values_method = "amputate")
    loader.sets["train"]["x"].to_csv("sapo.csv")
