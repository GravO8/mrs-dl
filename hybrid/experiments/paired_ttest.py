import os, pandas as pd, scipy.stats as stats, numpy as np

'''
see
https://www.statology.org/paired-samples-t-test/
https://www.jmp.com/en_nl/statistics-knowledge-portal/t-test/paired-t-test.html
'''

ALPHA   = 0.01
# METRIC  = "f1-score"
METRIC  = "auc"
DIR     = "../../../runs/table data/runs-ttest/"
RUNS    = [ "runs-ttest-ASTRAL",
            "runs-ttest-LR_8vars",
            "runs-ttest-LR_2vars",
            "runs-ttest-LR_5vars",
            "runs-ttest-LR_5vars_SN"]

# CONSTANTS
FIRST_COL = "run\other run"

if __name__ == "__main__":
    pvalues         = {FIRST_COL: []}
    significance    = {FIRST_COL: []}
    for run in os.listdir(DIR):
        if ("leuko" in run) or (run not in RUNS): continue
        if os.path.isfile(os.path.join(DIR, run, "performance.csv")):
            pvalues[run[11:]]        = []
            significance[run[11:]]   = []
    for run in os.listdir(DIR):
        if ("leuko" in run) or (run not in RUNS): continue
        path     = os.path.join(DIR, run, "performance.csv")
        if os.path.isfile(path):
            run      = run[11:]
            data     = pd.read_csv(path)
            data     = data[data.set == "test"]
            data     = data[METRIC].values
            pvalues[FIRST_COL].append(run)
            significance[FIRST_COL].append(run)
            for other_run in os.listdir(DIR):
                if ("leuko" in other_run) or (other_run not in RUNS): continue
                other_path  = os.path.join(DIR, other_run, "performance.csv")
                if os.path.isfile(other_path):
                    other_run   = other_run[11:]
                    other_data  = pd.read_csv(other_path)
                    other_data  = other_data[other_data.set == "test"]
                    other_data  = other_data[METRIC].values
                    assert len(other_data) == 10, f"Run {run} has {len(data)} values"
                    pvalue      = stats.ttest_rel(data, other_data).pvalue
                    pvalues[other_run].append(pvalue)
                    if np.isnan(pvalue):
                        significance[other_run].append("")
                    else:
                        if pvalue < ALPHA:
                            # if we observe a large p-value (> .05) we cannot 
                            # reject the null hypothesis that the samples have 
                            # identical average scores
                            if data.mean() > other_data.mean():
                                significance[other_run].append(">")
                            else:
                                significance[other_run].append("<")
                        else:
                            significance[other_run].append("?")
    pd.DataFrame(pvalues).to_csv(f"ttest-pvalues-{METRIC}.csv", index = False)
    pd.DataFrame(significance).to_csv(f"ttest-significance-{METRIC}-{ALPHA}.csv", index = False)
