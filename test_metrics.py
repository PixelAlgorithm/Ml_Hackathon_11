import pandas as pd
from src.eval.metrics import top1_accuracy, macro_f1, candidate_log_loss

df = pd.DataFrame({
    "trip_id": [1]*6,
    "obs_idx": [0, 0, 0, 1, 1, 1],
    "arc_id":  [10, 11, 12, 11, 12, 13],
    "score":   [0.1, 0.8, 0.1, 0.2, 0.7, 0.1],
    "label":   [0, 1, 0, 1, 0, 0],
})
print(top1_accuracy(df))
print(macro_f1(df))
print(candidate_log_loss(df))