import pandas as pd
from src.eval.metrics import (top1_accuracy, macro_f1,
                              candidate_log_loss, candidate_miss_rate)

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

# miss rate: 3 GPS points. Point 0 true road 11 (in candidates),
# point 1 true road 11 (in candidates), point 2 true road 99 (NOT in
# candidates, and has no candidates at all)
truth = pd.DataFrame({
    "trip_id": [1, 1, 1],
    "obs_idx": [0, 1, 2],
    "true_arc_id": [11, 11, 99],
})
print(candidate_miss_rate(df[["trip_id", "obs_idx", "arc_id"]], truth))