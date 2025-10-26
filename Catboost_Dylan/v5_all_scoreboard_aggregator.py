
"""
v5_all_scoreboard_aggregator.py
Collects all per-experiment scoreboard.csv files into one combined CSV.
Usage:
    python v5_all_scoreboard_aggregator.py
Outputs:
    ./outputs_v5/scoreboard_all.csv
"""
import os
import glob
import pandas as pd

def main():
    out = "./outputs_v5"
    rows = []
    for path in glob.glob(os.path.join(out, "*", "scoreboard.csv")):
        try:
            df = pd.read_csv(path)
            df["source"] = os.path.basename(os.path.dirname(path))
            rows.append(df)
        except Exception as e:
            print(f"Skip {path}: {e}")
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        all_df = all_df.sort_values(by="cv_rmsle").reset_index(drop=True)
        all_df.to_csv(os.path.join(out, "scoreboard_all.csv"), index=False)
        print(all_df)
    else:
        print("No scoreboard files found yet. Run the v5_* scripts first.")

if __name__ == "__main__":
    main()
