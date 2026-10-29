"""One command to build everything: python -m src.build   (ingest -> model -> anomalies)"""
import time
from . import ingest, model, anomaly


def main() -> None:
    t = time.time()
    df = ingest.build_analysis_table(); ingest.save(df)
    print(f"[1/3] ingestion done: {df.shape[0]:,} accounts x {df.shape[1]} columns ({time.time()-t:.0f}s)")
    r = model.run()
    print(f"[2/3] model done: LightGBM hold-out AUC={r['headline']['lightgbm']['auc']}, Gini={r['headline']['lightgbm']['gini']}, KS={r['headline']['lightgbm']['ks']} ({time.time()-t:.0f}s)")
    a = anomaly.run()
    print(f"[3/3] anomaly detection done: {int(a.anomaly_flag.sum())} accounts flagged ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()