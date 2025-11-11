import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="results/ga_log.csv")
    ap.add_argument("--out", type=str, default="figures/convergence.png")
    args = ap.parse_args()
    df = pd.read_csv(args.log)
    g = df.groupby("gen").agg(avg_best=("best","mean"), avg_mean=("mean","mean"),
                              lo=("best", lambda x: np.percentile(x, 5)),
                              hi=("best", lambda x: np.percentile(x, 95)))
    plt.figure()
    g["avg_best"].plot(label="Best (avg over seeds)")
    g["avg_mean"].plot(label="Mean (avg over seeds)")
    plt.fill_between(g.index, g["lo"], g["hi"], alpha=0.2, label="Best 90% range")
    plt.xlabel("Generation"); plt.ylabel("Score"); plt.legend(); plt.title("GA Convergence")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
