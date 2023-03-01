import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

figure(figsize=(7, 6), dpi=200)
plt.rcParams.update({'font.size': 20})
#========================================
parser = argparse.ArgumentParser()
parser.add_argument('--csv1', type=str, default="./data/exp/performance/bars/eval_result_20230223-174501.csv")
# parser.add_argument('--csv2', type=str, default="./data/exp/performance/learning_curve/run-ours-tag-scalars_eval_sucess_eps_rate.csv")
# parser.add_argument('--linewidth', type=int, default=4)
parser.add_argument('--show', action="store_true")
args = parser.parse_args()

df1 = pd.read_csv(args.csv1)
# df2 = pd.read_csv(args.csv2)
results = {}

results["ours"] = df1["sucess_eps_filter_rate"]
plt.bar(0,0.1, width=0.8)
plt.bar(1,results["ours"], width=0.8)
plt.xticks(np.arange(2), ('Dreamer', 'ours'))
# plt.plot(df1["Step"], df1["Value"], label="Dreamer", linewidth=args.linewidth)
# plt.plot(df2["Step"], df2["Value"], label="Ours",linewidth=args.linewidth)
# plt.ticklabel_format(style='sci', axis='x',scilimits=(0,4))
# plt.legend(loc='upper left')
# plt.xlabel("Timestep")
plt.ylabel("Success Rate")


plt.savefig("./data/exp/performance/sim_performance_bar.png")
# plt.show()
