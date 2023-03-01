import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

figure(figsize=(8, 6), dpi=200)
plt.rcParams.update({'font.size': 20})
#========================================
parser = argparse.ArgumentParser()
parser.add_argument('--csv1', type=str, default="./data/exp/performance/bars/eval_result_20230223-174501.csv")
parser.add_argument('--csv2', type=str, default="./data/exp/variation_adaptation/needle_large/eval_result_20230228-022203.csv")
parser.add_argument('--csv3', type=str, default="./data/exp/variation_adaptation/needle_small/eval_result_20230227-232128.csv")
parser.add_argument('--csv4', type=str, default="./data/exp/variation_adaptation/needle_unregular/eval_result_20230227-201608.csv")
parser.add_argument('--csv5', type=str, default="./data/exp/variation_adaptation/needle_unregular2/eval_result_20230228-145136.csv")
parser.add_argument('--width', type=int, default=0.8)

parser.add_argument('--show', action="store_true")
args = parser.parse_args()


def read_csv(_dir):
    _df = pd.read_csv(_dir)
    return _df["sucess_eps_filter_rate"].iloc[0]


# df2 = pd.read_csv(args.csv2)
results = {}

results["standard"] = read_csv(args.csv1)
results["large"] = read_csv(args.csv2)
results["small"] = read_csv(args.csv3)
results["irreg1"] = read_csv(args.csv4)
results["irreg2"] = read_csv(args.csv5)
def plot_bar(idx,name, width):
    plt.bar(name,results[name], width=width, label=name)

name_list = ["standard", "small", "large", "irreg1", "irreg2"]

for i in range(len(name_list)):
    plot_bar(i, name_list[i], args.width)
    print(results[name_list[i]])
    plt.annotate(str(results[name_list[i]]), xy=(i,  results[name_list[i]]), ha='center', va='bottom')
plt.xticks(np.arange(len(results)))

plt.plot([0.5,0.5],[0,1.0],linestyle="--",color='k')
plt.ylabel("Success Rate")
ax=plt.gca()  #gca:get current axis得到当前轴
#设置图片的右边框和上边框为不显示
ax.set_ylim([0, 1])
plt.text(-0.3,1,'Train',{'fontsize':18}, color = "tab:blue", )
plt.text(2.2,1,'Transfer',{'fontsize':18}, color = "tab:blue",)
plt.savefig("./data/exp/varation_peformance.pdf",bbox_inches='tight')
if args.show:
    plt.show()
# plt.show()
