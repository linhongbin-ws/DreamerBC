import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

figure(figsize=(12, 6), dpi=200)
plt.rcParams.update({'font.size': 20})
#========================================
parser = argparse.ArgumentParser()
parser.add_argument('--csv1', type=str,
                    default="./data/exp/robustness/noise_0/eval_result_20230223-174501.csv")
parser.add_argument('--csv2', type=str, 
                    default="./data/exp/robustness/noise_0_1/eval_result_20230301-211737.csv")
parser.add_argument('--csv3', type=str,
                    default="./data/exp/robustness/noise_0_5/eval_result_20230301-213152.csv")
parser.add_argument('--width', type=int, default=0.8)

parser.add_argument('--show', action="store_true")
args = parser.parse_args()


def read_csv(_dir):
    _df = pd.read_csv(_dir)
    return _df["sucess_eps_filter_rate"].iloc[0]


# df2 = pd.read_csv(args.csv2)
results = {}

results["0.0"] = read_csv(args.csv1)
results["0.1"] = read_csv(args.csv2)
results["0.5"] = read_csv(args.csv3)
def plot_bar(idx,name, width, label):
    plt.bar(label, results[name], width=width, label=label)

name_list = ["0.0", "0.1", "0.5"]
label_list = [r'$\eta_{n}=0$', r'$\eta_{n}=0.1$', r'$\eta_{n}=0.5$',]
for i in range(len(name_list)):
    plot_bar(i, name_list[i], args.width, label_list[i])
    print(results[name_list[i]])
    plt.annotate(str(results[name_list[i]]), xy=(i,  results[name_list[i]]), ha='center', va='bottom')
plt.xticks(np.arange(len(results)))

plt.plot([0.5,0.5],[0,1.0],linestyle="--",color='k')
plt.ylabel("Success Rate")
ax=plt.gca()  #gca:get current axis得到当前轴
#设置图片的右边框和上边框为不显示
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
ax.set_ylim([0, 1])
plt.text(-0.15,0.93,'Train',{'fontsize':18}, color = "tab:blue", )
plt.text(1.3, 0.93, 'Transfer', {'fontsize': 18}, color="tab:blue",)
plt.savefig("./data/exp/robustness_performance.pdf",bbox_inches='tight')
if args.show:
    plt.show()
# plt.show()
