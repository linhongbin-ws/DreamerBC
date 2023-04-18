import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# figure(figsize=(8, 6), dpi=200)
# plt.rcParams.update({'font.size': 20})
#========================================
parser = argparse.ArgumentParser()
parser.add_argument('--csv1', type=str,
                    default="./data/exp3/performance/run-oracle-dreamer-2023-04-16-3090-tag-scalars_eval_return.csv")
parser.add_argument('--csv2', type=str,
                    default="./data/exp3/performance/run-oracle-bc-2023-04-17-3090-tag-scalars_eval_return.csv")
parser.add_argument('--csv3', type=str,
                    default="./data/exp3/performance/run-oracle-dreamerbc7-2023-04-18-trs-tag-scalars_eval_return.csv")
parser.add_argument('--linewidth', type=int, default=4)
parser.add_argument('--smooth', type=int, default=0.0001)
parser.add_argument('--maxstep', type=int, default=90000)
parser.add_argument('--show', action="store_true")
args = parser.parse_args()

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def read_csv(_dir):
    ts_factor = args.smooth
    _df = pd.read_csv(_dir)
    _df = _df[_df["Step"]<=args.maxstep]
    _df["smooth"] = _df["Value"].ewm(alpha=(1 - args.smooth)).mean()
    return _df
df1 = read_csv(args.csv1)
df2 = read_csv(args.csv2)
df3 = read_csv(args.csv3)


# print(df1)
def plot_line(_df, label, linewidth, color, converge_idx=None, converged_performance_length=6):
    plt.plot(_df["Step"], _df["Value"],linewidth=args.linewidth*0.8, color=lighten_color(color,amount=1.5), alpha=0.1)
    plt.plot(_df["Step"], _df["smooth"], label=label,linewidth=args.linewidth, color=lighten_color(color,amount=0.5), alpha=0.7)
    if converge_idx is not None:
        plt.plot([_df["Step"].iloc[converge_idx],_df["Step"].iloc[converge_idx]], [0,_df["Value"].iloc[converge_idx]], color=lighten_color(color,amount=0.8),linewidth=args.linewidth*0.7,linestyle='--',alpha=0.7,label=label)
        plt.scatter([_df["Step"].iloc[converge_idx]], [0], color=lighten_color(color,amount=0.8),marker="o",s=100, alpha=1,clip_on=False)
        print("performance: {} {:.3f}".format(label, _df["Value"].iloc[converge_idx:converge_idx+converged_performance_length].mean()))

fig, ax = plt.subplots( nrows=1, ncols=1 )
plot_line(df1, "Dreamer", args.linewidth, color='tab:brown', converge_idx=17)
plot_line(df2, "BC", args.linewidth, color='tab:red', converge_idx=6)
plot_line(df3, "Ours", args.linewidth, color='tab:blue', converge_idx=11)

idx = 15
def plot_dash(idx,df):
    plt.plot([df["Step"].iloc[idx],df["Step"].iloc[idx]], [0,df["smooth"].iloc[idx]], color='tab:gray',linewidth=args.linewidth*0.7,linestyle='--')

plt.ylim([0, 1])
plt.ticklabel_format(style='sci', axis='x',scilimits=(0,4))
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.7, 1.28),
          ncol=3, frameon=False, shadow=True,title="                                  Value:\n Timesteps of Convergence:")


title = legend.get_title()
title.align="right"
title.set_color("b")
title.set_x(-500)
title.set_y(-80)
plt.xlabel("Timestep")
plt.ylabel("Average Return")

plt.tight_layout()
if args.show:
    plt.show()
else:
    fig.savefig("./data/exp2/minigrid/performance_oracle2.png",dpi=fig.dpi)

# print(df1)
# print(df2)


