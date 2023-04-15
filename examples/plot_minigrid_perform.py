import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# figure(figsize=(8, 6), dpi=200)
# plt.rcParams.update({'font.size': 20})
#========================================
parser = argparse.ArgumentParser()
parser.add_argument('--csv1', type=str, default="./data/exp2/minigrid/performance/run-2023-04-14-trs2_no_plan_8-tag-scalars_eval_return.csv")
parser.add_argument('--csv2', type=str, default="./data/exp2/minigrid/performance/run-2023-04-15-3090_no_actorgrad_10-tag-scalars_eval_return.csv")
parser.add_argument('--csv3', type=str, default="./data/exp2/minigrid/performance/run-2023-04-14-3090_8-tag-scalars_eval_return.csv")
parser.add_argument('--linewidth', type=int, default=4)
parser.add_argument('--smooth', type=int, default=0.0001)
parser.add_argument('--maxstep', type=int, default=220000)
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
def plot_line(_df, label, linewidth, color):
    plt.plot(_df["Step"], _df["Value"],linewidth=args.linewidth*0.8, color=lighten_color(color,amount=1.5), alpha=0.1)
    plt.plot(_df["Step"], _df["smooth"], label=label,linewidth=args.linewidth, color=lighten_color(color,amount=0.5), alpha=1)

plot_line(df1, "Dreamer", args.linewidth, color='tab:brown')
plot_line(df2, "CEM", args.linewidth, color='tab:red')
plot_line(df3, "Ours", args.linewidth, color='tab:blue')

plt.ticklabel_format(style='sci', axis='x',scilimits=(0,4))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, shadow=True)
plt.xlabel("Timestep")
plt.ylabel("Average Return")


plt.savefig("./data/exp2/minigrid/LFS_peformance.pdf",bbox_inches='tight')
if args.show:
    plt.show()
# print(df1)
# print(df2)


