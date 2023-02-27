import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# figure(figsize=(8, 6), dpi=200)
# plt.rcParams.update({'font.size': 20})
#========================================
parser = argparse.ArgumentParser()
parser.add_argument('--csv1', type=str, default="./data/exp/ablation/csv/run-01-no-bc-tag-scalars_eval_sucess_eps_filter_rate.csv")
parser.add_argument('--csv2', type=str, default="./data/exp/ablation/csv/run-02-no_actor_grad-tag-scalars_eval_sucess_eps_filter_rate.csv")
parser.add_argument('--csv3', type=str, default="./data/exp/ablation/csv/run-03-no_DSA-tag-scalars_eval_sucess_eps_filter_rate.csv")
parser.add_argument('--csv4', type=str, default="./data/exp/ablation/csv/run-04-no_virtual_clutch-tag-scalars_eval_sucess_eps_filter_rate.csv")
parser.add_argument('--csv5', type=str, default="./data/exp/ablation/csv/run-05-ours-tag-scalars_eval_sucess_eps_filter_rate.csv")
parser.add_argument('--linewidth', type=int, default=4)
parser.add_argument('--smooth', type=int, default=0.7)
parser.add_argument('--maxstep', type=int, default=140000)
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
df4 = read_csv(args.csv4)
df5 = read_csv(args.csv5)

# print(df1)
def plot_line(_df, label, linewidth, color):
    plt.plot(_df["Step"], _df["Value"],linewidth=args.linewidth*0.8, color=lighten_color(color,amount=1.5), alpha=0.1)
    plt.plot(_df["Step"], _df["smooth"], label=label,linewidth=args.linewidth, color=lighten_color(color,amount=0.5), alpha=1)

plot_line(df1, "No BC", args.linewidth, color='tab:purple')
plot_line(df2, "No Actor Grad", args.linewidth, color='tab:orange')
plot_line(df3, "No DSA", args.linewidth, color='tab:green')
plot_line(df4, "No Virtual Clutch", args.linewidth, color='tab:red')
plot_line(df5, "Ours", args.linewidth, color='tab:blue')

plt.ticklabel_format(style='sci', axis='x',scilimits=(0,4))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=True)
plt.xlabel("Timestep")
plt.ylabel("Success Rate")


plt.savefig("./data/exp/ablation_success_rate.pdf",bbox_inches='tight')
if args.show:
    plt.show()
# print(df1)
# print(df2)


