import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

sns.set_style('darkgrid')
sns.set_palette(sns.hls_palette(8, l=.3, s=.8))
witness_rate_plot()

def witness_rate_plot():
    print("lol")
    cdf = pd.read_csv("simulations/complete")
    cdf['bauc'] = cdf['aauc']
    n_10 = cdf[cdf.n == 10]
    n_50 = cdf[cdf.n == 50]
    n_100 = cdf[cdf.n == 100]
    n_150 = cdf[cdf.n == 150]
    n_200 = cdf[cdf.n == 200]

    n_dfs = [('10', n_10), ('50', n_50), ('100', n_100), ('150', n_150), ('200', n_200)]
    plots = [('Instance AUC', 'iauc'), ('Bag AUC', 'bauc')]
    for name, n_df in n_dfs:
        for plot_name, plot in plots: 
            sns.heatmap(n_df.pivot('p2', 'p1', plot), vmin=.3, vmax=1.0)
            plt.title("N = " + name + ", " + plot_name)
            plt.savefig("./figs/aaai2018/" + plot + "/" + "n_" + name)
            plt.clf()

def bag_size_plot():
    cdf = pd.read_csv("simulations/complete")
    cdf['bauc'] = cdf['aauc']

    # p1 and p2 and delta of .05
    small_d = cdf[(cdf.p1 == .15) & (cdf.p2 == .1)]

    # p1 and p2 and delta of .05
    small10_d = cdf[(cdf.p1 == .2) & (cdf.p2 == .1)]
    
    # p1 and p2 and delta of .05
    small15_d = cdf[(cdf.p1 == .25) & (cdf.p2 == .1)]
    small20_d = cdf[(cdf.p1 == .3) & (cdf.p2 == .1)]
    small25_d = cdf[(cdf.p1 == .35) & (cdf.p2 == .1)]
 

    # p1 and p2 at a delta of .3
    medium_d = cdf[(cdf.p1 == .4) & (cdf.p2 == .1)]
    small35_d = cdf[(cdf.p1 == .45) & (cdf.p2 == .1)]


    # p1 and p2 at a delta of .5
    large_d = cdf[(cdf.p1 == .6) & (cdf.p2 == .1)]
    d_55 = cdf[(cdf.p1 == .65) & (cdf.p2 == .1)]


    df_opts = [('05', small_d), ('10', small10_d), ('15', small15_d), ('20', small20_d), ('25', small25_d), ('30', medium_d), ('35', small35_d), ('50', large_d), ('55', d_55)]
    for name, delta_df in df_opts:


        plt.plot(delta_df['n'], delta_df['iauc'], "o-", label="Individual AUC")
        plt.plot(delta_df['n'], delta_df['bauc'], "o-", label="Bag AUC")
        plt.xlabel("Bag Size")
        plt.ylabel("AUC")
        plt.ylim(.4, 1)
        plt.legend()
        plt.title("AUC vs. Bag Size, " + "Delta = ." + name)
        plt.savefig("./figs/aaai2018/bag_size_" + name )
        plt.clf()