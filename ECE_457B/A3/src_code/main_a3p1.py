# %% Lib:
import numpy as np
import matplotlib.pyplot as plt

# custom lib:
import jx_lib

#%% P1
OUT_DIR_P1="output/P1"
jx_lib.create_all_folders(DIR=OUT_DIR_P1)


# %% ---- USER PARAMS:
S = [0, 6]
STEP = 0.01
PARAMS = {
    "lambda":   2,
    "n":        2,
    "a":        3,
}
def f_mu_A(x, params):
    return np.exp(- params["lambda"] * ((x - params["a"]) ** (params["n"])))

def A_alpha_cut(mu_A, alpha):
    return mu_A >= alpha

# Pre-Compute:
s = np.linspace(S[0], S[1], int((S[1]-S[0])/STEP + 1))
mu_A = f_mu_A(x=s, params=PARAMS)


# %%
def output_plot(
    data_dict,
    highlight_dict = None,
    Ylabel  = "",
    Xlabel  = "",
    figsize = (10,5),
    OUT_DIR = "",
    tag     = ""
):
    COLOR_TABLE = ["#001524","#15616d","#ff7d00","#ffecd1","#78290f"]
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    i = 0
    for name_, data_ in data_dict.items():
        linestyle="-"
        if "linestyle" in data_:
            linestyle = data_["linestyle"]
        plt.plot(data_["x"], data_["y"], 
            label=name_, color=COLOR_TABLE[i], linestyle=linestyle) 
        i += 1;
    # highlight content:
    if highlight_dict is not None:
        highlight_x = highlight_dict["highlight_x"]
        highlight_lb = highlight_dict["highlight_lb"]
        highlight_ub = highlight_dict["highlight_ub"]
        label = highlight_dict["label"]
        ax.fill_between(
            highlight_x, highlight_lb, highlight_ub, 
            label=label, facecolor=COLOR_TABLE[i])
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    plt.title("Plot [{}]".format(tag))
    fig.savefig("{}/plot_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight', dpi=300)
    plt.close(fig)
    return fig


# %% ---- Draw Membership Functions (a):
fx_1 = lambda mu_A_: np.array([mu_a_ if mu_a_ <= 0.5 else (1-mu_a_) for mu_a_ in mu_A_])

# Plot M1:
output_plot(
    data_dict= {
        "$\mu_A$" : {"x": s, "y": mu_A},
        "$1 - \mu_A$" : {"x": s, "y": (1 - mu_A), "linestyle":':'},
        "$f(x)_{M1}$": {"x": s, "y": fx_1(mu_A_=mu_A), "linestyle":'--'},
    },
    highlight_dict = {
        "highlight_x"   : s,
        "highlight_lb"  : 0,
        "highlight_ub"  : fx_1(mu_A_=mu_A),
        "label" : "$M_1$"
    },
    OUT_DIR = OUT_DIR_P1,
    tag = "M1",
)

# %% ---- Draw Membership Functions (b):
fx_2 = lambda mu_A_: np.abs(mu_A - A_alpha_cut(mu_A, alpha=0.5))

# Plot M2:
output_plot(
    data_dict= {
        "$\mu_A$" : {"x": s, "y": mu_A},
        "$\mu_{A_1/2}$" : {"x": s, "y": A_alpha_cut(mu_A, alpha=0.5), "linestyle":':'},
        "$f(x)_{M2}$": {"x": s, "y": fx_2(mu_A_=mu_A), "linestyle":'--'},
    },
    highlight_dict = {
        "highlight_x"   : s,
        "highlight_lb"  : mu_A,
        "highlight_ub"  : A_alpha_cut(mu_A, alpha=0.5),
        "label" : "$M_2$"
    },
    OUT_DIR = OUT_DIR_P1,
    tag = "M2",
)

# %% ---- Draw Membership Functions (b):
fx_3 = lambda mu_A_: np.abs(mu_A - (1 - mu_A))

# Plot M2:
output_plot(
    data_dict= {
        "$\mu_A$" : {"x": s, "y": mu_A},
        "$1 - \mu_A$" : {"x": s, "y": (1 - mu_A), "linestyle":':'},
        "$f(x)_{M3}$": {"x": s, "y": fx_3(mu_A_=mu_A), "linestyle":'--'},
    },
    highlight_dict = {
        "highlight_x"   : s,
        "highlight_lb"  : mu_A,
        "highlight_ub"  : (1 - mu_A),
        "label" : "$M_3$"
    },
    OUT_DIR = OUT_DIR_P1,
    tag = "M3",
)

# %%
