# %% Lib:
import numpy as np

# custom lib:
import jx_lib

#%% P1
OUT_DIR_P5="output/P5"
jx_lib.create_all_folders(DIR=OUT_DIR_P5)

# %%
X = [-4, 4]
step = 0.01
xs = np.linspace(X[0], X[1], int((X[1]-X[0])/step)+1) 
n = len(xs)

f_mu_A = lambda x, params: np.exp(- params["lambda"] * (np.abs(x - params["a"]) ** params["n"]))


# %%
TESTS = {
    "Varying lambda, Fix (a=0 n=1)": {
        "default": {"lambda":1, "a":0, "n":1},
        "test-subject-name": "$\lambda={}$",
        "test-subject": "lambda",
        "test-values": [-0.01, 0, 1, 2, 3, 5, 10],
    },
    "Varying a, Fix (lambda=1 n=1)": {
        "default": {"lambda":1, "a":0, "n":1},
        "test-subject-name": "$a={}$",
        "test-subject": "a",
        "test-values": [-5, -3, -1.5, 0, 1.5, 3, 5],
    },
    "Varying n, Fix (lambda=1 a=0)": {
        "default": {"lambda":1, "a":0, "n":1},
        "test-subject-name": "$n={}$",
        "test-subject": "n",
        "test-values": [-15, -5, -1, 0, 1, 5, 15],
    },
}
for TAG, test_set in TESTS.items():
    data_dict = {}
    for val in test_set["test-values"]:
        params = test_set["default"]
        params[test_set["test-subject"]] = val
        data_dict[test_set["test-subject-name"].format(val)] = \
            {"x": xs, "y": f_mu_A(x=xs, params=params), "linestyle":'-'}
    jx_lib.output_plot(
        data_dict= data_dict,
        Xlabel  = "$x$",
        Ylabel  = "$\mu_A(x)$",
        OUT_DIR = OUT_DIR_P5,
        tag     = "Membership Function ({})".format(TAG),
        COLOR_TABLE = ["#5A4CA8", "#6D97C9", "#8CC3A0", "#E9DA90", "#F29D72", "#D17484", "#333333"],
    )

# %%
