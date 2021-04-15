# %% Lib:
import numpy as np

# custom lib:
import jx_lib

#%% P1
OUT_DIR_P3="output/P3"
jx_lib.create_all_folders(DIR=OUT_DIR_P3)

# %%
V = np.linspace(0, 200, 21) # rev/s
v0 = 50 # rev/s
n = len(V)

F_LUT = {
    10: 0.1,
    20: 0.3,
    30: 0.6,
    40: 0.8,
    50: 1.0,
    60: 0.7,
    70: 0.5,
    80: 0.3,
    90: 0.1
}
F = [F_LUT[v] if v in F_LUT else 0 for v in V]
F_very = [F_LUT[v] if v in F_LUT else 0 for v in (V-v0)]
F_def = np.array(F) ** 2
F_pre = np.array(F) ** 0.5

# %%
jx_lib.output_plot(
    data_dict= {
        "Fast: $\mu_F(v)$" : {"x": V, "y": F},
        "Very Fast: $\mu_F(v-v_0)$" : {"x": V, "y": F_very, "linestyle":'-.'},
        "Definitely Fast: $\mu_F(v)^2$": {"x": V, "y": F_def, "linestyle":'--'},
        "Presumably Fast: $\mu_F(v)^{1/2}$": {"x": V, "y": F_pre, "linestyle":'--'},
    },
    Xlabel  = "$v$",
    Ylabel  = "$\mu$",
    OUT_DIR = OUT_DIR_P3,
    tag     = "Membership Functions",
)
