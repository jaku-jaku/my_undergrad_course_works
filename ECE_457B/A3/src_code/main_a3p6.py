# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import defuzzify
import copy

import jx_lib

#%% P6
OUT_DIR_P6="output/P6"
jx_lib.create_all_folders(DIR=OUT_DIR_P6)

# %% ---- Define:
class FuzzySys():
    members = {}
    
    def __init__(self):
        self.init_fuzzy()

    def init_fuzzy(self):
        # Antecedent: Input
        # Consequent: Output
        ang_universe = np.arange(-30,30,0.5)
        vel_universe = np.arange(-60,60,1.0)
        cnt_universe = np.arange(-3.0, 3.0, 0.05)
        
        # New Antecedent/Consequent objects hold universe variables and membership
        ang = {}
        vel = {}
        cnt = {}

        # Assign membership functions
        # Membership function: angle
        ang['PL'] = fuzz.trapmf(ang_universe, [-10, 20, 30, 30])
        ang['NL'] = fuzz.trapmf(ang_universe, [-30, -30, -20, 10])

        # Membership function: velocity
        vel['PL'] = fuzz.trapmf(vel_universe, [-20,40,60,60])
        vel['NL'] = fuzz.trapmf(vel_universe, [-60,-60,-40,20])

        # Membership function: control action
        cnt['PL'] = fuzz.trapmf(cnt_universe, [0,  2, 3, 3])
        cnt['NL'] = fuzz.trapmf(cnt_universe, [-3, -3, -2, 0])
        cnt['NC'] = fuzz.trimf(cnt_universe, [-2, 0, 2])
        
        self.members["ang"] = ang
        self.members["vel"] = vel
        self.members["cnt"] = cnt
        self.members["ang_universe"] = ang_universe
        self.members["vel_universe"] = vel_universe
        self.members["cnt_universe"] = cnt_universe


    def custom_visualize(
        self, 
        ang = None, 
        vel = None, 
        tag = "",
        figsize = (15,15),
        OUT_DIR = OUT_DIR_P6,
        COLOR_TABLE = ["#b74f6fff","#628395ff","#dbad6aff","#dfd5a5ff","#cf995fff"],
    ):
        HEADER_COL = ["ang", "vel", "cnt"]
        HEADER_ROW = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4' ]
        MEM_TABLE = [
            [ 'PL', 'PL', 'NL' ],
            [ 'PL', 'NL', 'NC' ],
            [ 'NL', 'PL', 'NC' ],
            [ 'NL', 'NL', 'PL' ],
        ]
        m,n = np.shape(MEM_TABLE)
        fig = plt.figure(figsize=figsize)

        where_ = lambda X, val, tol: next(i for i, _ in enumerate(X) if np.isclose(_, val, tol))

        cnt_universe, cnt_vals, cx_hat_centroid = None, None, None
        for j,row in enumerate(MEM_TABLE):
            mu_a_ang = None
            mu_a_vel = None
            # pie : prediction percentage
            for i,mem_func in enumerate(row):
                member_type = HEADER_COL[i]
                topic = MEM_TABLE[j][i]
                S = self.members["{}_universe".format(member_type)]
                mu_A = self.members["{}".format(member_type)][topic]
                ax = plt.subplot(m, n, j * n + i + 1)
                ax.plot(
                    S, 
                    mu_A, 
                    label="{}[{}]".format(member_type, topic), 
                    color=COLOR_TABLE[i]
                )
                
                if member_type == "ang" and ang is not None:
                    mu_a_ang = mu_A[where_(X=S, val=ang, tol=0.01)]
                    mu_A_ang = np.minimum(mu_A, mu_a_ang)
                    ax.plot([ang, ang, S[-1]], [0, mu_a_ang, mu_a_ang], 
                        linestyle='dashed', label="$\mu_{}={:.3f}$".format(ang,mu_a_ang))
                    ax.fill_between(S, mu_A_ang, color=COLOR_TABLE[i], alpha=0.3)
                
                if member_type == "vel" and vel is not None:
                    mu_a_vel = mu_A[where_(X=S, val=vel, tol=0.01)]
                    mu_A_vel = np.minimum(mu_A, mu_a_vel)
                    ax.plot([vel, vel, S[-1]], [0, mu_a_vel, mu_a_vel], 
                        linestyle='dashed', label="$\mu_{}={:.3f}$".format(vel, mu_a_vel))
                    ax.fill_between(S, mu_A_vel, color=COLOR_TABLE[i], alpha=0.3)

                if member_type == "cnt" and vel is not None and ang is not None:
                    mu_a_cnt = min(mu_a_ang, mu_a_vel)
                    mu_A_cnt = np.minimum(mu_A, mu_a_cnt)
                    ax.axhline(y=mu_a_cnt, 
                        linestyle='dashed', label="$\mu={:.3f}$".format(mu_a_cnt))
                    ax.fill_between(S, mu_A_cnt, color=COLOR_TABLE[i], alpha=0.3)
                    if cnt_vals is None:
                        cnt_vals = mu_A_cnt
                    else:
                        cnt_vals = np.maximum(cnt_vals,mu_A_cnt)

                ax.legend()
                if j == 0:
                    ax.set_title(HEADER_COL[i])
                if i == 0:
                    ax.set_ylabel(HEADER_ROW[j])
                
                ax.label_outer()
        

        # save:
        fig.tight_layout()
        fig.savefig("{}/plot_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight', dpi=300)

        if cnt_vals is not None:
            # plot final aggregated control plot:
            fig = plt.figure(figsize=(10,8))
            ax = plt.subplot(1,1,1)
            # compute centroid:
            cnt_universe = self.members["cnt_universe"]
            cx_hat_centroid = (np.sum([u * c for u, c in zip(cnt_universe, cnt_vals)]) / np.sum(cnt_vals))
            cy_hat_centroid = np.sum(cnt_vals)/2/len(cnt_vals)
            # plot:
            ax.plot(cnt_universe, cnt_vals)
            ax.set_ylim(0, 1)
            ax.fill_between(cnt_universe, cnt_vals, color=COLOR_TABLE[2], alpha=0.3)
            ax.axvline(x=cx_hat_centroid, ymin=0, ymax=1, color="green", linestyle='dashed', label="Defuzzified value $\hat{c}$ (Centroid)")
            ax.scatter(cx_hat_centroid, cy_hat_centroid, color="red", label="Centroid")
            ax.legend()
            plt.title("Final Control Inference Plot")
            fig.savefig("{}/plot_final_control_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight', dpi=300)
        
        return cnt_universe, cnt_vals, cx_hat_centroid




# %% ---- Run:
fuzzsys = FuzzySys()

# %% ---- Vis (a):
fuzzsys.custom_visualize(tag="Rule-Based Inference")

# %% ---- Vis (b):
cnt_universe, cnt_vals, cx_hat_centroid = fuzzsys.custom_visualize(tag="Control Inference", ang=5, vel=15)
print("Final Control Value: ", cx_hat_centroid)

# %%
