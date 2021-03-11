# python
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #,KFold,cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from icecream import ic
from enum import IntEnum, auto

# jx-lib
import jx_lib

# data header
class DATA_HEADER(IntEnum):
    PREGNANCIES              = 0
    GLUCOSE                  = auto()
    BLOODPRESSURE            = auto()
    SKINTHICKNESS            = auto()
    INSULIN                  = auto()
    BMI                      = auto()
    DIABETESPEDIGREEFUNCTION = auto()
    AGE                      = auto()
    OUTCOM                   = auto()

# misc:
def print_latex_header(SVC_PARAMS, folder):
        LINE = "\n\
\\begin{{figure}}[H]\n\
\\centering\n\
\\subfloat[1:5-Fold]{{\\includegraphics[height=200px]{{../src_code/{folder}/Confusion_matrix_[m:unmodified-C:{c}-K:{k}-(1:5)]}}}} \, \n\
\\subfloat[2:5-Fold]{{\\includegraphics[height=200px]{{../src_code/{folder}/Confusion_matrix_[m:unmodified-C:{c}-K:{k}-(2:5)]}}}} \, \n\
\\subfloat[3:5-Fold]{{\\includegraphics[height=200px]{{../src_code/{folder}/Confusion_matrix_[m:unmodified-C:{c}-K:{k}-(3:5)]}}}} \, \n\
\\subfloat[4:5-Fold]{{\\includegraphics[height=200px]{{../src_code/{folder}/Confusion_matrix_[m:unmodified-C:{c}-K:{k}-(4:5)]}}}} \, \n\
\\subfloat[5:5-Fold]{{\\includegraphics[height=200px]{{../src_code/{folder}/Confusion_matrix_[m:unmodified-C:{c}-K:{k}-(5:5)]}}}} \, \n\
\\caption{{Confusion Matrices for C:{c} K:{k} 5-fold}}\n\
\\label{{table:confusion:{itr}}}\n\
\\end{{figure}}\n\
        "
        ii = 0
        for c in SVC_PARAMS['C']:
            for k in SVC_PARAMS['kernel']:
                ii += 1
                print(LINE.format(c=c, k=k, folder=folder, itr=ii))

##############################
#####         MAIN       #####
##############################
def main():
    # USER DEFINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    mode = "unmodified"
    ENABLE_TRAINING = True
    SVC_PARAMS = {
        'C' : [0.1,1,5,10],
        'kernel': ['linear','poly','rbf','sigmoid']
    }
    categories = ["No Diabetes", "Diabetes"]
    N_FOLD = 5
    PRINT_LATEX = (ENABLE_TRAINING == False)

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    MODES_AVALIABLE = ["unmodified", "balance"]
    if mode not in MODES_AVALIABLE:
        raise ValueError("Invalid mode selection!!")
    ### Directory generation ###
    OUT_DIR = "output/p1/{}".format(mode)
    jx_lib.create_all_folders(DIR=OUT_DIR)
    if ENABLE_TRAINING:
        # directory cleaning
        jx_lib.clean_folder(DIR=OUT_DIR)
    if PRINT_LATEX:
        print_latex_header(SVC_PARAMS=SVC_PARAMS, folder=OUT_DIR)
    ### IMPORT DATA ###
    data = np.loadtxt(open("diabetes.csv"), delimiter=",")

    ### PRE-PROCESSING DATA ###
    # Feature-wise Normalization:
    X = data[:, 0:DATA_HEADER.OUTCOM]
    Y = data[:, DATA_HEADER.OUTCOM]
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    X =  (X - x_min)/(x_max - x_min)
    Y = np.int8(Y)

    ic(np.shape(X))
    ic(np.max(X, axis=0))
    ic(np.min(X, axis=0))
    ic(np.shape(Y))
    ic(np.max(Y))
    ic(np.min(Y))
    ### Pre-data diagnosis ###
    print("=== Pre-Data Diagnosis ===")
    def diag_if_balance(data_, tag_):
        n_positive = sum(data_)
        n_total = len(data_)
        n_negative = n_total - n_positive
        ic(n_total)
        ic(n_positive)
        ic(n_negative)
        if (n_positive != n_total/2):
            print("[Issue Found] Data not balanced!")
        fig = jx_lib.pie_plot(
            labels=["+", "-"],
            sizes=[n_positive, n_negative],
            title="Label Distribution ({})".format(tag_)
        )
        img_path = "{}/Pie_{}_{}_data.png".format(OUT_DIR, mode, tag_)
        fig.savefig(img_path, bbox_inches = 'tight')
        plt.close(fig)
    
    diag_if_balance(data_=Y, tag_="Original")

    ### Pre-processing ###
    print("=== Pre-processing ===")
    if "balance" in mode:
        print(" > Pre-processing: balance data:")
        X_pos = X[Y==1]
        X_neg = X[Y==0]
        Y_pos = Y[Y==1]
        Y_neg = Y[Y==0]

        n_pos = len(Y_pos)
        n_neg = len(Y_neg)
        d_n = (n_pos - n_neg)

        X_major = X_pos
        n_repeat = np.ceil(d_n / (n_pos))
        if (n_neg > n_pos):
            X_major = X_neg
            d_n = (n_neg - n_pos)
            n_repeat = np.ceil(d_n / (n_neg))

        X_extra = np.repeat(X_major, d_n, axis=0)
        np.random.shuffle(X_extra)
        X = np.concatenate((X, X_extra[0:d_n]))
        if (n_neg > n_pos):
            Y = np.concatenate((Y, np.ones(d_n)))
        else:
            Y = np.concatenate((Y, np.zeros(d_n)))

        ic(np.shape(X))
        ic(np.shape(Y))
        diag_if_balance(data_=Y, tag_="Balanced")

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    ### TRAIN & VALIDATE ###
    if ENABLE_TRAINING:
        print("=== Processing ===")
        ### SVC ###
        # Processing Automation:
        dict_of_status_log = {}
        for c in SVC_PARAMS['C']:
            status_log_k = {}
            for k in SVC_PARAMS['kernel']:
                print("=== [m:{}-C:{}-K:{}] ===".format(mode,c,k))
                # K-Fold : 5x => choose the best kernel score
                kf = KFold(n_splits=N_FOLD, shuffle=True) # randomize
                indices = kf.split(X)
                # Perform training and validation
                for trial, indices_pair in enumerate(indices):
                    # partition training and test data:
                    train_index, test_index = indices_pair
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]

                    # declare SVC model:
                    svc_ = SVC(
                        C           = c, # Regularization term
                        kernel      = k,
                    )
            
                    # FITTING ...
                    svc_classifier_ = svc_.fit(X_train,y_train)

                    # Report:
                    ic(svc_classifier_)

                    ### Test Estimators ###
                    y_predict = svc_classifier_.predict(X_test) # round to evaluate

                    ### REPORT GEN. ###
                    conf_mat = confusion_matrix(y_test, y_predict)

                    # print:
                    ic(conf_mat)

                    # Gen Confusion Matrix Plot
                    labels = ["True Neg","False Pos","False Neg","True Pos"]
                    fig, status = jx_lib.make_confusion_matrix(
                        cf          = conf_mat, 
                        group_names = labels,
                        categories  = categories, 
                        figsize     = (6,6),
                        cmap        = "rocket"
                    )

                    # store template
                    ic(trial)
                    if trial == 0:
                        status_log_k[k] = status
                    # log all trials
                    for key, val in status.items():
                        if trial == 0:
                            status_log_k[k][key] = [val]
                        else:
                            status_log_k[k][key].append(val)

                    name = "Confusion_matrix_[m:{}-C:{}-K:{}-({}:{})]"\
                        .format(mode, c, k, trial+1, N_FOLD)
                    fig.savefig("{}/{}.png".format(OUT_DIR, name), bbox_inches = 'tight')
                    plt.close(fig)

            # buffer log
            dict_of_status_log[c] = status_log_k

        ### Gen Comparison Summary ###
        ic(dict_of_status_log)
        for method in ['best', 'worst', 'average']:
            img_path = "{}/Summary_{}_{}.png".format(OUT_DIR, mode, method)
            ic(img_path)
            fig = jx_lib.make_comparison_matrix(
                dict_of_status_log = dict_of_status_log,
                report_method = method,
                figsize = (12,12),
                xlabel  = "kernel",
                ylabel  = "C",
            )
            fig.savefig(img_path, bbox_inches = 'tight')
            plt.close(fig)
            
if __name__ == "__main__":
    main()

