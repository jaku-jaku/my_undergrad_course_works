import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_files(DIR:str, file_end:str=".png"):
    return [ os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(file_end) ]

def create_all_folders(DIR:str):
    path_ = ""
    for folder_name_ in DIR.split("/"):
        path_ = os.path.join(path_, folder_name_)
        create_folder(path_, False)

def clean_folder(DIR:str):
    create_folder(DIR=DIR, clean=True)

def create_folder(DIR:str, clean:bool=False):
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    elif clean:
        filelist = get_files(DIR)
        for f in filelist:
            os.remove(f)

def make_confusion_matrix(
    cf,
    group_names=None,
    categories='auto',
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap='Blues',
    title=None
):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    https://github.com/DTrimarchi10/confusion_matrix
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    status = {}
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        status["Accuracy" ]  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            status["Precision"] = cf[1,1] / sum(cf[:,1])
            status["Recall"   ] = cf[1,1] / sum(cf[1,:])
            status["F1 Score" ] = 2 * status["Precision"] * status["Recall"] / (status["Precision"] + status["Recall"])
            
    stats_text = "\n\n" + " | ".join(["{:10s}={:.3f}".format(key, status[key]) for key in status])

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    ax.set_aspect(1)
    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    
    return fig, status


def make_comparison_matrix(
    dict_of_status_log,
    report_method="best",
    xlabel="",
    ylabel="",
    cbar=True,
    figsize=None,
    cmap='Blues',
    title=None
):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    '''

    y_header = list(dict_of_status_log)
    x_header = list(dict_of_status_log[y_header[0]])
    entry_list = list(dict_of_status_log[y_header[0]][x_header[0]])
    sqr = np.ceil(np.sqrt(len(entry_list)))

    fig = plt.figure(figsize=figsize)
    for i, entry in enumerate(entry_list):
        # fetch data
        data = np.zeros((len(y_header), len(x_header)))
        for j,x in enumerate(x_header):
            for k,y in enumerate(y_header):
                if report_method == 'best':
                    data[k, j] = np.max(dict_of_status_log[y][x][entry])
                elif report_method == 'worst':
                    data[k, j] = np.min(dict_of_status_log[y][x][entry])
                elif report_method == 'average':
                    data[k, j] = np.average(dict_of_status_log[y][x][entry])
                else:
                    raise ValueError("Only 'best/worst/average' is implemented!")
                    

        group_labels = ["{:.2f}".format(value) for value in data.flatten()]
        box_labels = np.asarray(group_labels).reshape(data.shape[0], data.shape[1])

        # MAKE THE HEATMAP VISUALIZATION
        ax = plt.subplot(sqr,sqr,i+1)
        sns.heatmap(data,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=x_header,yticklabels=y_header)
        ax.set_aspect(1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title("{} ({})".format(entry, report_method))
        
    return fig

def pie_plot(
        labels,
        sizes,
        title,
        figsize=(6,6),
        startangle=90,
        shadow=False
    ):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=shadow, startangle=startangle)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    return fig

def imgs_plot(
        dict_of_imgs,
        figsize=(6,6)
    ):
    fig = plt.figure(figsize=figsize)
    sqr = np.ceil(np.sqrt(len(dict_of_imgs)))

    for i,label in enumerate(dict_of_imgs):
        ax = plt.subplot(sqr,sqr,i+1)
        ax.imshow(dict_of_imgs[label])
        plt.xlabel(label)

    plt.tight_layout()
    return fig


def progress_plot(
    h,
    figsize=(6,6)
):
    # Plot
    fig = plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(h.history['accuracy'], label="training")
    plt.plot(h.history['val_accuracy'], label="validation")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.xticks(list(range(1, 1+len(h.history['accuracy']))))
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot((h.history['loss']), label="training")
    plt.plot((h.history['val_loss']), label="validation")
    plt.xticks(list(range(1, 1+len(h.history['loss']))))
    plt.ylabel("Loss (cross-entropy)")
    plt.xlabel("epoch")
    plt.legend()

    return fig

def output_prediction_result_plot(
    labels,
    dict_input_x,
    dict_prob,
    figsize = (12,6),
    OUT_DIR = "",
    tag     = ""
):
    for test_name in dict_prob:
        fig = plt.figure(figsize=figsize)
        # pie : prediction percentage
        ax = plt.subplot(1, 2, 1)
        predict_label = np.argmax(dict_prob[test_name])
        explode = np.zeros(len(labels))
        explode[predict_label] = 0.1
        ax.pie(dict_prob[test_name], labels=tuple(labels), autopct='%1.1f%%', explode=explode)
        plt.xlabel("Prediction Confidence (Probability)")
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # test img
        ax = plt.subplot(1, 2, 2)
        ax.imshow(dict_input_x[test_name])
        plt.xlabel("Input Test Image Data [{}]".format(test_name))
        fig.savefig("{}/test_sample_prediction_{}[{}].png".format(OUT_DIR, tag, test_name), bbox_inches = 'tight')
        plt.close(fig)

    return fig