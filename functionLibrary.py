import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
from mpl_axes_aligner import align
from pathlib import Path
import re
import math
import random
import os
import ftplib
import glob
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
import copy
from scipy.spatial.distance import directed_hausdorff
from operator import itemgetter
import pandas as pd
from distutils.dir_util import copy_tree
from shutil import copyfile
from datetime import datetime
from collections import Counter
import pickle
import webbrowser
import warnings
import time

from API_includes.MW_API import MW_API


def mw_connect():
    """
    Connect to microworkers.com by API

    Returns:
    ----------
        mw_api: class instance
    """
    api_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # personal microworkers api key
    api_url = "https://ttv.microworkers.com/api/v2"  # base api url
    mw_api = MW_API(api_key, api_url)  # call class MW_API
    return mw_api


def create_campaign(method, config):
    """
    Create microworkers campaign with predefined settings

    Parameters:
    ----------
        method: string
            "acqui", "questions"

        config: dict

    """
    # General settings
    # categoryId = "09"            # --> Data Collection/Mining/Extraction/AI Training
    # groupId = "18f477774339"     # --> Global Users
    # maxPositionPerWorker = 1
    # qtRequired = True
    # description = "Please read the instructions carefully before answering questions"
    # paymentPerTask = 0.10
    # height = "500"
    # width = "100%"
    # timeToWaitAfterSlotExpiration = 2

    if method == "acqui":
        available_positions = config["jobs"]["number_of_jobs"] * config["jobs"]["number_of_acquisitions"]
        url = config["campaign-acquisitions"]["url"]
        minutes_to_finish = config["campaign-acquisitions"]["minutesToFinish"]
        title = config["campaign-acquisitions"]["title"]
        # Settings for campaign
        # url = "https://geoinf-rs.bplaced.net/Crowdinterface_Acquisitions?worker={{MW_ID}}&campaign={{CAMP_ID}}&slot={{SLOT_ID}}&rand_key={{RAND_KEY}}"
        # minutesToFinish = 10
        # title = "Mark Cars in Aerial Image Strip"
        # return_params = "time"

    if method == "questions":
        _, number_files = calc_sub_it_numb(config)
        available_positions = number_files * config["interface_questions"]["it_numb"]
        url = config["campaign-questions"]["url"]
        minutes_to_finish = config["campaign-questions"]["minutesToFinish"]
        title = config["campaign-questions"]["title"]
        # url = "https://geoinf-rs.bplaced.net/Crowdinterface_Questions?worker={{MW_ID}}&campaign={{CAMP_ID}}&slot={{SLOT_ID}}&rand_key={{RAND_KEY}}"
        # minutesToFinish = 10
        # title = "Marked Car in an Aerial Photo Strip: Answer Questions"
        # return_params = "time"

    params = {
        "availablePositions": available_positions,
        "categoryId": config["campaign-general"]["categoryId"],
        "groupId": config["campaign-general"]["groupId"],
        "maxPositionPerWorker": config["campaign-general"]["maxPositionPerWorker"],
        "externalTemplate": {
            "url": url,
            "height": config["campaign-general"]["height"],
            "width": config["campaign-general"]["width"],
            # "parameters": [
            #     #return_params
            # ],
            # "timeToWaitAfterSlotExpiration": timeToWaitAfterSlotExpiration
        },
        "minutesToFinish": minutes_to_finish,
        "title": title,
        "qtRequired": config["campaign-general"]["qtRequired"],
        "description": config["campaign-general"]["description"],
        # "maximumJobLimit": {
        #     "enabled": config["campaign-general"]["maximumJobLimit_enabled"],
        #     "limitPerDay": config["campaign-general"]["maximumJobLimit_limitPerDay"]
        # },
        "paymentPerTask": config["campaign-general"]["paymentPerTask"],
        "ttr": config["campaign-general"]["ttr"],
    }

    if method == "acqui":
        params["removePositionOnNokRating"] = True

    # Establish API connection to microworkers.com
    mw_api = mw_connect()

    # Create campaign
    create_campaign = mw_api.do_request(method="post", action="/hire-group-campaigns", params=params)


def fetch_data(root_dir_server, root_dir_local, sub_dir, config, pre_rating=None):
    """
    Fetch data from bplaced server

    Parameters:
        root_dir_server: string
        root_dir_local: string
        sub_dir: string
        config: dict()

    Returns:

    """
    # Get data from server via ftp
    ftp = ftplib.FTP()
    ftp.connect(config["ftp"]["url"])
    ftp.login(user=config["ftp"]["user"], passwd=config["ftp"]["passwd"])
    print("\nConnected to bplaced: " + ftp.getwelcome() + "\n")

    # Create directory structure
    create_dir(root_dir_local, sub_dir)

    # Move existing files to backup folder
    move_existing_ = True
    if move_existing_:
        timestamp = move_existing(root_dir_local, sub_dir)

    # Download data via ftp
    for cur_subDir in sub_dir:
        path = root_dir_server + cur_subDir
        ftp.cwd(root_dir_server + cur_subDir)  # change dir to subdir
        # ftp.dir()   # list files

        for fname in ftp.nlst():
            try:
                if fname[0:5] == "Dummy":  # Skip files        #if fname == "Dummy":    # Skip folder
                    continue
                if fname == "deleted":  # Skip folder: consists of deleted files, <21min since last change & empty
                    continue
                if fname[0:4] == "1000":  # Skip files: created when all slots were taken
                    continue
                print("Downloading from folder=\"{}\" file=\"{}\"".format(cur_subDir, fname))
                local_filename = os.path.join(os.getcwd() + "/" + root_dir_local + "/" + cur_subDir[1:] + "/" + fname)
                local_file = open(local_filename, "wb")
                ftp.retrbinary("RETR " + fname, local_file.write)
                local_file.close()
            except:
                print("Error downloading files with ftp -> fname=", fname)

    if pre_rating:
        ftp.cwd(pre_rating)
        for fname in ftp.nlst():
            if fname[-4:] == ".txt":
                try:
                    print("Downloading from =\"{}\" file=\"{}\"".format(pre_rating, fname))
                    local_filename = os.path.join(os.getcwd() + "/Crowdinterface/Pre Rating/" + fname)
                    local_file = open(local_filename, "wb")
                    ftp.retrbinary("RETR " + fname, local_file.write)
                    local_file.close()
                except:
                    print("Error downloading files with ftp -> fname=", fname)
    ftp.quit()


def mw_match(mw_api, slotId, workerId):
    """
    Match local worker and slot Id with data on mw server

    Parameters:
    ----------
        mw_api:

        slotId:

        workerId:

    Returns:
    ----------
        skip: bool
            True if slot and workerId match

    """

    # Get slot info
    slot_info = mw_api.do_request("get", "/slots/" + slotId)

    # Check status
    if not slot_info["value"]["status"] == "NOTRATED":
        skip = True
        return skip
        # raise ValueError("Error: Slot \"{}\" is not open for rating".format(slot_info["value"]["id"]))

    # Check matching microworkers <-> rating result params (slotId, workerId)
    if not slot_info["value"]["workerId"] == workerId:
        skip = True
        return skip
        # raise ValueError("mw worker id \"{}\" and worker id \"{}\" do not match for slot \"{}\"".format(workerId, slot_info["value"]["workerId"], slot_info["value"]["id"]))

    if not slot_info["value"]["id"] == slotId:
        skip = True
        return skip
        # raise ValueError("mw slot id \"{}\" and slot id \"{}\" do not match for worker \"{}\"".format(slot_info["value"]["workerId"], slot_info["value"]["id"], workerId))

    skip = False
    return skip


def dbscan(meanX, meanY, cur_job, eps, minpts, config, plot, savePlot):
    """
    Perfom DBSCAN and return values

    Parameters
    ----------
        meanX, meanY: float, list
            meanX, meanY of acquisition
        cur_job: int
            Index of current Job
        eps: float, default=0.5
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other. See DBSCAN documentation.
        minpts: int, default=5
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. See DBSCAN documentation.
        plot: boolean, default=True
            If True, plot results
        savePlot: boolean, default=False
            If True, save plot

    Returns
    ----------
        noise_mask: boolean
            True: Sample belongs to cluster core
            False: Sample is an outlier
        labels: int
            Clusterlabel of corresponding sample
        centerX, centerY: float, array{ "x": [], "y": [] }
            Coordinates of mean cluster center
        cluster_core_idc: int, array{ [], [], [], ... }
            Array of core cluster indices, with lists of all acquisition indices belonging to this cluster
        cluster_all_idc: int, array{ [], [], [], ... }
            Array of all cluster indices
    """
    # Initiate return values
    noise_mask = []
    labels = []
    centerX = []
    centerY = []
    cluster_core_idc = []
    cluster_all_idc = []

    # Prepare for clustering using DBSCAN
    print("--------------------------")
    print("Running DBSCAN")
    print("eps = {}, minpts = {}".format(eps, minpts))

    db = DBSCAN(eps=eps, min_samples=minpts)
    db.fit(np.transpose([meanX, meanY]))

    labels = db.labels_  # == ptsC in Matlab
    core_idc = db.core_sample_indices_  # indices of clusters
    # components = db.components_  # == input (mean x, mean y)

    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Create Boolean Mask for noise
    noise_mask = np.zeros_like(db.labels_, dtype=bool)
    noise_mask[labels == -1] = True

    # Create Boolean Mask for cluster points
    cluster_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    cluster_samples_mask[labels != -1] = True

    # Create Boolean Mask for core samples #https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # Create array with only false, format like db.labels
    core_samples_mask[db.core_sample_indices_] = True

    # Add to main dict (outdated)
    # cars_in_shds[cur_job]["dbscan"]["cluster_idc_core"] = db.core_sample_indices_

    # Plot Noise, Core samples & Rest of samples
    if plot:
        cur_img = plt.imread(config["directories"]["Img_Folder"] + 'job' + str(cur_job + 1) + '/' + 'shd.png')
        f1 = plt.figure(1)
        ax = plt.subplot(111)

        height, width = cur_img.shape
        extent = [-1, width - 1, height - 1,
                  -1]  # Account for different coordinate origin Html Canvas(0,0) == upper left corner, Matlab(1,1)==Pixel Center Upper left corner, Python(0,0) Pixel Center Upper left corner

        ax.imshow(cur_img, cmap="gray", origin="upper", extent=extent, interpolation='none')
        # plt.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')

        if cur_job == 0:
            x_min = 1365;
            x_max = 1410
            y_min = 0;
            y_max = 45
        if cur_job == 1:
            x_min = 778;
            x_max = 778 + 163
            y_min = 95;
            y_max = 95 + 150
        if cur_job == 2:
            x_min = 1283;
            x_max = 1283 + 185
            y_min = 4;
            y_max = 4 + 150
        if cur_job == 3:
            x_min = 2332;
            x_max = 2332 + 209
            y_min = 54;
            y_max = 4 + 209
        if cur_job == 4:
            x_min = 1812;
            x_max = 1812 + 172
            y_min = 247;
            y_max = 247 + 141
        if cur_job == 5:
            x_min = 421;
            x_max = 421 + 217
            y_min = 252;
            y_max = 252 + 200

        f2, (ax1, ax2) = plt.subplots(2)
        # Plot image
        ax1.imshow(cur_img, cmap="gray", origin="upper", extent=extent, interpolation='none')
        ax2.imshow(cur_img, cmap="gray", origin="upper", extent=extent, interpolation='none')

        # if cur_job == 0:
        #    f2, (ax1, ax2) = plt.subplots(2)
        #    # Plot image
        #    ax1.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
        #    ax2.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)  # All labels of respective class
        class_pt_count = sum(class_member_mask)  # Count of point for respective class
        print("Points in Cluster {} = {}".format(k, class_pt_count))

        # Get x,y of core samples being part of the current class
        # x_core = np.array(meanX)[class_member_mask & core_samples_mask]
        # y_core = np.array(meanY)[class_member_mask & core_samples_mask]

        x_cluster = np.array(meanX)[class_member_mask & cluster_samples_mask]
        y_cluster = np.array(meanY)[class_member_mask & cluster_samples_mask]

        # Estimate mean center of cluster
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # x_mean = np.mean(x_core)
            # y_mean = np.mean(y_core)

            x_mean = np.mean(x_cluster)
            y_mean = np.mean(y_cluster)
            # print(len(np.array(cars_in_shds[cur_job]["mean"]["x"])[class_member_mask & core_samples_mask]))

        if k == -1:
            # print("k == -1 (noise)", unique_labels == k)
            col = [0, 0, 0, 1]  # Black used for noise.
            edge_col = 'r'
        else:
            edge_col = 'k'
            # Save return values
            centerX.append(x_mean)
            centerY.append(y_mean)

            # Add cluster idcs of points in current cluster/class
            cluster_core_idc.append([i for i, x in enumerate([class_member_mask & core_samples_mask][0].tolist()) if x])
            cluster_all_idc.append([i for i, x in enumerate(class_member_mask.tolist()) if x])

        if plot:
            plt.figure(1)  # Activate figure
            # Plot cluster, noise and mean center
            col_ = list(col)  # Add transparency
            col_[3] = 0.2
            col_ = tuple(col_)

            ax.plot(x_cluster, y_cluster, "+", markerfacecolor="None",  # Cluster samples
                    markeredgecolor='cyan', markeredgewidth=0.5, markersize=5, label="Cluster samples")  # 5
            # ax.plot(x_core, y_core, "+", markerfacecolor="None",   # Core samples
            #         markeredgecolor='cyan', markeredgewidth=0.5, markersize=15, label="Core samples")       # 5
            ax.plot(x_mean, y_mean, "1", markerfacecolor="None",  # col_,   # Mean cluster center
                    markeredgecolor='blue', markersize=10, label="Mean Cluster Center")  # 10

            # if cur_job == 0:
            plt.figure(2)
            ax1.plot(x_cluster, y_cluster, "+", markerfacecolor="None",  # Cluster samples
                     markeredgecolor='cyan', markeredgewidth=0.5, markersize=5, label="Cluster samples")  # 5
            ax2.plot(x_cluster, y_cluster, "+", markerfacecolor="None",  # Cluster samples
                     markeredgecolor='cyan', markeredgewidth=0.5, markersize=5, label="Cluster samples")  # 5
            # ax1.plot(x_mean, y_mean, "1", markerfacecolor="None",#col_,   # Mean cluster center
            #     markeredgecolor='blue', markersize=10, label="Mean Cluster Center")   #10
            # ax2.plot(x_mean, y_mean, "1", markerfacecolor="None",#col_,   # Mean cluster center
            #     markeredgecolor='blue', markersize=10, label="Mean Cluster Center")   #10

            x_noise = np.array(meanX)[class_member_mask & ~cluster_samples_mask]  # Class member but only noise
            y_noise = np.array(meanY)[class_member_mask & ~cluster_samples_mask]

            # x_noise = np.array(meanX)[class_member_mask & ~core_samples_mask]     # Class member but only noise + non core
            # y_noise = np.array(meanY)[class_member_mask & ~core_samples_mask]

            if edge_col == "r":
                plt.figure(1)
                ax.plot(x_noise, y_noise, "+", color="red",
                        markersize=5)  # FINAL 5#15 #markerfacecolor="red", markersize=5, label="Noise and non core samples")
                # markeredgecolor=edge_col, markeredgewidth=0.5, markersize=5, label="Noise and non core samples"   #5      # Non core -> face=class-color, Noise -> face=black

                # plt.plot(x_noise, y_noise, "o", markerfacecolor=col,
                #        markeredgecolor=edge_col, markersize=5, label="Noise and non core samples")          # Non core -> face=class-color, Noise -> face=black
                # print("xycluster {}, {}".format(x_cluster, y_cluster))
                # if cur_job == 0:
                plt.figure(2)
                ax1.plot(x_noise, y_noise, "+", color="red", markersize=5)  # !! FINAl 5
                ax2.plot(x_noise, y_noise, "+", color="red", markersize=5)
    if plot:
        # legend patches
        noise_patch = Line2D([0], [0], marker='+', linestyle="", color='red', label='Ausreißer', markersize=10)  # 5
        cluster_all_patch = Line2D([0], [0], marker='+', linestyle="", color='cyan', label='Kernpunkt',
                                   markersize=10)  # 15
        center_patch = Line2D([0], [0], marker='1', color='w', label='Clusterzentrum', markeredgecolor="blue",
                              markersize=10, markeredgewidth=0.5)  # 20

        cluster_patch = Line2D([0], [0], marker='+', linestyle="", color='cyan', label='Clusterpunkt',
                               markersize=10)  # 15)

        # if eps==10:
        #     ax.legend(handles=[ noise_patch, cluster_patch, center_patch ], fancybox=True, shadow=True, handlelength=1, loc='center left', bbox_to_anchor=(1, 0.5))
        #     ax.axis("off")
        #     plt.show()
        # if cur_job == 0:
        plt.figure(2)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_max, y_min)

        ax1.axis("off")
        ax2.axis("off")
        # ,center_patch
        ax1.legend(handles=[noise_patch, cluster_patch], fancybox=True, shadow=True, handlelength=1, loc='center left',
                   bbox_to_anchor=(1, 0.5))
        ax1.add_patch(Rectangle([x_min, y_max], x_max - x_min, y_min - y_max, fill=False, linestyle="-", linewidth=1,
                                color="orange"))
        ax2.add_patch(Rectangle([x_min, y_max], x_max - x_min, y_min - y_max, fill=False, linestyle="-", linewidth=1,
                                color="orange"))

        plt.title('DBSCAN (minpts={}, eps={}): Clusteranzahl={} '.format(minpts, eps, n_clusters_))

        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        if savePlot:
            path = 'figures/{}'.format(cur_job + 1)
            create_dir_if_not_exist(path)

            fname = '/job_{}_dbscan_cluster_eps_{}_minpts_{}_ZOOMED.png'.format(cur_job + 1, eps, minpts)
            path += fname
            f2.savefig(path, format='png', dpi=300, bbox_inches="tight")

        # Activate figure
        plt.figure(1)
        # Remove axis ticks
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # legend_elements          core_patch  cluster_patch,  , non_core_patch
        ax.legend(handles=[noise_patch, cluster_patch, center_patch], loc='upper center',
                  bbox_to_anchor=(0.5, 0))  # loc='center left', bbox_to_anchor=(1, 0.5)

        # plt.legend(handles=[cluster_patch, noise_patch, core_patch, non_core_patch], loc='upper right')
        plt.title('DBSCAN (minpts={}, eps={}): Clusteranzahl={} '.format(minpts, eps, n_clusters_))

        # if eps==10:
        #    plt.show()

        fig1 = plt.gcf()  # Needed because after plt.show() new fig is created and savefig would safe empty fig
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        # figManager = plt.get_current_fig_manager() # for fullscreen
        # figManager.window.state("zoomed")

        if savePlot:
            path = 'figures/{}'.format(cur_job + 1)
            create_dir_if_not_exist(path)

            fname = '/job_{}_dbscan_cluster_eps_{}_minpts_{}.png'.format(cur_job + 1, eps, minpts)
            path += fname
            fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")
            plt.close("all")

            # Non core cluster elements count as noise
    # for count, cluster in enumerate(cluster_all_idc):
    #    cluster_idc_diff = list(set(cluster) - set(cluster_core_idc[count]))
    #    if cluster_idc_diff:
    #        # Update return parameters
    #        noise_mask[cluster_idc_diff] = True
    #        labels[cluster_idc_diff] = -1
    #        n_noise_ += len(cluster_idc_diff)

    # Create dict with values to return
    db_result = dict()
    db_result = {
        "noise_mask": noise_mask,
        "labels": labels.tolist(),
        "centerX": centerX,
        "centerY": centerY,
        "cluster_core_idc": cluster_core_idc,
        "cluster_all_idc": cluster_all_idc,
        "n_clusters_": n_clusters_,
        "n_noise_": n_noise_
    }
    return db_result


def create_dir(root_dir_local, sub_dir):
    """
    Create directory structure if not already existing
    """
    for cur_subDir in sub_dir:
        path = os.path.join(os.getcwd() + "/" + root_dir_local + "/" + cur_subDir[1:] + "/")
        if not os.path.exists(path):
            os.makedirs(path)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):  # Source: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html ; last checked: 27.01.2021
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Input adminReturnData.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    # print("cov",cov)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])  # Correlation coefficient

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl adminReturnDataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    # ellipse.contains_points(np.transpose([x, y]))
    return ellipse, pearson


def create_dir_if_not_exist(sub_dir):
    """
    Create sub directory in current working directory if it doesn't already exist
    
    Parameters
    ----------
    sub_dir: string
    """
    try:
        path = os.path.join(os.getcwd() + "/" + sub_dir + "/")
        if not os.path.exists(path):
            os.makedirs(path)        
        
        # Path(foldername).mkdir(parents=True, exist_ok=True)
        # print("Folder '{}' structure is up to date".format(path))
    except:
        print("Error trying to update folder structure")


def diff_reference_acqui(cur_job, rating_path, input_path, config):
    """ 
    Calculate the similarity between the reference (drawn with the admininterface) 
    and the acquisitions (through crowdinterface) of uncertain cluster

    Parameters
    ----------
    rating_path: string
        admin_return: .txt
            Data acquired through the admininterface
    input_path: string   
        admin_input: .txt
            Data acquired through the crowdinterface, clustering & integrating

    Returns
    -------
    return_dict: dict()
        { 
            cluster_idx:
                [{ 
                    acquiIdx: 
                        acqui_coord: [ x y z w],
                        cluster_coord: [ x_mean y_mean ],
                        workerId: [],
                        finalRating: "OK" or "NOK",
                        reason: "angle" or "len" test failed
                }
        }
  
    """    
    return_dict = dict()
    
    cluster_idx = int()
    acqui_idc = []
    acqui_coord = []
    cluster_coord = []
    answer = str()
    reference = []

    admin_return = dict()
    admin_input = dict()
    subdict = dict()
    
    # Load rating file
    # path = "Admininterface/Post Rating/{}_ell.txt".format(cur_job)
    try:
        with open(rating_path, 'r') as f:
            for _ in range(2):  # Skip first 2 rows
                next(f)
            for linecount, line in enumerate(f):      # print("Read line {}".format(linecount))
                line = line.split(",")  # ['35', ' 0 1 2', ' Yes', ' 290.29 492.67 307.09 498.67\n']
                line[-1] = line[-1].strip() # remove \n
                
                cluster_idx = int(line[0].strip())    # .strip() -> remove whitespace at beginning and end
                acqui_idc = [int(x) for x in line[1].split()]
                answer = line[2].split()
                if answer[0] == "Yes":
                    reference = [float(x) for x in line[4].split()]       # reference in original image coordinate system      
                # print("Complete line", line)
                # print("cluster_idx {}".format(cluster_idx))
                # print("acqui_idc {}".format(acqui_idc))
                # print("answer {}".format(answer))
                # print("reference {}".format(reference))
                
                # Create dict with values to calculate with
                subdict = {
                    # "cluster_idx": cluster_idx,
                    "acqui_idc":    acqui_idc,
                    "answer":   answer,
                    "reference":    reference
                }
                admin_return[cluster_idx] = subdict
    except IOError:
        print("File not accessible:", rating_path)
            
    # print("admin_return", admin_return)
    
    # Load acquisitions
    # path = "Admininterface/{}/{}_uncertain_cluster.txt".format(cur_job, cur_job)
    try:
        with open(input_path, 'r') as f:
            for _ in range(2):  # Skip first 2 rows
                next(f)
            for linecount, line in enumerate(f):
                line = line.split(",")
                line[-1] = line[-1].strip() # remove \n
                cluster_idx = int(line[0].strip())
                acqui_coord = [float(x) for x in line[1].split()]
                cluster_coord = [float(x) for x in line[2].split()]
                workerId = line[3].split()
                
                subdict = {
                    "acqui_coord":    acqui_coord,
                    "cluster_coord":   cluster_coord,
                    "workerId":    workerId
                }
                
                if cluster_idx in admin_input:
                    admin_input[cluster_idx].append(subdict)
                else:
                    admin_input[cluster_idx] = [subdict]
        
        return_dict = admin_input.copy()
    except IOError:
        print("File not accessible:", input_path)
        
    # Check if adminReturnData matches
    for idx_return in admin_return:
        for acquiIdx in admin_return[idx_return]["acqui_idc"]:
            
            try:           
                data_input = admin_input[idx_return]  # check if entry for cluster idx exists
                # print("Cluster index match")
            except:
                print("ERROR cluster do not match -> Please rate all cluster of job {}".format(cur_job))
                
            try:    # check if acquisition count matches 
                len(data_input) == len(admin_return[idx_return]["acqui_idc"])
                # print("Acquisition count matches")
            except:
                print("ERROR acquisition count mismatch")               

    # Calculate difference between reference and corresponding acquisitions
    for cur_cluster in admin_return:
        # if cur_cluster != 18:  # for testing specific cluster
        #    continue
        # When cluster wrongfully detected -> Rating "NOK"
        if admin_return[cur_cluster]["answer"][0] == "No":
            
            # Plot
            fig2, ax = plt.subplots(1,1)
            cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
            height, width = cur_img.shape
            extent = [-1, width-1, height-1, -1]                 
            ax.imshow(cur_img, cmap="gray", origin="upper", extent=extent, interpolation='none')
            fig2.suptitle("Streifen {}, Cluster {}, Bewertung = NOK, Grund = Cluster markiert kein Fahrzeug".format(cur_job+1, cur_cluster))

            xmin = 50000
            ymin = 50000
            xmax = 0
            ymax = 0
            
            # ALT
            for idx, acqui in enumerate(admin_input[cur_cluster]):
                # print("Rating: NOK -> Cluster does not mark car")
                return_dict[cur_cluster][idx]["finalRating"] = "NOK"
                return_dict[cur_cluster][idx]["reason"] = "cluster does not mark car"
            # ENDE ALT
                x = acqui["acqui_coord"][0]
                y = acqui["acqui_coord"][1]
                x2 = acqui["acqui_coord"][2]
                y2 = acqui["acqui_coord"][3]

                if x < xmin:
                    xmin = x
                if x2 < xmin:
                    xmin = x2
                if x > xmax:
                    xmax = x
                if x2 > xmax:
                    xmax = x2
                if y < ymin:
                    ymin = y
                if y2 < ymin:
                    ymin = y2
                if y > ymax:
                    ymax = y
                if y2 > ymax:
                    ymax = y2                        
                ax.plot([x, x2], [y, y2], color="red", linewidth=2)
            
            ax.plot([], [], color="red", linewidth=2, label="Erfassung")
            ax.axis([xmin-20, xmax+20, ymax+20, ymin-20])
            ax.set_xlabel("x [px]")
            ax.set_ylabel("y [px]")
            ax.legend()
            # plt.show()
            fig = plt.gcf()
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            save_plot = True
            if save_plot:
                path = 'figures/{}/Verify_Rating/diff_reference_acqui/{}/'.format(cur_job+1, acqui["finalRating"])
                create_dir_if_not_exist(path)
                
                fname = "job_{}_cluster_{}_acqui_0-{}.png".format(cur_job+1, cur_cluster, idx)
                path = path + fname
                
                fig2.savefig(path, bbox_inches='tight')
                plt.close("all")
            continue    # skip to next cluster        
        
        # print("cur_cluster: {}".format(cur_cluster))
        # Reference Car axis angle
        # Calculate car axis angle
        LP = []
        RP = []
        startX = admin_return[cur_cluster]["reference"][0]
        startY = admin_return[cur_cluster]["reference"][1]
        endX = admin_return[cur_cluster]["reference"][2]
        endY = admin_return[cur_cluster]["reference"][3]
        if startX < endX:
            LP.extend([startX, startY])
            RP.extend([endX, endY])
        else:
            LP.extend([endX, endY])
            RP.extend([startX, startY])
        
        # lineRef = [ LP, RP ]
        # slopeRef = slope(lineRef[0][0], lineRef[0][1], lineRef[1][0], lineRef[1][1])
        x1 = LP[0]; y1 = LP[1]
        x2 = RP[0]; y2 = RP[1]
        
        ref_yaxis = False
        # If reference slope is +-inf -> angle between (reference/y-axis) and (acquisition)
        if x2-x1 == 0:
            angle_ref_yaxis = 90 * math.pi / 180
            ref_yaxis = True
        else:
            # If acqui slope is +-inf -> compare angles between (acqui/y-axis) and (reference)
            if y2-y1 == 0:    # If reference is parallel to x-axis -> 0°
                angle_ref_yaxis = 0 * math.pi / 180
            else: 
                angle_ref_yaxis = np.arctan((x2-x1) / np.abs(y2-y1))
            # Slope         
            m_ref = (y2-y1)/(x2-x1)
        
        angle_reference = np.abs((np.arctan2(RP[1]-LP[1], RP[0] - LP[0])))
        axis_len_reference = np.sqrt((startX - endX) ** 2 + (startY - endY) ** 2)
            
        mean_x_ref = (startX + endX) / 2
        mean_y_ref = (startY + endY) / 2
        
        # Test each acquisition
        for idx, acqui in enumerate(admin_input[cur_cluster]):
            # print("acqui: {}".format(acqui))
            # Acquisition Car axis angle            
            LP = []
            RP = []
            startX = acqui["acqui_coord"][0]
            startY = acqui["acqui_coord"][1]
            endX = acqui["acqui_coord"][2]
            endY = acqui["acqui_coord"][3]
            if startX < endX:
                LP.extend([startX, startY])
                RP.extend([endX, endY])
            else:
                LP.extend([endX, endY])
                RP.extend([startX, startY])
            angle_acquisition = np.abs((np.arctan2(RP[1]-LP[1], RP[0]-LP[0])))
            
            # lineAcqui = [LP, RP]
            
            x1 = LP[0]; y1 = LP[1]
            x2 = RP[0]; y2 = RP[1]
            
            # If reference slope +-inf -> angle (reference/y-axis) and (acquisition)
            if ref_yaxis:
                if x2-x1 == 0:
                    diffAngle = 0 * math.pi/ 180
                else:
                    diffAngle = np.arctan((x2-x1) / np.abs(y2-y1))
            else:
                # If acquisition slope +-inf -> angle (reference) and (acquisition/y-axis) 
                if x2-x1 == 0:
                    diffAngle = np.abs(angle_ref_yaxis)
                else:
                    # Slope
                    m_acqui = (y2-y1)/(x2-x1)
                    # If perpendicular
                    if m_acqui * m_ref == -1:
                        diffAngle = 90 * math.pi / 180
                    # If parallel
                    elif m_acqui == m_ref:
                        diffAngle = 0
                    else:
                        diffAngle = np.abs(np.arctan((m_ref - m_acqui) / (1+m_ref*m_acqui)))
            
            # slopeAcqui = slope(lineAcqui[0][0], lineAcqui[0][1], lineAcqui[1][0], lineAcqui[1][1])
            #
            # angleRefAcqui = np.abs(angle(slopeRef, slopeAcqui))
            # print("angle={}, angleOld={}".format(angleRefAcqui, np.abs((angle_reference - angle_acquisition)*180/math.pi)))
            
            # Difference in axis length
            max_len_deviation = config["integration"]["max_len_deviation"]    
            axis_len_acquisition = np.sqrt((startX - endX) ** 2 + (startY - endY) ** 2)
            diff_axis_len = np.abs(axis_len_acquisition - axis_len_reference)
            
            if diff_axis_len > max_len_deviation:
                # print("Rating: NOK -> failed axis len test")
                return_dict[cur_cluster][idx]["finalRating"] = "NOK"
                return_dict[cur_cluster][idx]["reason"] = "failed axis len test: diff_axis_len={:.2f}m".format(diff_axis_len *0.1)
                continue
            
            # Difference car axis angle
            threshold = 15 * math.pi / 180                     
              
            # if np.abs(angle_reference - angle_acquisition) > threshold:
            if diffAngle > threshold:
                # print("angle ref = {:.2f}, acq = {:.2f}".format(angle_reference, angle_acquisition))
                # print("Rating: NOK -> failed angle test")
                return_dict[cur_cluster][idx]["finalRating"] = "NOK"
                return_dict[cur_cluster][idx]["reason"] = "Angle={:.2f}°".format(180/math.pi * diffAngle)
                continue
            
            # Difference center
            max_center_deviation = config["integration"]["max_distance_correspondence"]
            mean_x_acquisition = (startX + endX) / 2
            mean_y_acquisition = (startY + endY) / 2
            
            # diff_meanX_center = np.abs(mean_x_acquisition - mean_x_ref)
            # diff_meanY_center = np.abs(mean_y_acquisition - mean_y_ref)
            
            euclDist = math.dist([mean_x_ref, mean_y_ref], [mean_x_acquisition, mean_y_acquisition])
            
            if euclDist > max_center_deviation:
                # print("Rating: NOK -> failed center difference test")
                return_dict[cur_cluster][idx]["finalRating"] = "NOK"
                return_dict[cur_cluster][idx]["reason"] = "failed center difference test: euclDist={:.2f}m".format(euclDist*0.1)
                continue
            
            # print(return_dict[cur_cluster][idx]["workerId"])
            # Remaining unrated acquisitions are set to be rated "OK"
            return_dict[cur_cluster][idx]["finalRating"] = "OK"
            return_dict[cur_cluster][idx]["reason"] = "Passed ref <-> acqui diff tests, angle={:.2f}°, diff_axis_len={:.2f}m, euclDist={:.2f}m".format(180/math.pi * diffAngle, diff_axis_len*0.1, euclDist*0.1)
            
        # Plot "NOK": reference, acquisition, reason
        save_plot = True
            
        for idx, acqui in enumerate(return_dict[cur_cluster]):
            # if acqui["finalRating"] == "NOK":
            x = acqui["acqui_coord"][0]
            y = acqui["acqui_coord"][1]
            x2 = acqui["acqui_coord"][2]
            y2 = acqui["acqui_coord"][3]
            
            x_ref = admin_return[cur_cluster]["reference"][0]
            y_ref = admin_return[cur_cluster]["reference"][1]
            x2_ref = admin_return[cur_cluster]["reference"][2]
            y2_ref = admin_return[cur_cluster]["reference"][3]
            
            fig2, ax = plt.subplots(1, 1)
            fig2.suptitle("Streifen {}, WorkerId {}, Cluster {}, Erfassung {}, Bewertung = {}, reason {}".format(cur_job+1, acqui["workerId"][0], cur_cluster, idx, acqui["finalRating"], acqui["reason"]))
            
            # Load batch image
            cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
            height, width = cur_img.shape
            extent = [-1, width-1, height-1, -1]                 
            ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
            
            xmin = 50000
            ymin = 50000
            xmax = 0
            ymax = 0
            
            if x < xmin:
                xmin = x
            if x2 < xmin:
                xmin = x2
            if x > xmax:
                xmax = x
            if x2 > xmax:
                xmax = x2
            if y < ymin:
                ymin = y
            if y2 < ymin:
                ymin = y2
            if y > ymax:
                ymax = y
            if y2 > ymax:
                ymax = y2            
            ax.plot([x, x2], [y, y2], color="red", linewidth= 2, label="Erfassung")
            ax.plot([x_ref, x2_ref], [y_ref, y2_ref], color="green", linewidth= 2, label="Referenz")            
            ax.axis([xmin-20, xmax+20, ymax+20, ymin-20])
            ax.set_xlabel("x [px]")
            ax.set_ylabel("y [px]")
            ax.legend()      
            # plt.show()
            fig = plt.gcf()
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            # figManager = plt.get_current_fig_manager() # for fullscreen
            # figManager.window.state("zoomed")
            
            if save_plot:
                path = 'figures/{}/Verify_Rating/diff_reference_acqui/{}/'.format(cur_job+1, acqui["finalRating"])
                create_dir_if_not_exist(path)
                
                fname = "job_{}_worker_{}_cluster_{}_acqui_{}.png".format(cur_job+1, acqui["workerId"][0], cur_cluster, idx)
                path = path + fname
                
                fig2.savefig(path, bbox_inches='tight')
                plt.close("all")  
    return return_dict


def set_dir(method):
    """ 
    Set directory for given method.
    
    Parameters:
    ----------
        method: string 
            "admin", "acquisitions" or "questions".
        
    Returns:
    ----------
        directory: dict
            rootDir and subDir on bplaced
    """
    if method == "admin":
        directory = {
            "rootDir_server": "/www/Admininterface/Post Rating",
            "rootDir_local": "Admininterface/Post Rating"
        }
    
    elif method == "acquisitions":
        directory = {
            "rootDir_server": "/www/Crowdinterface_Acquisitions",
            "rootDir_local": "Data/post/acquisitions",
            "subDir": [ "/results", "/time", "/fb" ]
        }
        
    elif method == "questions":
        directory = {
            "rootDir_server": "/www/Crowdinterface_Questions",
            "rootDir_local": "Data/post/questions",
            "subDir": [ "/results", "/time_clicks", "/fb" ],
            "pre_rating": "/www/Crowdinterface_Questions/Data"
        }
        
    # directory = dict()
    # directory = { "rootDir_server": rootDir_server,
    #              "rootDir_local": rootDir_local,
    #              "subDir": subDir
    #            }
    return directory


def calc_quality_params(cars_in_shds, time, config, method=None, ok_weak_cluster=None, ok_int_cluster=None, step=5, db_rating_result=None):
    """
    Calculate and plot quality parameters:
        - Precision, recall, f1-score
        - Position-, orientation-, and length error + hausdorff distance
    
    Parameters
    ----------
        cars_in_shds: dict
            all integration data
        time: string
            "pre" or "post" -> in regard to verification
        config
        method: string
            "admin" or "crowd"
        ok_weak_cluster: 
            cluster with 100% "OK" rating post verification
        ok_int_cluster:
            cluster with 100% "OK" rating pre verification
        step: int
            number of answers taken into account when calculating final rating 
        db_rating_result: dict
            rating result for uncertain cluster detected with 2nd DBSCAN iteration
    
    Returns
    ----------
        quality_params: dict
    
    """
    position_error = {};    len_error = {};    orientation_error = {}
    precision = [];    recall = [];     f1_score = [] 
    print("\n-----------------------------------------------")
    if time == "pre":
        print("Quality Parameters pre verification".format(method))
    if time == "post":
        print("Quality Parameters post verification with {}interface".format(method))
    print("-----------------------------------------------")

    quality_params = {}    
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        
        quality_params[cur_job] = {
            "x": [], "y": [], "x2": [], "y2": [],   # x,y,x2,y2 of integrated line
            "xMean": [], "yMean": [],   # Center of integrated cluster
            "TP": [], "FP": [], "FN": [],     # Quality params
            "precision": [], "recall": [], "f1-score": [],  # Quality params
            "err": {
                "pos": [],
                "len": [],
                "ori": [],
                "hausdorff": []
            }
        }
        
        # Reorganize data
        # Ellipse data      
        if time == "post":
            cluster_idc = ok_int_cluster[cur_job]

        if time == "pre":
            cluster_idc = range(len(cars_in_shds[cur_job]["kmeans"]["cluster_idc"]))  

        for cur_cluster_idx in cluster_idc:
            cur_cluster_idx = int(cur_cluster_idx)
            
            quality_params[cur_job]["x"].append(cars_in_shds[cur_job]["kmeans"]["int_line_start"]["x"][cur_cluster_idx])
            quality_params[cur_job]["y"].append(cars_in_shds[cur_job]["kmeans"]["int_line_start"]["y"][cur_cluster_idx])
            quality_params[cur_job]["x2"].append(cars_in_shds[cur_job]["kmeans"]["int_line_end"]["x"][cur_cluster_idx])
            quality_params[cur_job]["y2"].append(cars_in_shds[cur_job]["kmeans"]["int_line_end"]["y"][cur_cluster_idx])
            quality_params[cur_job]["xMean"].append(cars_in_shds[cur_job]["kmeans"]["mean"]["x"][cur_cluster_idx])
            quality_params[cur_job]["yMean"].append(cars_in_shds[cur_job]["kmeans"]["mean"]["y"][cur_cluster_idx])

        if time == "post":
            # Db weak data
            for cur_cluster_idx in ok_weak_cluster[cur_job]:
                cur_cluster_idx = int(cur_cluster_idx)
                sx = []; sy = []; ex = []; ey = []
                for acqui in db_rating_result[cur_job][cur_cluster_idx]:
                    if method == "crowd":
                        if acqui["finalRating"]["steps"][step] == "OK":
                            if acqui["acquiCoord"]["x"] < acqui["acquiCoord"]["z"]:  # == startX<endX
                                sx.append(acqui["acquiCoord"]["x"])
                                sy.append(acqui["acquiCoord"]["y"])
                                ex.append(acqui["acquiCoord"]["z"])
                                ey.append(acqui["acquiCoord"]["w"])
                            else:           
                                sx.append(acqui["acquiCoord"]["z"])
                                sy.append(acqui["acquiCoord"]["w"])
                                ex.append(acqui["acquiCoord"]["x"])
                                ey.append(acqui["acquiCoord"]["y"])
                            continue
                    if method == "admin":
                        if acqui["finalRating"] == "OK":
                            if acqui["acquiCoord"][0] < acqui["acquiCoord"][2]:  # == startX<endX
                                sx.append(acqui["acquiCoord"][0])
                                sy.append(acqui["acquiCoord"][1])
                                ex.append(acqui["acquiCoord"][2])
                                ey.append(acqui["acquiCoord"][3])
                            else:           
                                sx.append(acqui["acquiCoord"][2])
                                sy.append(acqui["acquiCoord"][3])
                                ex.append(acqui["acquiCoord"][0])
                                ey.append(acqui["acquiCoord"][1])
                            continue
                    
                # Find mean start/end point of integrated line 
                sxex = sx[:]
                sxex.extend(ex)
                syey = sy[:]
                syey.extend(ey)                
                
                # KMeans with 2 Clusters -> 1 Cluster car start, 1 Cluster car end
                km = KMeans(n_clusters=2, max_iter=300, random_state=42)
                km.fit(np.transpose([sxex, syey]))
                
                labels = km.labels_
                cluster_center = km.cluster_centers_
                n_iter_conv = km.n_iter_
                
                mean_cluster_center = np.mean(cluster_center, axis=0)
                
                x1 = cluster_center[0, 0]
                x2 = cluster_center[1, 0]
                y1 = cluster_center[0, 1]
                y2 = cluster_center[1, 1]
                
                quality_params[cur_job]["x"].append(x1)
                quality_params[cur_job]["y"].append(y1)
                quality_params[cur_job]["x2"].append(x2)
                quality_params[cur_job]["y2"].append(y2)
                quality_params[cur_job]["xMean"].append(mean_cluster_center[0])
                quality_params[cur_job]["yMean"].append(mean_cluster_center[1])

        # FP, TP        
        for cur_cluster_idx in range(len(quality_params[cur_job]["x"])):
            x = quality_params[cur_job]["x"][cur_cluster_idx]
            y = quality_params[cur_job]["y"][cur_cluster_idx]
            x2 = quality_params[cur_job]["x2"][cur_cluster_idx]
            y2 = quality_params[cur_job]["y2"][cur_cluster_idx]         
            
            mean_x = quality_params[cur_job]["xMean"][cur_cluster_idx]
            mean_y = quality_params[cur_job]["yMean"][cur_cluster_idx]   
             
            axis_length = np.sqrt((x - x2)**2 + (y - y2)**2)
            axis_angle = np.arctan2((y2 - y), (x2 - x))
            
            gt_mean_x = np.array(cars_in_shds[cur_job]["gt"]["mean"]["x"])
            gt_mean_y = np.array(cars_in_shds[cur_job]["gt"]["mean"]["y"])
            
            # Distance: GT <-> integrated acquisition
            diff_2_gt = np.sqrt((gt_mean_x - mean_x)**2 + (gt_mean_y - mean_y)**2)
            
            min_idx = np.argmin(diff_2_gt)
            min_dist = diff_2_gt[min_idx]
            
            if min_dist > config["integration"]["max_distance_correspondence"]:
                quality_params[cur_job]["FP"].append(True) 
                quality_params[cur_job]["TP"].append(False)
            else:
                quality_params[cur_job]["FP"].append(False)
                quality_params[cur_job]["TP"].append(True)
                quality_params[cur_job]["err"]["pos"].append(min_dist)
                quality_params[cur_job]["err"]["len"].append(np.abs(cars_in_shds[cur_job]["gt"]["car_axis_len"][min_idx] - axis_length))
                
                ###### Orientation error, using slopes 
                # Reference angle
                LP = []
                RP = []
                startX = cars_in_shds[cur_job]["gt"]["start"]["x"][min_idx]
                startY = cars_in_shds[cur_job]["gt"]["start"]["y"][min_idx]
                endX = cars_in_shds[cur_job]["gt"]["end"]["x"][min_idx]  
                endY = cars_in_shds[cur_job]["gt"]["end"]["y"][min_idx]
                
                if startX < endX:
                    LP.extend([startX, startY])
                    RP.extend([endX, endY])
                else:
                    LP.extend([endX, endY])
                    RP.extend([startX, startY])
                sx = LP[0]; sy = LP[1]
                ex = RP[0]; ey = RP[1]
                
                ref_yaxis = False
                # If reference slope is +-inf -> angle between (reference/y-axis) and (acquisition)
                if ex-sx == 0:
                    angle_ref_yaxis = 90 * math.pi / 180
                    ref_yaxis = True
                else:
                    # If acqui slope is +-inf -> compare angles between (acqui/y-axis) and (reference)
                    if ey-sy == 0:    # If reference is parallel to x-axis -> 0°
                        angle_ref_yaxis = 0 * math.pi / 180
                    else: 
                        angle_ref_yaxis = np.arctan((ex-sx) / np.abs(ey-sy))
                    # Slope         
                    m_ref = (ey-sy)/(ex-sx)
                    
                # Acquisition angle               
                LP = []; RP = [] 
                startX = x
                startY = y
                endX = x2        
                endY = y2
                if startX < endX:
                    LP.extend([startX, startY])
                    RP.extend([endX, endY])
                else:
                    LP.extend([endX, endY])
                    RP.extend([startX, startY])
                
                sx = LP[0]; sy = LP[1]
                ex = RP[0]; ey = RP[1]
                
                # If reference slope +-inf -> angle (reference/y-axis) and (acquisition)
                if ref_yaxis:
                    if ex-sx == 0:
                        diff_angle = 0 * math.pi / 180
                    else:
                        diff_angle = np.arctan((ex-sx) / np.abs(ey-sy))
                else:
                    # If acquisition slope +-inf -> angle (reference) and (acquisition/y-axis) 
                    if ex-sx == 0:
                        diff_angle = np.abs(angle_ref_yaxis)
                    else:
                        # Slope
                        m_acqui = (ey-sy)/(ex-sx)
                        # If perpendicular
                        if m_acqui * m_ref == -1:
                            diff_angle = 90 * math.pi / 180
                        # If parallel
                        elif m_acqui == m_ref:
                            diff_angle = 0
                        else:
                            diff_angle = np.abs(np.arctan((m_ref - m_acqui) / (1+m_ref*m_acqui)))
                quality_params[cur_job]["err"]["ori"].append(diff_angle * 180 / math.pi)
                
                # Hausdorff
                cur_car = np.array([(x, y),
                                    (x2, y2)])
                ref_car = np.array([(cars_in_shds[cur_job]["gt"]["start"]["x"][min_idx], cars_in_shds[cur_job]["gt"]["start"]["y"][min_idx]),
                                    (cars_in_shds[cur_job]["gt"]["end"]["x"][min_idx], cars_in_shds[cur_job]["gt"]["end"]["y"][min_idx])])
                
                hausdorff = max(directed_hausdorff(cur_car, ref_car)[0], directed_hausdorff(ref_car, cur_car)[0])
                quality_params[cur_job]["err"]["hausdorff"].append(hausdorff)

        # FN
        for gt_idx in range(len(cars_in_shds[cur_job]["gt"]["mean"]["x"])):
            gt_mean_x = cars_in_shds[cur_job]["gt"]["mean"]["x"][gt_idx]
            gt_mean_y = cars_in_shds[cur_job]["gt"]["mean"]["y"][gt_idx]
            
            center_x = np.array(quality_params[cur_job]["xMean"])
            center_y = np.array(quality_params[cur_job]["yMean"])
            
            # Distance: GT <-> integrated acquisition
            diff_2_gt = np.sqrt((center_x - gt_mean_x)**2 + (center_y - gt_mean_y)**2)

            min_idx = np.argmin(diff_2_gt)
            min_dist = diff_2_gt[min_idx]

            if min_dist > config["integration"]["max_distance_correspondence"]:
                quality_params[cur_job]["FN"].append(True)
            else:
                quality_params[cur_job]["FN"].append(False)

        # Output:
        pos_dst = np.array(quality_params[cur_job]["err"]["pos"]) * config["integration"]["cellSize"]
        len_dst = np.array(quality_params[cur_job]["err"]["len"]) * config["integration"]["cellSize"]
        ori_dst = np.array(quality_params[cur_job]["err"]["ori"])  # * 180 / math.pi
        hausdorff_dst = np.array(quality_params[cur_job]["err"]["hausdorff"]) * config["integration"]["cellSize"]
        
        mean_pos_dst = np.mean(pos_dst)
        mean_len_dst = np.mean(len_dst)
        mean_ori_dst = np.mean(ori_dst)
        mean_hausdorff_dst = np.mean(hausdorff_dst)
        
        std_pos = np.std(pos_dst)
        std_len = np.std(len_dst)
        std_ori = np.std(ori_dst)
        std_hausdorff = np.std(hausdorff_dst)
        
        total_TP = sum(quality_params[cur_job]["TP"])
        total_FP = sum(quality_params[cur_job]["FP"])
        total_FN = sum(quality_params[cur_job]["FN"])
        
        precision = total_TP / (total_TP + total_FP)
        recall = total_TP / (total_TP + total_FN)
        f1_score = 2*(precision * recall)/(precision + recall)    # harmonic mean of recall and precision
        
        quality_params[cur_job]["precision"] = precision 
        quality_params[cur_job]["recall"] = recall
        quality_params[cur_job]["f1-score"] = f1_score
        
        print("Job {}: ".format(cur_job))
        print("\t Mean position error    = {:.4f} m, std = {:.4f} m".format(mean_pos_dst, std_pos))
        print("\t Mean axis length error = {:.4f} m, std = {:.4f} m".format(mean_len_dst, std_len))
        print("\t Mean orientation error = {:.4f} °, std = {:.4f} °".format(mean_ori_dst, std_ori))
        print("\t Mean hausdorff error   = {:.4f} m, std = {:.4f} m".format(mean_hausdorff_dst, std_hausdorff))
        print("")
        print("\t Total TP = {}, FP = {}, FN = {}".format(total_TP, total_FP, total_FN))
        print("\t Precision = {:.4f} %".format(quality_params[cur_job]["precision"] * 100))
        print("\t Recall    = {:.4f} %".format(quality_params[cur_job]["recall"] * 100))
        print("\t F1-Score  = {:.4f} %".format(quality_params[cur_job]["f1-score"] * 100))  # harmonic mean of recall and precision
        print("---------------------------------")
        
        # Plot quality parameters
        # Position error, Length error, Orientation error, Hausdorff error
        plot = True
        if plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))
            
            if time == "pre":
                plt.suptitle('Histogramme der Fehler für Streifen {}\nVor der Überprüfung'.format(cur_job+1))
            if time == "post":
                if method == "admin":
                    plt.suptitle('Histogramme der Fehler für Streifen {}\nNach der Überprüfung mittels Admininterface'.format(cur_job+1))
                if method == "crowd":
                    plt.suptitle('Histogramme der Fehler für Streifen {}\nNach der Überprüfung mittels Crowd'.format(cur_job+1))
            
            xmax = 0.65; ymax = 0.35
            if np.max(pos_dst) > xmax:
                print("Attention!!!! pos of {} job - max exceeds manual set value, histogram \"range=(0,xmax)\" cuts of max x value --> adjust xmax. Old_max = {}, new_max = {}".format(cur_job+1, xmax, np.max(pos_dst)))
            color = "navajowhite"; edgecolor = "k"; alpha = 0.65
            results, edges = np.histogram(pos_dst, density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax1.axvline(mean_pos_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax1.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)
            ax1.set_xlim(right=xmax)
            ax1.set_ylim(top=ymax)
            ax1.plot([mean_pos_dst, mean_pos_dst+std_pos], [ax1.get_ylim()[1] / 2, ax1.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65)  # Stdev horizontal line, Origin in mean
            ax1.set_xlabel("Positionsfehler [m]")
            ax1.set_ylabel("Relative Häufigkeit")
            legend_elements = [ Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=2, alpha=0.65),
                                Line2D([0], [0], color="indianred", linestyle="solid", linewidth=2, alpha=0.65),]
            ax1.legend(labels=("Mittelwert = {:.2f} m".format(mean_pos_dst), "$\sigma$ = {:.2f} m".format(std_pos) ,), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)

            xmax = 1.6; ymax = 0.35
            if np.max(len_dst) > xmax:
                print("Attention!!!! len of {} job - max exceeds manual set value, histogram \"range=(0,xmax)\" cuts of max x value --> adjust xmax. Old_max = {}, new_max = {}".format(cur_job+1, xmax, np.max(len_dst)))
            results, edges = np.histogram(len_dst, density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax2.axvline(mean_len_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax2.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)
            ax2.set_xlim(right=xmax)
            ax2.set_ylim(top=ymax)
            ax2.plot([mean_len_dst, mean_len_dst+std_len], [ax2.get_ylim()[1] / 2, ax2.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65)  # Stdev horizontal line, Origin in mean
            ax2.set_xlabel("Längenfehler [m]")
            ax2.set_ylabel("Relative Häufigkeit")
            legend_elements = [Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=2, alpha=0.65),
                                Line2D([0], [0], color="indianred", linestyle="solid", linewidth=2, alpha=0.65), ]
            ax2.legend(labels=("Mittelwert = {:.2f} m".format(mean_len_dst), "$\sigma$ = {:.2f} m".format(std_len), ), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)
            
            ymax = 0.35
            results, edges = np.histogram(ori_dst, density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax3.axvline(mean_ori_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax3.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)
            ax3.set_ylim(top=ymax)
            ax3.plot([mean_ori_dst, mean_ori_dst+std_ori], [ax3.get_ylim()[1] / 2, ax3.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65)  # Stdev horizontal line, Origin in mean
            ax3.set_xlabel("Orientierungsfehler [°]")
            ax3.set_ylabel("Relative Häufigkeit")
            legend_elements = [Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=2, alpha=0.65),
                               Line2D([0], [0], color="indianred", linestyle="solid", linewidth=2, alpha=0.65), ]
            ax3.legend(labels=("Mittelwert = {:.2f} °".format(mean_ori_dst), "$\sigma$ = {:.2f} °".format(std_ori), ), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)
            
            ymax = 0.35
            results, edges = np.histogram(hausdorff_dst, density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax4.axvline(mean_hausdorff_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax4.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)            
            ax4.set_ylim(top=ymax)
            ax4.plot([mean_hausdorff_dst, mean_hausdorff_dst+std_hausdorff] , [ax4.get_ylim()[1] / 2, ax4.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65)  # Stdev horizontal line, Origin in mean
            ax4.set_xlabel("Hausdorff-Metrik [m]")
            ax4.set_ylabel("Relative Häufigkeit")            
            legend_elements = [Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=1, alpha=0.65),
                               Line2D([0], [0], color="indianred", linestyle="solid", linewidth=1, alpha=0.65), ]
            ax4.legend(labels=("Mittelwert = {:.2f} m".format(mean_hausdorff_dst), "$\sigma$ = {:.2f} m".format(std_hausdorff), ), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)
            
            fig.tight_layout()
            
            fig1 = plt.gcf()
            # figManager = plt.get_current_fig_manager() # for fullscreen
            # figManager.window.state("zoomed")
            # figManager.full_screen_toggle()
            # plt.show()     # Crashes gui, dont know why ?!
            savePlot = True
            if savePlot:
                path = 'figures/{}/'.format(cur_job+1)
                create_dir_if_not_exist(path) 
                
                if time == "pre":
                    fname = 'job_{}_{}_verification_error_distribution.png'.format(cur_job+1, time)
                if time == "post":
                    if method == "crowd":
                        fname = 'job_{}_{}_{}_error_distribution_step_{}.png'.format(cur_job+1, time, method, step)
                    if method == "admin":
                        fname = 'job_{}_{}_{}_error_distribution.png'.format(cur_job+1, time, method)
                
                path += fname
                
                fig1.savefig(path, format='png', dpi=300)
                plt.close("all") 
                
        # TP specific acquisitions ins shd
        plot = True; savePlot = True
        if plot:
            if total_TP > 0:
                cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')

                f1 = plt.figure(1)
                ax = plt.subplot(111)

                height, width = cur_img.shape    
                # extent = [0.5, width+0.5, height+0.5, 0.5]      # Account for different coordinate origin Html Canvas(0,0) == upper left corner, Matlab(1,1)==Pixel Center Upper left corner, Python(0,0) Pixel Center Upper left corner
                extent = [-1, width-1, height-1, -1]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-1, -1) via addition of constants in original webinterface
                
                ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
                
                for idx, c in enumerate(quality_params[cur_job]["TP"]):
                    if c:   #c==True
                        ax.plot([quality_params[cur_job]["x"][idx], 
                                quality_params[cur_job]["x2"][idx]],
                                [quality_params[cur_job]["y"][idx],
                                quality_params[cur_job]["y2"][idx]], color="red", linewidth=2)
                if time == "pre":
                    plt.title('TP: Streifen ={}, gesamte TP ={}\nVor der Überprüfung'.format(cur_job+1, np.sum(quality_params[cur_job]["TP"])))
                if time == "post":
                    if method == "admin":
                        plt.title('TP: Streifen ={}, gesamte TP={}\nNach der Überprüfung mittels Admin'.format(cur_job+1, np.sum(quality_params[cur_job]["TP"])))
                    if method == "crowd":
                        plt.title('TP: Streifen ={}, gesamte TP={}\nNach der Überprüfung mittels Crowd'.format(cur_job+1, np.sum(quality_params[cur_job]["TP"])))
               
                fig1 = plt.gcf()                
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                # figManager = plt.get_current_fig_manager() # for fullscreen
                # figManager.window.state("zoomed")
                # figManager.full_screen_toggle()
                # if cur_job == 1 and method == "admin":
                #    plt.show()
                    
                if savePlot:
                    path = 'figures/{}/'.format(cur_job+1)
                    create_dir_if_not_exist(path) 
                    
                    if time == "pre":
                        fname = 'job_{}_pre_verification_TP.png'.format(cur_job+1)
                    if time == "post":
                        if method == "crowd":
                            fname = 'job_{}_{}_{}_step_{}_TP.png'.format(cur_job+1, time, method, step)
                        if method == "admin":
                            fname = 'job_{}_{}_{}_TP.png'.format(cur_job+1, time, method)

                    path += fname
                    
                    fig1.savefig(path, format='png', dpi=300)
                    plt.close("all")  
        
        # FP and FN -> specific acquistions in shd
        plot = True; savePlot = True
        if plot:       
            # False Positives (FP) 
            if total_FP > 0:
                cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')

                f1 = plt.figure(1)
                ax = plt.subplot(111)

                height, width = cur_img.shape    
                # extent = [0.5, width+0.5, height+0.5, 0.5]      # Account for different coordinate origin Html Canvas(0,0) == upper left corner, Matlab(1,1)==Pixel Center Upper left corner, Python(0,0) Pixel Center Upper left corner
                extent = [-1, width-1, height-1, -1]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-1, -1) via addition of constants in original webinterface
                
                ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
                
                for idx, c in enumerate(quality_params[cur_job]["FP"]):
                    if c:   # c==True
                        ax.plot([quality_params[cur_job]["x"][idx], 
                                quality_params[cur_job]["x2"][idx]],
                                [quality_params[cur_job]["y"][idx],
                                quality_params[cur_job]["y2"][idx]], color="red", linewidth=2)
                if time == "pre":
                    plt.title('FP: Streifen ={}, gesamte FP={}\nVor der Überprüfung'.format(cur_job+1, np.sum(quality_params[cur_job]["FP"])))
                if time == "post":
                    if method == "admin":
                        plt.title('FP: Streifen ={}, gesamte FP={}\nNach der Überprüfung mittels Admin'.format(cur_job+1, np.sum(quality_params[cur_job]["FP"])))
                    if method == "crowd":
                        plt.title('FP: Streifen ={}, gesamte FP={}\nNach der Überprüfung mittels Crowd'.format(cur_job+1, np.sum(quality_params[cur_job]["FP"])))
                fig1 = plt.gcf()
                
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                # figManager = plt.get_current_fig_manager() # for fullscreen
                # figManager.window.state("zoomed")
                # figManager.full_screen_toggle()
                # plt.show()
                if savePlot:
                    path = 'figures/{}/'.format(cur_job+1)
                    create_dir_if_not_exist(path) 
                    
                    if time == "pre":
                        fname = 'job_{}_pre_verification_FP.png'.format(cur_job+1)
                    if time == "post":
                        if method == "crowd":
                            fname = 'job_{}_{}_{}_step_{}_FP.png'.format(cur_job+1, time, method, step)
                        if method == "admin":
                            fname = 'job_{}_{}_{}_FP.png'.format(cur_job+1, time, method)
                    
                    path += fname
                    
                    fig1.savefig(path, format='png', dpi=300)
                    plt.close("all")  
            # False Negatives (FN)
            if total_FN > 0:
                cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
                
                f1 = plt.figure(1)
                ax = plt.subplot(111)
                
                height, width = cur_img.shape    
                # extent = [0.5, width+0.5, height+0.5, 0.5]      # Account for different coordinate origin Html Canvas(0,0) == upper left corner, Matlab(1,1)==Pixel Center Upper left corner, Python(0,0) Pixel Center Upper left corner
                extent = [-1, width-1, height-1, -1]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-1, -1) via addition of constants in original webinterface
                
                ax.imshow(cur_img, cmap="gray", origin="upper", extent=extent, interpolation='none')
                
                for idx in range(len(cars_in_shds[cur_job]["gt"]["start"]["x"])):
                    if quality_params[cur_job]["FN"][idx]:  # == True
                        ax.plot([cars_in_shds[cur_job]["gt"]["start"]["x"][idx], 
                                cars_in_shds[cur_job]["gt"]["end"]["x"][idx]],
                                [cars_in_shds[cur_job]["gt"]["start"]["y"][idx],
                                cars_in_shds[cur_job]["gt"]["end"]["y"][idx]], color="red", linewidth=2)
                
                if time == "pre":
                    plt.title('FN: Streifen ={}, gesamte FN={}\nVor der Überprüfung'.format(cur_job+1, np.sum(quality_params[cur_job]["FN"])))
                if time == "post":
                    if method == "admin":
                        plt.title('FN: Streifen ={}, gesamte FN={}\nNach der Überprüfung mittels Admin'.format(cur_job+1, np.sum(quality_params[cur_job]["FN"])))
                    if method == "crowd":
                        plt.title('FN: Streifen ={}, gesamte FN={}\nNach der Überprüfung mittels Crowd'.format(cur_job+1, np.sum(quality_params[cur_job]["FN"])))
                
                fig1 = plt.gcf()
                
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                # figManager = plt.get_current_fig_manager() # for fullscreen
                # figManager.full_screen_toggle()
                # figManager.window.state("zoomed")
                # plt.show()
                if savePlot:
                    path = 'figures/{}/'.format(cur_job+1)
                    create_dir_if_not_exist(path) 
                    
                    if time == "pre":
                        fname = 'job_{}_pre_verification_FN.png'.format(cur_job+1)
                    if time == "post":
                        if method == "crowd":
                            fname = 'job_{}_{}_{}_step_{}_FN.png'.format(cur_job+1, time, method, step)
                        if method == "admin":
                            fname = 'job_{}_{}_{}_FN.png'.format(cur_job+1, time, method)
                    
                    path += fname
                    
                    fig1.savefig(path, format='png', dpi=300)
                    plt.close("all")  

    ## Plot Combined Error Distributions
    pos_dst =[]; len_dst = []; ori_dst = []; hausdorff_dst = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):      
        pos_dst.extend(quality_params[cur_job]["err"]["pos"])
        len_dst.extend(quality_params[cur_job]["err"]["len"])
        ori_dst.extend(quality_params[cur_job]["err"]["ori"])  # * 180 / math.pi
        hausdorff_dst.extend(quality_params[cur_job]["err"]["hausdorff"])
    
    pos_dst = np.array(pos_dst) * config["integration"]["cellSize"]
    len_dst = np.array(len_dst) * config["integration"]["cellSize"]
    hausdorff_dst = np.array(hausdorff_dst) * config["integration"]["cellSize"]
    
    mean_pos_dst = np.mean(pos_dst)
    mean_len_dst = np.mean(len_dst)
    mean_ori_dst = np.mean(ori_dst)
    mean_hausdorff_dst = np.mean(hausdorff_dst)
    
    std_pos = np.std(pos_dst)
    std_len = np.std(len_dst)
    std_ori = np.std(ori_dst)
    std_hausdorff = np.std(hausdorff_dst)
    
    print("Combination of all jobs: ".format(cur_job))
    print("\t Mean position error    = {:.4f} m, std = {:.4f} m".format(mean_pos_dst, std_pos))
    print("\t Mean axis length error = {:.4f} m, std = {:.4f} m".format(mean_len_dst, std_len))
    print("\t Mean orientation error = {:.4f} °, std = {:.4f} °".format(mean_ori_dst, std_ori))
    print("\t Mean hausdorff error   = {:.4f} m, std = {:.4f} m".format(mean_hausdorff_dst, std_hausdorff))
    print("")
    
    # Export to display n = 25 and n = 50 combined
    export = {"pos_dst": pos_dst, "len_dst": len_dst, "hausdorff_dst": hausdorff_dst, "ori_dst": ori_dst}
    
    if len(cars_in_shds[0]["workerId"]) == 50:
        if time == "post":
            path = "plotData/qualityParams_n=50_{}_{}".format(time, method)
        if time == "pre":
            path = "plotData/qualityParams_n=50_{}".format(time)
    elif len(cars_in_shds[0]["workerId"]) == 25:
        if time == "post":
            path = "plotData/qualityParams_n=25_{}_{}".format(time, method)
        if time == "pre":
            path = "plotData/qualityParams_n=25_{}".format(time)
    with open(path, 'wb') as file:
        obj = (export)
        pickle.dump(obj, file)
    
    # Plot quality parameters
    # Position error, Length error, Orientation error, Hausdorff error
    try:
        plot = True
        if plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))
            
            if time == "pre":
                plt.suptitle('Histogramme der Fehler für alle Streifen\nVor der Überprüfung')
            if time == "post":
                if method == "admin":
                    plt.suptitle('Histogramme der Fehler für alle Streifen\nNach der Überprüfung mittels Admin')
                if method == "crowd":
                    plt.suptitle('Histogramme der Fehler für alle Streifen\nNach der Überprüfung mittels Crowd')
            
            # def Standardize(distribution):
            #    newDistribution = (distribution-np.mean(distribution))/np.std(distribution)
            #    return newDistribution
            
            xmax = 0.7; ymax = 0.35 # Max of pre, admin, crowd ratings  !!!!!!!!!!!!!!!!!!!!!!!!
            if np.max(pos_dst) > xmax:
                print("Attention!!!! pos - max exceeds manual set value, histogram \"range=(0,xmax)\" cuts of max x value --> adjust xmax. Old_max = {}, new_max = {}".format(xmax, np.max(pos_dst)))
            
            color = "navajowhite"; edgecolor = "k"; alpha = 0.65
            results, edges = np.histogram(pos_dst, range=(0, xmax), density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax1.axvline(mean_pos_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax1.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)
            ax1.set_xlim(right=xmax)
            ax1.set_ylim(top=ymax)
            ax1.plot([mean_pos_dst, mean_pos_dst+std_pos], [ax1.get_ylim()[1] / 2, ax1.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65)  # Stdev horizontal line, Origin in mean
            ax1.set_xlabel("Positionsfehler [m]")
            ax1.set_ylabel("Relative Häufigkeit")
            legend_elements = [Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=2, alpha=0.65),
                               Line2D([0], [0], color="indianred", linestyle="solid", linewidth=2, alpha=0.65), ]
            ax1.legend(labels=("Mittelwert = {:.2f} m".format(mean_pos_dst), "$\sigma$ = {:.2f} m".format(std_pos), ), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)

            xmax = 1.8; ymax = 0.35
            if np.max(len_dst) > xmax:
                print("Attention!!!! length - max exceeds manual set value, histogram \"range=(0,xmax)\" cuts of max x value --> adjust xmax; Old_max = {}, new_max = {}".format(xmax, np.max(len_dst)))
            results, edges = np.histogram(len_dst, range=(0, xmax), density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax2.axvline(mean_len_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax2.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)
            ax2.set_xlim(right=xmax)
            ax2.set_ylim(top=ymax)
            ax2.plot([mean_len_dst, mean_len_dst+std_len], [ax2.get_ylim()[1] / 2, ax2.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65)  # Stdev horizontal line, Origin in mean
            ax2.set_xlabel("Längenfehler [m]")
            ax2.set_ylabel("Relative Häufigkeit")
            legend_elements = [Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=2, alpha=0.65),
                               Line2D([0], [0], color="indianred", linestyle="solid", linewidth=2, alpha=0.65),]
            ax2.legend(labels=("Mittelwert = {:.2f} m".format(mean_len_dst), "$\sigma$ = {:.2f} m".format(std_len), ), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)
            
            xmax = 12; ymax = 0.40
            if np.max(ori_dst) > xmax:
                print("Attention!!!! Orientation - max exceeds manual set value, histogram \"range=(0,xmax)\" cuts of max x value --> adjust xmax; Old_max = {}, new_max = {}".format(xmax, np.max(ori_dst)))
            results, edges = np.histogram(ori_dst, range=(0, xmax), density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax3.axvline(mean_ori_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax3.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)
            ax3.set_xlim(right=xmax)
            ax3.set_ylim(top=ymax)
            ax3.plot([mean_ori_dst, mean_ori_dst+std_ori] , [ax3.get_ylim()[1] / 2, ax3.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65)  # Stdev horizontal line, Origin in mean
            ax3.set_xlabel("Orientierungsfehler [°]")
            ax3.set_ylabel("Relative Häufigkeit")
            legend_elements = [ Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=2, alpha=0.65),
                                Line2D([0], [0], color="indianred", linestyle="solid", linewidth=2, alpha=0.65),]
            ax3.legend(labels=("Mittelwert = {:.2f} °".format(mean_ori_dst), "$\sigma$ = {:.2f} °".format(std_ori) ,), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)
            
            xmax = 1.2; ymax = 0.35
            if np.max(hausdorff_dst) > xmax:
                print("Attention!!!! hausdorff - max exceeds manual set value, histogram \"range=(0,xmax)\" cuts of max x value --> adjust xmax; Old_max = {}, new_max = {}".format(xmax, np.max(hausdorff_dst)))   
            results, edges = np.histogram(hausdorff_dst, range=(0, xmax), density=True, bins=10)
            binWidth = edges[1] - edges[0]
            ax4.axvline(mean_hausdorff_dst, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=2)   # Mean vertical line
            ax4.bar(edges[:-1], results*binWidth, binWidth, color=color, edgecolor=edgecolor, alpha=alpha)
            ax4.set_xlim(right=xmax)
            ax4.set_ylim(top=ymax)
            ax4.plot([mean_hausdorff_dst, mean_hausdorff_dst+std_hausdorff] , [ax4.get_ylim()[1] / 2, ax4.get_ylim()[1] / 2], color="indianred", linestyle="solid", linewidth=2, alpha=0.65) # Stdev horizontal line, Origin in mean
            ax4.set_xlabel("Hausdorff-Metrik [m]")
            ax4.set_ylabel("Relative Häufigkeit")            
            legend_elements = [ Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=1, alpha=0.65),
                                Line2D([0], [0], color="indianred", linestyle="solid", linewidth=1, alpha=0.65),]
            ax4.legend(labels=("Mittelwert = {:.2f} m".format(mean_hausdorff_dst), "$\sigma$ = {:.2f} m".format(std_hausdorff),), handles=legend_elements, shadow=True, loc="upper right", handlelength=1.5)
            
            fig.tight_layout()
            
            plt.subplots_adjust(bottom=0.045, hspace=0.27, top=0.95, right=0.975, wspace=0.202, left=0.079)
            
            fig1 = plt.gcf()
            # figManager = plt.get_current_fig_manager() # for fullscreen
            # figManager.window.state("zoomed")
            # figManager.full_screen_toggle()
            # plt.show()     # Crashes gui, dont know why ?!
            savePlot = True
            if savePlot:
                path = 'figures/overall_quality_parameters/'.format(cur_job+1)
                create_dir_if_not_exist(path) 
                
                if time == "pre":
                    fname = 'combined_jobs_{}_verification_error_distribution.png'.format(time)
                if time == "post":
                    if method == "crowd":
                        fname = 'combined_jobs_{}_{}_error_distribution_step_{}.png'.format(time, method, step)
                    if method == "admin":
                        fname = 'combined_jobs_{}_{}_error_distribution.png'.format(time, method)
                
                path += fname
                
                fig1.savefig(path, format='png', dpi=300)
                plt.close("all") 
    except:
        print("Error: Plotting Position error, Length error, Orientation error, Hausdorff error")
    
    ## Plot Precision, Recall, F1-Score
    try:
        # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py 2.2.2021
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot 2.2.2021
        labels = [ str(ele) for ele in [*range(1,config["jobs"]["number_of_jobs"]+1)] ]

        x = np.arange(len(labels)) # label location
        barWidth = 0.25 # bar width

        # Set position of bar on x axis
        r1 = np.arange(len(labels))
        r2 = [x + barWidth + 0.01 for x in r1]
        r3 = [x + barWidth + 0.01 for x in r2]

        fig, ax = plt.subplots(figsize=(14,3))
        
        precision = []; recall = []; f1_score = []
        for cur_job in range(config["jobs"]["number_of_jobs"]):        
            precision.append(quality_params[cur_job]["precision"])
            recall.append(quality_params[cur_job]["recall"])
            f1_score.append(quality_params[cur_job]["f1-score"])

        precision_p = [ round(elem*100, 1) for elem in precision ]    # Round value 2decimal
        recall_p = [ round(elem*100, 1) for elem in recall ]
        f1_score_p = [ round(elem*100, 1) for elem in f1_score ]
        
        # Write to file for CD
        __path = "plotted_data_textformat/allgemeine_daten/"
        create_dir_if_not_exist(__path)
        if time == "pre":
            __fname = "Genauigkeitsmasse_{}_Ueberpruefung.txt".format(time)
        if time == "post":
            __fname = "Genauigkeitsmasse_{}_Ueberpruefung_durch_{}.txt".format(time, method)
            
        with open(__path + __fname, "w") as f:
            f.write("Genauigkeitsmasse_{}_Ueberpruefung\n".format(time))
            f.write("Precision in [%]\n")
            for ele in precision_p:
                f.write("{}, ".format(ele))
            f.write("\n\nRecall in [%]\n")
            for ele in recall_p:
                f.write("{}, ".format(ele))
            f.write("\n\nF1-Score in [%]\n")
            for ele in f1_score_p:
                f.write("{}, ".format(ele))

        color_prec = (0.7019607843137254, 0.803921568627451, 0.8901960784313725, 1.0)
        color_rec = (0.8705882352941177, 0.796078431372549, 0.8941176470588236, 1.0)
        color_f1 = (0.996078431372549, 0.8509803921568627, 0.6509803921568628, 1.0)

        rects1 = ax.bar(r1, precision_p, width=barWidth, label = "Precision", color=color_prec)
        rects2 = ax.bar(r2, recall_p, width=barWidth, label = "Recall", color=color_rec)
        rects3 = ax.bar(r3, f1_score_p, width=barWidth, label = "F1-Score", color=color_f1)    
        
        def autolabel(rects):
            """ 
            Attach a text label above each bar in *rects*, displaying its height.
            """
            for rect in rects:
                height = rect.get_height()             
                #print((rect.get_x() + rect.get_width() / 2, height - 5))
                ax.annotate("{}".format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height - 5),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)    
        ax.set_xticks([r + barWidth for r in range(len(labels))])
        ax.set_xticklabels(labels)
        ax.set_ylim(80, 105)
        ax.set_xlabel("Streifen")
        ax.set_ylabel("%")
        if time == "pre":
            ax.set_title("Genauigkeitsmaße (Vor der Überprüfung):\nPrecision, Recall und F1-Score")
        if time == "post":
            if method == "admin":
                ax.set_title("Genauigkeitsmaße (Nach der Überprüfung mittels Admin):\nPrecision, Recall und F1-Score")
            if method == "crowd":
                ax.set_title("Genauigkeitsmaße (Nach der Überprüfung mittels Crowd):\nPrecision, Recall und F1-Score")
        # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15),
        #      fancybox=True, shadow=True, ncol=5)
        fig.tight_layout()

        fig1 = plt.gcf()
        
        manager = plt.get_current_fig_manager()
        # manager.resize(*manager.window.maxsize())
        # figManager = plt.get_current_fig_manager() # for fullscreen
        # figManager.full_screen_toggle()
        # figManager.window.state("zoomed")
        # plt.show()
        if savePlot:
            path = "figures/overall_quality_parameters/"
            create_dir_if_not_exist(path) 

            if time == "pre":
                fname = "pre_verification_precision_recall_f1.png"
            if time == "post":
                if method == "crowd":
                    fname = "post_{}interface_step_{}_precision_recall_f1.png".format(method, step)
                if method == "admin":
                    fname = "post_{}interface_precision_recall_f1.png".format(method)

            path += fname

            fig1.savefig(path, format="png", dpi=300)
            plt.close("all")  
    except:
        print("Error plotting precision, recall, f1")
        
    return quality_params


def calc_sub_it_numb(config):
    path = "Crowdinterface/Pre Rating/"        
    sub_it_numbers = []; number_of_files = 0
    for cur_batch in range(config["jobs"]["number_of_jobs"]):
        sub_it_numb = len(glob.glob(path + str(cur_batch+1) + "*"))
        number_of_files += sub_it_numb
        sub_it_numbers.append([*range(sub_it_numb)])
    
    # sub_it_numb = [ [*range(x)] for x in config["interface_questions"]["sub_it_numb"]]
    return sub_it_numbers, number_of_files


def rate_questions(directory, config):
    """
    Calculates Rating using 1, 3 or 5 answers followed by a majority voting, in the thesis, only option with all 5 answers is included
    
    Parameters:
    ----------
        directory: string
        
    Returns:
    ----------
        returnDict: dict
            all answers
        dbRatingResult: dict
            rating result for uncertain acquisitions detected via 2nd DBSCAN, same form as used in admin rating process
        ellRatingResult: dict
            rating result for uncertain acquisitions detected via ellipse criteria, same form as used in admin rating process
        count_clear_majority: dict
            count clear/unclear majority occurrences for 3rd question
    """
    
    root_local_result = directory["rootDir_local"]
    subDir = "/results"
    
    # Read result .txt files
    batch_numb = [*range(1,config["jobs"]["number_of_jobs"]+1)]
    
    sub_it_numb, _ = calc_sub_it_numb(config)
    # sub_it_numb = [ [*range(x)] for x in config["interface_questions"]["sub_it_numb"]]
    it_numb = [*range(1, config["interface_questions"]["it_numb"]+1)]  
    
    # Init dict to store data
    returnDict = {key: { keySub: {"method": [], "clusterIdx": [], "workerId": [], "coord": {"x": [], "y": [], "x2": [],"y2": []}, "clusterCoord": {"x": [], "y": []}, "answers": {"all": [], "steps": []}} for keySub in sub_it_numb[key-1]} for key in batch_numb}
    
    dbRatingResult = {}
    ellRatingResult = {}
    count_clear_majority = {1: [], 3: [], 5: []}
    id_no_clear_majority = {1: [], 3: [], 5: []}
    for cur_batch in batch_numb:
        print("calc rating for batch {} ...".format(cur_batch))
        for cur_sub_it in sub_it_numb[cur_batch-1]:     
            # Read interface input file
            root_local_input = "Crowdinterface/Pre Rating/"
            fname = "{}_{}_uncertain_crowd.txt".format(cur_batch, cur_sub_it)
            with open(root_local_input + fname) as f:
                for _ in range(3):
                    next(f) # Skip header (2 lines) and first data line (reference question)  
                for line in f:                    
                    line = line.split(",")
                    
                    method = re.sub("\n", "", line[-1])
                    workerId = line[-2]
                    
                    returnDict[cur_batch][cur_sub_it]["method"].append(method)
                    returnDict[cur_batch][cur_sub_it]["workerId"].append(workerId)   # Get workerId for cur_batch and cur_sub_it
                    
                    clusterIdx = int(line[0])
                    returnDict[cur_batch][cur_sub_it]["clusterIdx"].append(clusterIdx)
                    
                    acquiCoord = line[1].split(" ")
                    returnDict[cur_batch][cur_sub_it]["coord"]["x"].append(float(acquiCoord[0]))
                    returnDict[cur_batch][cur_sub_it]["coord"]["y"].append(float(acquiCoord[1]))
                    returnDict[cur_batch][cur_sub_it]["coord"]["x2"].append(float(acquiCoord[2]))
                    returnDict[cur_batch][cur_sub_it]["coord"]["y2"].append(float(acquiCoord[3]))
                    
                    clusterCoord = line[2].split(" ")
                    returnDict[cur_batch][cur_sub_it]["clusterCoord"]["x"].append(float(clusterCoord[0]))
                    returnDict[cur_batch][cur_sub_it]["clusterCoord"]["y"].append(float(clusterCoord[1]))
            
            # Read interface result files
            answers = []
            for cur_it in it_numb:
                fname = "{}-{}-{}.txt".format(cur_batch, cur_sub_it, cur_it)
                path = root_local_result + subDir + "/"
                answer = []
                with open(path + fname) as f:
                    next(f) # Skip answer for reference
                    for line in f:
                        line = line.split("\t")
                        a1 = line[0]
                        a2 = line[1]
                        a3 = line[2]
                        a4 = re.sub("\n", "", line[-1])
                        answer.append([a1, a2, a3, a4])
                answers.append(answer)

            # Reorganize and calculate final rating           
            returnDict[cur_batch][cur_sub_it]["answers"]["all"] = [None] * len(answer)
            returnDict[cur_batch][cur_sub_it]["answers"]["steps"] = [None] * len(answer)  

            try:
                for x in range(len(answer)) :
                    returnDict[cur_batch][cur_sub_it]["answers"]["all"][x] = []
                    returnDict[cur_batch][cur_sub_it]["answers"]["steps"][x] = []
                    for cur_it in it_numb:
                        cur_answer = answers[cur_it-1][x]
                        
                        returnDict[cur_batch][cur_sub_it]["answers"]["all"][x].append(cur_answer)
                    
                    # Evaluate answers in staggered steps 1, 3 or 5 answers for majority vote
                    steps = [1, 3, 5]            
                    returnDict[cur_batch][cur_sub_it]["answers"]["steps"][x] = { key: {keySub: None for keySub in ["a1", "a2", "a3", "a4", "rating"]} for key in steps }#{ key: None for key in steps }
                    
                    for step in steps:
                        answers_of_interest = []
                        for idx in range(step):                        
                            answers_of_interest.append(returnDict[cur_batch][cur_sub_it]["answers"]["all"][x][idx])
                            
                        # Rating
                        answers_split = list(zip(*answers_of_interest))

                        # Simple "Yes"/"No" questions
                        majority_a1 = max(set(answers_split[0]), key=answers_split[0].count)   # Most occuring element
                        majority_a2 = max(set(answers_split[1]), key=answers_split[1].count)
                        majority_a4 = max(set(answers_split[3]), key=answers_split[3].count)
                        # 0, 1, 2 question -> # Problem if no clear majority, e.g. step3: [0,1,2], step5: [2,2,1,1,0], [2,2,1,0,0], [2,1,1,0,0] -> take other answers into account
                        # [0,1,2] -> [1] (Majority thinks at least 1 point, no clear majority so 1 is assumed to be correct)
                        # [2,2,1,1,0] -> [1] (Majority thinks at least 1 point, no clear majority so 1 is assumed to be correct)
                        # [2,2,1,0,0] -> [2] (Majority thinks at least 1 point, and majority of that is 2)
                        # [2,1,1,0,0] -> [1] (Majority thinks at least 1 point, and majority of that is 1)
                        # x = ("2", "1", "1", "0", "0")
                        c = Counter(answers_split[2])
                        maxFreq = max(c.values())                    
                        indices = [i for i, x in enumerate(c.values()) if x == maxFreq]

                        if len(indices ) == 3: # e.g. step 3 -> ("2", "1", "0" ) --> indices = [0,1,2]
                            majority_a3 = "1"                            
                            count_clear_majority[step].append(False)
                            id_no_clear_majority[step].append("batch={}_subit={}_quest={}".format(cur_batch, cur_sub_it, x))
                            # print("majority vote not clear set to {}, step {}".format(majority_a3, step))
                        if len(indices ) == 2: # e.g. step 3 -> ("2", "2", "1", "1", "0" ) --> indices = [1,2]
                            # val1 = int(list(c.keys())[indices[0]])
                            # val2 = int(list(c.keys())[indices[1]])
                            # majority_a3 = str(max([val1,val2]))
                            
                            vals = [int(list(c.keys())[indices[0]]), int(list(c.keys())[indices[1]])]
                            vals.sort(reverse = True)   # Sort descending [1,2].sort(reverse=True) -> [2,1]                        
                            if vals[0] == 2 and vals[1] == 1:
                                majority_a3 = "1"
                            elif vals[0] == 2 and vals[1] == 0:
                                majority_a3 = "2"
                            elif vals[0] == 1 and vals[1] == 0:
                                majority_a3 = "1"                            
                            
                            count_clear_majority[step].append(False)
                            id_no_clear_majority[step].append("batch={}_subit={}_quest={}".format(cur_batch, cur_sub_it, x))
                            # print("majority vote not clear set to {}, step {}".format(majority_a3, step))
                            
                        if len(indices) == 1: # e.g. step 5 -> ("2", "2", "2", "2", "0") --> indice = [1]
                            majority_a3 = max(set(answers_split[2]), key=answers_split[2].count)
                            count_clear_majority[step].append(True)
                        if majority_a3 is None:
                            print("ERROR: Something went wrong while calculating the ratings")
                            raise Exception("Something went wrong while calculating the ratings")
                        
                        if majority_a1 == "Yes":    # Cluster?                        
                            
                            if majority_a2 == "Yes":    # Fully visible?
                                
                                if majority_a3 == "0" or majority_a3 == "1":    # Only 1 or 2 line ends touch front/back edge of car 
                                    rating = "NOK"       
                                                            
                                if majority_a3 == "2":  # 2 line ends touch front/back edge of car
                                    
                                    if majority_a4 == "Yes":    # Parallel to axis?
                                        rating = "OK"
                                    else:
                                        rating = "NOK"                            
                                    
                            elif majority_a2 == "No":   # NOT fully visible?
                                
                                if majority_a3 == "0":  # Line touches no edge of front/back of car
                                    rating = "NOK"
                                
                                if majority_a3 == "1":  # 1 line end touches edge?
                                    
                                    if majority_a4 == "Yes":    # Parallel?
                                        rating = "OK"
                                    else:
                                        rating = "NOK"
                                
                                if majority_a3 == "2":    # 2 line ends touch front/back edge of car 
                                    rating = "OK"
                                
                        else:   # Not part of correct cluster
                            rating = "NOK"

                        if count_clear_majority[step][-1] == True:
                            plot=True; savePlot=True
                            if plot:
                                x1 = returnDict[cur_batch][cur_sub_it]["coord"]["x"][x]
                                y1 = returnDict[cur_batch][cur_sub_it]["coord"]["y"][x]
                                x2 = returnDict[cur_batch][cur_sub_it]["coord"]["x2"][x]
                                y2 = returnDict[cur_batch][cur_sub_it]["coord"]["y2"][x]
                                
                                cur_img = plt.imread(config["directories"]["Img_Folder"] + 'job' + str(cur_batch) + '/' + 'shd.png')
                                
                                f1 = plt.figure(1)
                                ax = plt.subplot(111)                                
                                height, width = cur_img.shape
                                extent = [-1, width-1, height-1, -1 ]
                                ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
                                
                                xmin = 50000
                                ymin = 50000
                                xmax = 0
                                ymax = 0
                                                                    
                                if x1<xmin:
                                    xmin = x1
                                if x2<xmin:
                                    xmin = x2
                                if x1>xmax:
                                    xmax = x1
                                if x2>xmax:
                                    xmax = x2
                                if y1<ymin:
                                    ymin = y1
                                if y2<ymin:
                                    ymin = y2
                                if y1>ymax:
                                    ymax = y1
                                if y2>ymax:
                                    ymax = y2                        
                                ax.plot([x1, x2],
                                        [y1, y2], color="red", linewidth= 2)
                                
                                ax.plot([],[], color="red", label=r'Erfassung')  # empty only legend label

                                ax.axis([xmin-20,xmax+20,ymax+20,ymin-20])          #flipped    , origin upper left corner 
                                ax.set_title('Streifen {}, Sub iter nummer={}, eindeutige Mehrheit für Schritt={},\nbestimmte Mehrheit={} -> Bewertung = {}'.format(
                                         cur_batch, cur_sub_it+1, step, [majority_a1, majority_a2, majority_a3, majority_a4], rating))
                                ax.set_xlabel("x [px]")
                                ax.set_ylabel("y [px]")
                                ax.legend()

                                fig1 = plt.gcf()
                                manager = plt.get_current_fig_manager()
                                manager.resize(*manager.window.maxsize())
                                if savePlot:                                        
                                    path = config["directories"]["Figures"] +'{}/Verify_Rating/crowd/{}/{}/'.format(cur_batch, step, rating)
                                    create_dir_if_not_exist(path)                                    
                                    fname = 'job_{}_sub_it_{}_step_{}_entry_{}.png'.format(cur_batch, cur_sub_it, step, x)
                                    path += fname                                    
                                    fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")       # , bbox_inches="tight" -> save legend to plot
                                    plt.close("all")

                        # Plot acquisitions where majority vote is unclear
                        if count_clear_majority[step][-1] == False:
                            plot=True; savePlot=True
                            if plot:
                                x1 = returnDict[cur_batch][cur_sub_it]["coord"]["x"][x]
                                y1 = returnDict[cur_batch][cur_sub_it]["coord"]["y"][x]
                                x2 = returnDict[cur_batch][cur_sub_it]["coord"]["x2"][x]
                                y2 = returnDict[cur_batch][cur_sub_it]["coord"]["y2"][x]

                                cur_img = plt.imread(config["directories"]["Img_Folder"] + 'job' + str(cur_batch) + '/' + 'shd.png')

                                f1 = plt.figure(1)
                                ax = plt.subplot(111)                                
                                height, width = cur_img.shape
                                extent = [-1, width-1, height-1, -1 ]
                                ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')

                                xmin = 50000
                                ymin = 50000
                                xmax = 0
                                ymax = 0

                                if x1<xmin:
                                    xmin = x1
                                if x2<xmin:
                                    xmin = x2
                                if x1>xmax:
                                    xmax = x1
                                if x2>xmax:
                                    xmax = x2
                                if y1<ymin:
                                    ymin = y1
                                if y2<ymin:
                                    ymin = y2
                                if y1>ymax:
                                    ymax = y1
                                if y2>ymax:
                                    ymax = y2                        
                                ax.plot([x1, x2],
                                        [y1, y2], color="red", linewidth= 2)
                                
                                ax.plot([],[], color="red", label=r'Erfassung')  # empty only legend label
                                    
                                ax.axis([xmin-20,xmax+20,ymax+20,ymin-20])          #flipped    , origin upper left corner 
                                ax.set_title('Streifen {}, Sub iter nummer={}, Mehrheit uneindeutig für Schritt={},\nbestimmte Mehrheit={} -> Bewertung = {}'.format(
                                         cur_batch, cur_sub_it+1, step, [majority_a1, majority_a2, majority_a3, majority_a4], rating))
                                ax.set_xlabel("x [px]")
                                ax.set_ylabel("y [px]")
                                ax.legend()

                                fig1 = plt.gcf()
                                manager = plt.get_current_fig_manager()
                                manager.resize(*manager.window.maxsize())
                                if savePlot:
                                    path = config["directories"]["Figures"] +'{}/Verify_Rating/crowd/{}/{}/'.format(cur_batch, step, rating)
                                    create_dir_if_not_exist(path) 

                                    fname = 'job_{}_sub_it_{}_majority_unclear_step_{}_entry_{}.png'.format(cur_batch, cur_sub_it, step, x)
                                    path += fname

                                    fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")       # , bbox_inches="tight" -> save legend to plot
                                    plt.close("all")

                        # if rating == "OK":
                        # print("\n")
                        # print("batch={}, subit={}, zeile in txt={}, step={}".format(cur_batch, cur_sub_it, x+1,step))
                        # print(majority_a1 + ", " + majority_a2 + ", " + majority_a3 + ", " + majority_a4 + " -> rating => ", rating)
                        returnDict[cur_batch][cur_sub_it]["answers"]["steps"][x][step]["a1"] = majority_a1
                        returnDict[cur_batch][cur_sub_it]["answers"]["steps"][x][step]["a2"] = majority_a2
                        returnDict[cur_batch][cur_sub_it]["answers"]["steps"][x][step]["a3"] = majority_a3
                        returnDict[cur_batch][cur_sub_it]["answers"]["steps"][x][step]["a4"] = majority_a4
                        returnDict[cur_batch][cur_sub_it]["answers"]["steps"][x][step]["rating"] = rating
            except:
                print("Error calculating Ratings 1")
            
        # Reorganize to same format as used in the admininterface, unnötig, hätte ich au glei in das format lesen können :D
        idc_db_acqui = []; idc_ell_acqui = []; sub_it = []
        try:
            for cur_sub_it in returnDict[cur_batch]:
                sub_it_data = returnDict[cur_batch][cur_sub_it]            
                for idx, method in enumerate(sub_it_data["method"]):
                    if method == "db_weak":
                        sub_it.append(cur_sub_it)
                        idc_db_acqui.append(sub_it_data["clusterIdx"][idx])
                    if method =="ellipse":
                        idc_ell_acqui.append(sub_it_data["clusterIdx"][idx])
            
            dbRatingResult[cur_batch-1] = {}
            ellRatingResult[cur_batch-1] = {}
        except:
            print("Error reorganizing dicts")
        
        try:
            for cluster_idx in set(idc_db_acqui):   # set(idc_db_acqui) -> unqiue cluster idc list         
                dbRatingResult[cur_batch-1][cluster_idx] = []
                
                acqui_count = idc_db_acqui.count(cluster_idx)
                # Fill dict
                for cur_sub_it in returnDict[cur_batch]:
                    sub_it_data = returnDict[cur_batch][cur_sub_it]            
                    for idx, method in enumerate(sub_it_data["method"]):
                        if method == "db_weak":
                            if cluster_idx == sub_it_data["clusterIdx"][idx]:       
                                dbRatingResult[cur_batch-1][cluster_idx].append({ "acquiCoord": {"x": [], "y": [], "w": [], "z": []}, "clusterCoord": {"xMean": [], "yMean": []}, "workerId": [], "finalRating": {"steps": {1: [], 3: [], 5: []}}})
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["x"] = sub_it_data["coord"]["x"][idx]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["y"] = sub_it_data["coord"]["y"][idx]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["z"] = sub_it_data["coord"]["x2"][idx]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["w"] = sub_it_data["coord"]["y2"][idx]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["clusterCoord"]["xMean"] = sub_it_data["clusterCoord"]["x"][idx]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["clusterCoord"]["yMean"] = sub_it_data["clusterCoord"]["y"][idx]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["workerId"] = sub_it_data["workerId"][idx]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["finalRating"]["steps"][1] = sub_it_data["answers"]["steps"][idx][1]["rating"]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["finalRating"]["steps"][3] = sub_it_data["answers"]["steps"][idx][3]["rating"]
                                dbRatingResult[cur_batch-1][cluster_idx][-1]["finalRating"]["steps"][5] = sub_it_data["answers"]["steps"][idx][5]["rating"]
        except:
            print("Error reorganizing dicts 2")
        
        try: 
            for cluster_idx in set(idc_ell_acqui):            
                ellRatingResult[cur_batch-1][cluster_idx] = []
                
                acqui_count = idc_ell_acqui.count(cluster_idx)
                # for acqui_idx in range(acqui_count):
                # Fill dict
                for cur_sub_it in returnDict[cur_batch]:
                    sub_it_data = returnDict[cur_batch][cur_sub_it]            
                    for idx, method in enumerate(sub_it_data["method"]):
                        if method =="ellipse":
                            if cluster_idx == sub_it_data["clusterIdx"][idx]:      
                                ellRatingResult[cur_batch-1][cluster_idx].append({ "acquiCoord": {"x": [],"y": [],"w": [],"z": []}, "clusterCoord": {"xMean": [], "yMean": []}, "workerId": [], "finalRating": {"steps": {1:[],3:[],5:[]}} })                  
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["x"] = returnDict[cur_batch][cur_sub_it]["coord"]["x"][idx]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["y"] = returnDict[cur_batch][cur_sub_it]["coord"]["y"][idx]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["z"] = returnDict[cur_batch][cur_sub_it]["coord"]["x2"][idx]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["acquiCoord"]["w"] = returnDict[cur_batch][cur_sub_it]["coord"]["y2"][idx]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["clusterCoord"]["xMean"] = returnDict[cur_batch][cur_sub_it]["clusterCoord"]["x"][idx]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["clusterCoord"]["yMean"] = returnDict[cur_batch][cur_sub_it]["clusterCoord"]["y"][idx]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["workerId"] = returnDict[cur_batch][cur_sub_it]["workerId"][idx]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["finalRating"]["steps"][1] = returnDict[cur_batch][cur_sub_it]["answers"]["steps"][idx][1]["rating"]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["finalRating"]["steps"][3] = returnDict[cur_batch][cur_sub_it]["answers"]["steps"][idx][3]["rating"]
                                ellRatingResult[cur_batch-1][cluster_idx][-1]["finalRating"]["steps"][5] = returnDict[cur_batch][cur_sub_it]["answers"]["steps"][idx][5]["rating"]
        except:
            print("Error reorganizing dicts 3")
    return [ returnDict, dbRatingResult, ellRatingResult, count_clear_majority, id_no_clear_majority ] 


def combine_ratings(cars_in_shds, ell_rating_result, db_rating_result, method, config, step=None):
    """ 
    Combine Ratings (Clear Outlier, Integrated Data, Rating Result from Admin/Crowdwebinterface)
    
    Parameters:
    ----------
        method: string
            "admin" if admininterface is rating base
            "crowd" of crowdinterface is rating base
    
    Returns:
    ----------
        worker_rating: dict
    """
    print("\n--------------------------------------------------")
    print(" Combine Ratings with results from {}interface ".format(method))
    print("--------------------------------------------------")
    # Create dict to store ratings in
    worker_rating = {}
    
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        worker_rating[cur_job] = dict((el,{"RateByCrowd": { "idc": [], "a": []}, "OK": { "idc": [], "a": [] }, "NOK": {"idc": [], "reason": []}, "slotId": [], "finalRating": {"admin": {"individual_success":{ "EST":[], "GT":[] }, "rating": [], "comment": []}, "crowd": {"individual_success": {"EST":[], "GT":[]}, "rating": [], "comment":[],"reason": []}} }) for el in cars_in_shds[cur_job]['workerId'])
    
        # Add slotId
        for count, cur_worker in enumerate(cars_in_shds[cur_job]['workerId']):
            worker_rating[cur_job][cur_worker]["slotId"] = cars_in_shds[cur_job]["slotId"][count]
    
        # Clear Outlier: axis_too_short
        acqui_idc = cars_in_shds[cur_job]["removed"]["axis_too_short"]["idc"]            
        acqui_idc = [i for i in acqui_idc if i]  # Remove empty elements
    
        nmb_ = 0
        for count, cur_worker in enumerate(cars_in_shds[cur_job]['workerId']):
            if cur_worker in set(cars_in_shds[cur_job]["removed"]["axis_too_short"]["workerId"]):
                idx = cars_in_shds[cur_job]["removed"]["axis_too_short"]["workerId"].index(cur_worker)
                NOK_idc = acqui_idc[idx]  # Indices of the acquisitions by cur_worker
                # Update rating list
                worker_rating[cur_job][cur_worker]["NOK"]["idc"].extend(NOK_idc)
                for i in range(len(NOK_idc)):
                    worker_rating[cur_job][cur_worker]["NOK"]["reason"].append("Axis too short")
                    nmb_ +=1
                    
        # print("clear outlier = {}".format(nmb_))
        
        # Clear Outlier: DBSCAN
        fst_noise_mask = cars_in_shds[cur_job]["dbscan"]["noise_mask"]
        snd_noise_mask = cars_in_shds[cur_job]["dbscanWEAK"]["noise_mask"]
        
        idc_true = np.array([i for i, x in enumerate(fst_noise_mask) if x])  # indices where fst_noise_mask is true 
        
        # idc_true[~snd_noise_mask] # idc to check by admin
        outlier_idc = idc_true[snd_noise_mask] # clear outlier idc
        
        nmb_ = 0
        for count, index_list in enumerate(cars_in_shds[cur_job]["job_idc"] ):           
            cur_worker = cars_in_shds[cur_job]['workerId'][count]
            
            # temp = set(index_list) # Indices list of matching element from other list
            cur_outlier_idc = [i for i, val in enumerate(index_list) if val in outlier_idc]   #https://www.geeksforgeeks.org/python-indices-list-of-matching-element-from-other-list/
            
            cur_outlier_count = len(cur_outlier_idc)            
            # np.array(outlier_idc)[cur_outlier_idc]
            # res_noise_mask = [outlier_idc[i] for i in index_list]  # Noise mask for index list of current worker
            # outlier_count = sum(res_noise_mask) # Count "true" occurencies -> outlier
            if cur_outlier_count > 0:
                # [i for i, x in enumerate(res_noise_mask) if x]
                # outlier_idc = list(np.where(res_noise_mask)[0]) # Idc of outlier
                
                remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][count]
                
                # NOK_idc = [remaining_idc[i] for i in outlier_idc]  # NOK_idc
                NOK_idc = np.array(remaining_idc)[cur_outlier_idc]                
                
                # Update rating list
                worker_rating[cur_job][cur_worker]["NOK"]["idc"].extend(NOK_idc)
                for i in range(len(NOK_idc)):
                    worker_rating[cur_job][cur_worker]["NOK"]["reason"].append("2nd DBSCAN clear outlier")
                    nmb_ += 1
        # print("DBSCAN clear outlier = {}".format(nmb_))
        # Clear Outlier: axis len diff too big
        nmb_ = 0
        acqui_idc_removed = cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["idc"]
        for cur_acqui_idx in acqui_idc_removed:
            for job_idx, acqui_idc in enumerate(cars_in_shds[cur_job]["job_idc"]):  # find worker id for "cur_acqui_idx"
                try:
                    remaining_idx = acqui_idc.index(cur_acqui_idx)      # Except happens here, when value is not part of the list
                    remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][job_idx]
                    
                    # Update rating list
                    cur_worker = cars_in_shds[cur_job]['workerId'][job_idx]          
                    NOK_idx = remaining_idc[remaining_idx]
                    worker_rating[cur_job][cur_worker]["NOK"]["idc"].append(NOK_idx)
                    worker_rating[cur_job][cur_worker]["NOK"]["reason"].append("Difference to integrated mean axis len too big")
                    nmb_ += 1
                    break
                except:
                    continue
        #print("Axis len too big clear outlier = {}".format(nmb_))
                    #print("Skip to next worker, acqui idx not part of current workers job")
        
        # Look for cluster idc in Input4Interfaces, if there is no matching cluster idx -> the whole cluster is OK      
        list1 = list(range(len(cars_in_shds[cur_job]["kmeans"]["cluster_idc"]))) # Original cluster
        list2 = cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"]    # Cluster which failed ellipse criteria -> to be checked by admin/crowd
        cluster_idc_ok = list(set(list1) - set(list2))    # Cluster idc which are completly OK
        # Complete cluster is ok
        nmb_=0
        if cluster_idc_ok:
            # Not empty: Remaining cluster are ok
            #print("cluster idc do not match completely")            
            for cluster_idx in cluster_idc_ok:
                for cur_acqui_idx in cars_in_shds[cur_job]["kmeans"]["cluster_idc"][cluster_idx]:
                    for job_idx, acqui_idc in enumerate(cars_in_shds[cur_job]["job_idc"]):  # find worker id for "cur_acqui_idx"
                        try:
                            remaining_idx = acqui_idc.index(cur_acqui_idx)      # Except happens here, when value is not part of the list
                            remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][job_idx]
                            
                            # Update rating list
                            cur_worker = cars_in_shds[cur_job]['workerId'][job_idx]          
                            OK_idc = remaining_idc[remaining_idx]
                            worker_rating[cur_job][cur_worker]["OK"]["idc"].append(OK_idc)
                            nmb_ += 1 #len(OK_idc)
                            break
                        except:
                            continue
        #print("OK Cluster = {}".format(nmb_))
                            #print("Skip to next worker, acqui idx not part of current workers job")
        # Check rest of acquisitions of the clusters (list2) which are "OK", useless in the moment
        nmb_ = 0
        for idx, cluster_idx in enumerate(list2):
            acqui_idc_all = cars_in_shds[cur_job]["kmeans"]["cluster_idc"][cluster_idx]
            acqui_idc_to_check = cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["acqui_idc"][idx]
            
            # Indices list of matching element from other list
            ok_acqui_idc = [i for i, val in enumerate(acqui_idc_all) if val not in acqui_idc_to_check]                        
            ok_acqui_idc = np.array(cars_in_shds[cur_job]["kmeans"]["cluster_idc"][cluster_idx])[ok_acqui_idc]
            
            for cur_acqui_idx in ok_acqui_idc:
                for job_idx, acqui_idc in enumerate(cars_in_shds[cur_job]["job_idc"]):                            
                    try:
                        remaining_idx = acqui_idc.index(cur_acqui_idx)                  ## WIESO remaining_idx = 8 wenn cur_acqui_idx = 0 für cur_job = 1
                        remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][job_idx]
                        
                        cur_worker = cars_in_shds[cur_job]['workerId'][job_idx] 
                        OK_idc = remaining_idc[remaining_idx]
                        worker_rating[cur_job][cur_worker]["OK"]["idc"].append(OK_idc)
                        nmb_ += 1   
                        break
                    except:
                        continue
                        #print("Skip to next worker, acqui idx not part of current workers job")
        #print("OK rest Cluster = {}".format(nmb_))
        ## Rated by admin/crowd interface
        
        if method == "admin":
            # Ell rated
            try:
                len(ell_rating_result[cur_job]) == len(cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"])
            except:
                print("Error: len of input/output differs")
            
            nmbr_ell_added = 0
            for idx, cluster_idx in enumerate(cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"]):
                for count, acqui in enumerate(ell_rating_result[cur_job][cluster_idx]):
                    rating = acqui['finalRating']
                    workerId = acqui['workerId']
                    reason = acqui['reason']
                    
                    cur_acqui_idx = cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["acqui_idc"][idx][count]
                    for job_idx, acqui_idc in enumerate(cars_in_shds[cur_job]["job_idc"]):                        
                        try:
                            remaining_idx = acqui_idc.index(cur_acqui_idx)      # Except happens here, when value is not part of the list
                            remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][job_idx]
                            # Update rating list
                            cur_worker = cars_in_shds[cur_job]['workerId'][job_idx]    
                            if workerId[0] != cur_worker:
                                raise Exception (" Worker id do not match ")                           
                            idx_result = remaining_idc[remaining_idx]
                            
                            if rating == 'OK':                            
                                worker_rating[cur_job][cur_worker]["OK"]["idc"].append(idx_result)
                            if rating == 'NOK':
                                worker_rating[cur_job][cur_worker]["NOK"]["idc"].append(idx_result)    
                                worker_rating[cur_job][cur_worker]["NOK"]["reason"].append(reason) 
                            #print("ell -> entry added, job {}, cluster {}, worker {} idx {}".format(cur_job, cluster_idx, workerId, idx_result))
                            nmbr_ell_added += 1
                            break
                        except:
                            continue
                            #print("Skip to next worker, acqui idx not part of current workers job")
            #print(nmbr_ell_added)
            # Db rated
            # cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"] # input
            # dbRatingResult # result
            try:
                len(db_rating_result[cur_job]) == len(cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"])
            except:
                print("Error: len of input/output differs")
            
            fst_noise_mask = cars_in_shds[cur_job]["dbscan"]["noise_mask"]
            snd_noise_mask = cars_in_shds[cur_job]["dbscanWEAK"]["noise_mask"]
            
            idc_true = np.array([i for i, x in enumerate(fst_noise_mask) if x])
            idc_admin_check = idc_true[~snd_noise_mask]
            nmbr_db_added = 0
            for cluster_idx, acqui_idc in enumerate(cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"]):
                for acqui_idx1, cur_acqui_idx in enumerate(acqui_idc):
                    
                    cur_acqui = db_rating_result[cur_job][cluster_idx][acqui_idx1]
                    idx_ = idc_true[cur_acqui_idx]                
                    rating = cur_acqui['finalRating']
                    workerId = cur_acqui['workerId']
                    reason = cur_acqui['reason']
                    for job_idx, index_list in enumerate(cars_in_shds[cur_job]["job_idc"]): 
                        try:
                            remaining_idx = index_list.index(idx_)      # Except happens here, when value is not part of the list              
                            remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][job_idx]
                            
                            cur_worker = cars_in_shds[cur_job]['workerId'][job_idx]     
                            if workerId[0] != cur_worker:
                                raise Exception (" Worker id do not match ")           
                            idx_result = remaining_idc[remaining_idx]
                            
                            if rating == 'OK':                            
                                worker_rating[cur_job][cur_worker]["OK"]["idc"].append(idx_result)
                            if rating == 'NOK':
                                worker_rating[cur_job][cur_worker]["NOK"]["idc"].append(idx_result)     
                                worker_rating[cur_job][cur_worker]["NOK"]["reason"].append(reason)                      
                            #print("db -> entry added, job {}, cluster {}, worker {} idx {}".format(cur_job, cluster_idx, workerId, idx_result))
                            nmbr_db_added += 1
                            break
                        except:
                            continue
                            #print("Skip to next worker, acqui idx not part of current workers job")
            #print(nmbr_db_added)
                         
        if method == "crowd":            
            #step = 5    # step 1, 3, 5
            
            # Ell rated
            try:
                len(ell_rating_result[cur_job]) == len(cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"])
            except:
                print("Error: len of input/output differs")
            
            nmbr_ell_added = 0
            for idx, cluster_idx in enumerate(cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"]):
                ratings = []; workerIds = []; coords = []
                for count, acqui in enumerate(ell_rating_result[cur_job][cluster_idx]):
                    ratings.append(acqui['finalRating']["steps"][step])
                    workerIds.append(acqui['workerId'])
                    coords.append(acqui["acquiCoord"])
                
                cur_acqui_idc = cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["acqui_idc"][idx]
                
                for cur_acqui_idx in cur_acqui_idc:                
                    # Test first entry
                    for job_idx, index_list in enumerate(cars_in_shds[cur_job]["job_idc"]): 
                        try:
                            remaining_idx = index_list.index(cur_acqui_idx)
                            remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][job_idx]
                            
                            indices = [i for i, x in enumerate(workerIds) if x == cars_in_shds[cur_job]['workerId'][job_idx]]   # Except happens here, when value is not part of the list        
                            if not indices:
                                continue
                                #raise Exception(" Worker id do not match ")   ->  test next job idc
                            match = False
                            for c in indices:   # Search for corresponding data
                                workerId = workerIds[c]
                                rating = ratings[c]
                                coord = coords[c]

                                x = "{:.2f}".format(cars_in_shds[cur_job]["start"]["x"][cur_acqui_idx])
                                y = "{:.2f}".format(cars_in_shds[cur_job]["start"]["y"][cur_acqui_idx])
                                z = "{:.2f}".format(cars_in_shds[cur_job]["end"]["x"][cur_acqui_idx])
                                w = "{:.2f}".format(cars_in_shds[cur_job]["end"]["y"][cur_acqui_idx])
                                if x == "{:.2f}".format(coord["x"]) and y == "{:.2f}".format(coord["y"]) and z == "{:.2f}".format(coord["z"]) and w == "{:.2f}".format(coord["w"]): # should be replaced by unique id system. to clearly identify acquisition
                                    idx_result = remaining_idc[remaining_idx]

                                    if rating == "OK":                            
                                        worker_rating[cur_job][workerId]["OK"]["idc"].append(idx_result)
                                    if rating == "NOK":
                                        worker_rating[cur_job][workerId]["NOK"]["idc"].append(idx_result)     
                                        worker_rating[cur_job][workerId]["NOK"]["reason"].append("Crowd majority voted NOK")
                                    #print("ell -> entry added, job {}, cluster {}, worker {} idx {}".format(cur_job, cluster_idx, workerId, idx_result))
                                    nmbr_ell_added += 1     
                                    match = True                      
                                    break
                            if not match:                            
                                raise Exception ("No match found")
                            break
                        except:
                            #raise Exception ("Error looking for corresponding worker")
                            continue
            #print(nmbr_ell_added)  
            # Db rated 
            # cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"] # input
            # dbRatingResult # result
            try:
                len(db_rating_result[cur_job]) == len(cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"])
            except:
                print("Error: len of input/output differs")
            
            fst_noise_mask = cars_in_shds[cur_job]["dbscan"]["noise_mask"]
            snd_noise_mask = cars_in_shds[cur_job]["dbscanWEAK"]["noise_mask"]
            
            idc_true = np.array([i for i, x in enumerate(fst_noise_mask) if x])
            idc_admin_check = idc_true[~snd_noise_mask]
            nmbr_db_added = 0
            for cluster_idx, acqui_idc in enumerate(cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"]):
                
                idx_ = idc_true[np.array(acqui_idc)]
                
                workerIds = []; ratings = []; coords = []
                for i, cur_acqui in enumerate(db_rating_result[cur_job][cluster_idx]):
                    workerIds.append(cur_acqui["workerId"])
                    ratings.append(cur_acqui["finalRating"]["steps"][step])
                    coords.append(cur_acqui["acquiCoord"])

                # Test first entry
                for idx in idx_:
                    for job_idx, index_list in enumerate(cars_in_shds[cur_job]["job_idc"]): 
                        try:
                            remaining_idx = index_list.index(idx)
                            remaining_idc = cars_in_shds[cur_job]["job_idc_remaining"][job_idx]
                            
                            indices = [i for i, x in enumerate(workerIds) if x == cars_in_shds[cur_job]['workerId'][job_idx]]   # Except happens here, when value is not part of the list        
                            if not indices:
                                continue
                                #raise Exception(" Worker id do not match ")   ->  test next job idc
                            
                            match = False
                            for c in indices:   # Search for corresponding data
                                workerId = workerIds[c]
                                rating = ratings[c]
                                coord = coords[c]

                                x = "{:.2f}".format(cars_in_shds[cur_job]["start"]["x"][idx])
                                y = "{:.2f}".format(cars_in_shds[cur_job]["start"]["y"][idx])
                                z = "{:.2f}".format(cars_in_shds[cur_job]["end"]["x"][idx])
                                w = "{:.2f}".format(cars_in_shds[cur_job]["end"]["y"][idx])
                                if x == "{:.2f}".format(coord["x"]) and y == "{:.2f}".format(coord["y"]) and z == "{:.2f}".format(coord["z"]) and w == "{:.2f}".format(coord["w"]): # to clearly identify acquisition
                                    idx_result = remaining_idc[remaining_idx]

                                    if rating == "OK":                            
                                        worker_rating[cur_job][workerId]["OK"]["idc"].append(idx_result)
                                    if rating == "NOK":
                                        worker_rating[cur_job][workerId]["NOK"]["idc"].append(idx_result)     
                                        worker_rating[cur_job][workerId]["NOK"]["reason"].append("Crowd majority voted NOK")
                                    # print("db -> entry added, job {}, cluster {}, worker {} idx {}".format(cur_job, cluster_idx, workerId, idx_result))
                                    nmbr_db_added += 1
                                    match = True                                  
                                    break
                            if not match:
                                raise Exception ("No match found" )
                            break
                        except:
                            # raise Exception ("Error looking for corresponding worker")
                            continue
            # print(nmbr_db_added)

    return worker_rating


def plot_pie_chart(filepath_acqui_n_25, filepath_acqui_n_50, filepath_quest):
    """ 
    Plot countries in pie chart
    
    Parameters:
    ----------
        filepath: string
    """
    # Load file
    paths = [filepath_acqui_n_25, filepath_acqui_n_50, filepath_quest]
    
    countries = []; values = []
    for path in paths:
        country_number = []
        with open (path, "r") as f:
            try:
                for _ in range(1):  # Skip header
                    next(f)      
                for line in f:
                    line = re.sub("\n", "", line)
                    line = line.split("\t")
                    
                    if line[0] in countries:
                        idx = countries.index(line[0])
                        values[idx] += int(line[1])
                    else:
                        countries.append(line[0])
                        values.append(int(line[1]))
            except:
                print("Error: loading country file -> {}".format(path))
    
    # country_number = np.array()
    # country_number = []
    # with open (filepath, "r") as f:
    #    countries = []; values = []
    #    for _ in range(1):  # Skip header
    #        next(f)      
    #    for line in f:
    #        line = re.sub("\n", "", line)
    #        line = line.split("\t")
    #        countries.append(line[0])
    #        values.append(int(line[1]))
    
    # Full data set
    df = pd.DataFrame(
        data={
            "country": countries, "value": values
        }
    ).sort_values("value", ascending=False)
    
    # Top 4 countries
    df2 = df[:4].copy()
    
    # New category "others", group small positions
    new_row = pd.DataFrame(
        data={
            "country": ["Sonstige"],
            "value": [df["value"][4:].sum()]
        }
    )
    
    # Combine Top 5 with "Others"
    df2 = pd.concat([df2, new_row])
    
    pctdistance = [.5, .8, .8, .8, .5]
    explode = (0.02, 0.03, 0.03, 0.03, 0.03)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    df2.plot(kind="pie", y="value", labels=df2["country"], ax=ax, autopct='%1.1f%%', pctdistance=0.83, explode=explode)
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    ax.axis('equal')     
    ax.set_title("Top Herkunftsländer")
    ax.set_ylabel("")
    ax.legend(loc='lower left', bbox_to_anchor=(0.8, 0.7), fancybox=True, shadow=True)
    
    path = 'plotData/Herkunft/'    
    fname = 'herkunft.png'           
    path += fname            
    fig.savefig(path, format='png', dpi=700)
    
    plt.show()
    
    plt.clf()
    plt.close()


def interpolated_intercepts(x, y1, y2):
    # https://stackoverflow.com/questions/42464334/find-the-intersection-of-two-curves-given-by-x-y-data-with-high-precision-in, 28.02.2021
    """Find the intercepts of two curves, given by the same x data"""
    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.
        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)
        Returns: the intercept, in (x,y) format
        """
        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            x = Dx / D
            y = Dy / D
            return x,y
        L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = line([point3[0], point3[1]], [point4[0], point4[1]])
        R = intersection(L1, L2)
        return R
    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xcs = []
    ycs = []
    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)


def calc_final_rating(cars_in_shds, worker_rating, db_rating_result, ell_rating_result, method, config, **kwargs):
    """
    Calculate final rating
    
    Parameters:
    ----------
        cars_in_shds: dict
        
        worker_rating: dict
        
        dbRatingResult: dict
        
        ellRatingResult: dict
        
        method: string
            "admin" or "crowd"
        
        config
    
    
    Returns:
    ----------
        OK_int_cluster_idc:
        
        approvedCluster_weak_idc:
        
        worker_rating: dict
    
    """
    print("\n--------------------------------------------------------")
    print("Calculate final rating with results from {}interface".format(method))
    print("--------------------------------------------------------")
    
    step = kwargs.get("step", None)
    
    # Calculate final rating
    clusterCount_final = {}
    approvedCluster_weak_idc = {}
    OK_int_cluster_idc = {}

    for cur_job in range(config["jobs"]["number_of_jobs"]):
        #if cur_job == 1:
        #    print("xd")
        OK_int_cluster_idc[cur_job] = copy.deepcopy(cars_in_shds[cur_job]["kmeans"]["OK_int_cluster_idc"])     # Integrated cluster idc which have "OK" rating
        
        clusterCount_final[cur_job] = []
        # Cluster count after integration
        clusterCount_integration = len(cars_in_shds[cur_job]["kmeans"]["cluster_idc"])
        
        # Cluster count of uncertain clusters before checking via admininterface/crowdinterface
        clusterCount_preCheck = len(cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"])
        
        # Cluster approved by interface
        discardedCluster = []
        approvedCluster_ell_idc = []
        clusterCount_postCheck = 0
        for clusterIdx in ell_rating_result[cur_job]:
            cluster = ell_rating_result[cur_job][clusterIdx]
            approved = False
            ok = False
            for acqui in cluster:   # When any acquisition of uncertain cluster receives "OK" rating -> cluster is approved
                if method == "admin":
                    if acqui["finalRating"] == "OK":
                        ok = True
                elif method == "crowd":
                    if acqui["finalRating"]["steps"][step] == "OK":
                        ok = True
                
                if ok: 
                    clusterCount_postCheck += 1    
                    approvedCluster_ell_idc.append(clusterIdx)
                    approved = True                
                    break
            
            if approved is False:
                discardedCluster.append(clusterIdx)
        
        # Check if all clusters are approved
        diff = clusterCount_preCheck - clusterCount_postCheck
        if diff > 0:
            clusterCount_final[cur_job] = clusterCount_integration - diff
        else:
            clusterCount_final[cur_job] = clusterCount_integration
        
        # Cluster count of uncertain clusters via 2nd DBSCAN
        clusterCount_postCheck = 0
        approvedCluster_weak_idc[cur_job] = []
        for clusterIdx in db_rating_result[cur_job]:
            cluster = db_rating_result[cur_job][clusterIdx]
            ok = False
            for acqui in cluster:   # When any acquisition of uncertain cluster receives "OK" rating -> cluster is in approval process
                if method == "admin":
                    if acqui["finalRating"] == "OK":
                        ok = True
                elif method == "crowd":
                    if acqui["finalRating"]["steps"][step] == "OK":
                        ok = True

                if ok:
                    # Calculate distance to all integrated cluster centers                    
                    # Create idx list [0,1,2, ... , cluster count]
                    integrated_cluster_idc = [*range(len(cars_in_shds[cur_job]["kmeans"]["mean"]["x"]))]
                    # Remove discarded cluster idc from integrated cluster center list
                    integrated_cluster_idc = [ele for ele in integrated_cluster_idc if ele not in discardedCluster]
                    
                    # Get mean cluster center
                    integrated_mean_x = np.array(cars_in_shds[cur_job]["kmeans"]["mean"]["x"])[integrated_cluster_idc]
                    integrated_mean_y = np.array(cars_in_shds[cur_job]["kmeans"]["mean"]["y"])[integrated_cluster_idc]
                    
                    if method == "admin":                    
                        dbMean_x = acqui["clusterCoord"][0]
                        dbMean_y = acqui["clusterCoord"][1]   
                    if method == "crowd":
                        dbMean_x = acqui["clusterCoord"]["xMean"]
                        dbMean_y = acqui["clusterCoord"]["yMean"]           
                    
                    max_center_deviation = 15  # config["integration"]["max_distance_correspondence"]#15   # 15 * 0.1m = 1.5m
                    sameCluster = False
                    for i in range(len(integrated_mean_x)):
                        # Euclid dist from weak dbscan center to integrated center     
                        euclDist = math.dist([ integrated_mean_x[i], integrated_mean_y[i]], [dbMean_x, dbMean_y]) 
                        # print("euclDist", euclDist)
                        # If same center (deviation minimal)
                        if euclDist < max_center_deviation:   # config["integration"]["max_distance_correspondence"]:
                            sameCluster = True                                                  
                            break
                        else:                            
                            # Cluster differs -> Cluster found via 2nd DBSCAN is new cluster
                            continue        
                    if sameCluster is False:
                        # print("2nd DBSCAN detected confirmed cluster ({}), x={:.2f}, y={:.2f}".format(clusterIdx, dbMean_x, dbMean_y))
                        clusterCount_final[cur_job] +=1   
                        
                        # For quality parameter estimation -> recall, precision, f1-score
                        approvedCluster_weak_idc[cur_job].append(clusterIdx)
                    #else:
                    #    print("Cluster detected with 2nd DBSCAN was already detected with integration")
                    break
        
        # Integrated "OK" cluster idc -> via DBSCAN, Dist, ellipse route    -> for precision, recall, f1-score needed later
        # cars_in_shds[cur_job]["kmeans"]["OK_int_cluster_idc"].extend(approvedCluster_ell_idc) # Integrated cluster idc which have "OK" rating

        OK_int_cluster_idc[cur_job].extend(approvedCluster_ell_idc)
        # -----------------------------------------------------------------------------
        #
        # Compute success rating for each worker -> plot
        #
        # -----------------------------------------------------------------------------
        
        GT_cluster_count = len(cars_in_shds[cur_job]["gt"]["start"]["x"])
        EST_cluster_count = clusterCount_final[cur_job]
        
        # If cluster count differs: GT <-> Estimation -> Plot missing cluster   -> False negative !!!!!!!!!! TO DOOOOOOOOOOOOOO
        if EST_cluster_count != GT_cluster_count:
            print("Estimated Cluster Count differs from GT: GT = {}, Estimated = {}, diff = {}".format(GT_cluster_count, EST_cluster_count, np.abs(GT_cluster_count-EST_cluster_count)))
        else:
            print("Estimated Cluster Count matches GT: GT = Estimated = {}".format(GT_cluster_count))
            
        # Calculate overall success rating for each worker
        for idx, worker in enumerate(worker_rating[cur_job]):
            worker = worker_rating[cur_job][worker]
            
            success = len(worker["OK"]["idc"]) / EST_cluster_count
            worker["finalRating"][method]["individual_success"]["EST"] = success
            
            success_GT = len(worker["OK"]["idc"]) / GT_cluster_count
            worker["finalRating"][method]["individual_success"]["GT"] = success_GT
            
            #if success > config.success_threshold:
            #    worker["finalRating"]["admin"]["rating"] = "OK"
            #    worker["finalRating"]["admin"]["comment"] = "Good Job! You detected more than 70% of the visible cars"
            #    successfulWorker += 1
            #else:
            #    worker["finalRating"]["admin"]["rating"] = "NOK"
            #    worker["finalRating"]["admin"]["comment"] = "Less than 30% of the cars were precisely detected"

        # Plot: "OK" rated acquisitions by worker 
        savePlot = True       
        ok_count = []
        nok_count = []
        success_ratio_est = []
        success_ratio_gt = []
        for worker in worker_rating[cur_job]:
            worker = worker_rating[cur_job][worker]
            
            ok_count.append(len(worker["OK"]["idc"]))
            nok_count.append(len(worker["NOK"]["idc"]))
            
            success_ratio_est.append(len(worker["OK"]["idc"]) / EST_cluster_count * 100)
            success_ratio_gt.append(len(worker["OK"]["idc"]) / GT_cluster_count * 100)

        #fig, (ax1, ax2) = plt.subplots(1, 2)    #, bins=int(GT_cluster_count/2)
        fig, ax2 = plt.subplots()
        #n1, bins, patches = ax1.hist(ok_count, density=True, bins=int(GT_cluster_count/2), label="OK ratings", color="navajowhite", edgecolor="k", alpha=0.65)
        #ax1.axvline(np.mean(ok_count), color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=1)   # Mean vertical line
        #ax1.axvline(np.median(ok_count), color="y", linestyle="dashed", alpha=0.65, linewidth=1)   # median
        ##ax1.axvline(config.threshold_payment, color="orange", linestyle="dashed", alpha=0.65, linewidth=1)
        #fig.suptitle("Success Distribution (each Acquisition):\nJob {}, Cluster Count: GT = {}, Estimated = {}\nVerification with {}interface".format(cur_job, GT_cluster_count, EST_cluster_count, method))
        ##ax1.set_yticks(np.arange(0, max(n1))) # 1 tick more than needed to place legend
        #ax1.set_title("Absolute")
        #ax1.set_xlabel("# of \"OK\" rated Acquisitions")
        #ax1.set_ylabel("Relative Probability")        
        ## Add text box
        #props = {'facecolor': 'lightseagreen', 'alpha': 0.6, 'pad': 5}
        ##ax1.text(0.5, max(n) - 0.5, "{:.2F}".format(np.mean(ok_count)), bbox=props)
        #legend_elements = [ Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=1, alpha=0.65),
        #                    Line2D([0], [0], color="y", linestyle="dashed", linewidth=1, alpha=0.65)]
        #ax1.legend(labels=("Mean = {:.2f}".format(np.mean(ok_count)),"Median = {:.2f}".format(np.median(ok_count)),), handles=legend_elements, shadow=True, loc="upper left", handlelength=1.5)
        
        # Plot: "OK" ratio by worker
        x1, bins1, p1 = ax2.hist(success_ratio_est, len(success_ratio_est), alpha=0.5, label="Berechnete Clusteranzahl={}".format(EST_cluster_count), range=[0, 100])#, histtype='step', stacked=True, fill=False)
        x2, bins2, p2 = ax2.hist(success_ratio_gt, len(success_ratio_gt), alpha=0.5, label="Referenz Clusteranzahl={}".format(GT_cluster_count), range=[0, 100])#, histtype='step', stacked=True, fill=False)
        
        #n2, bins, patches = ax2.hist(success_ratio_est, bins=int(GT_cluster_count/2), label="OK ratings", color="navajowhite", edgecolor="k", alpha=0.65)
        #ax2.axvline(np.mean(success_ratio_est), color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=1)   # mean
        #ax2.axvline(np.median(success_ratio_est), color="y", linestyle="dashed", alpha=0.65, linewidth=1)   # median
        
        #ax2.set_yticks(np.append(np.arange(0, max(n2)+1, step=1),max(n2)+2))
        ax2.set_title("Histogramm der Erfolgsquote (basierend auf berechneter Clusteranzahl und auf der Referenz)")
        ax2.set_xlabel("Erfolgsquote einzelner Crowdworker [%]")
        ax2.set_ylabel("Anzahl an Crowdworkern")
        #legend_elements = [ Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=1, alpha=0.65),
        #                    Line2D([0], [0], color="y", linestyle="dashed", linewidth=1, alpha=0.65)]
        #ax2.legend(labels=("Mean = {:.2f} %".format(np.mean(success_ratio_est))
        #                    ,"Median = {:.2f} %".format(np.median(success_ratio_est)),), 
        #            handles=legend_elements, shadow=True, loc="upper left", handlelength=1.5)
        ax2.legend()
        
        fig.tight_layout()                
        fig1 = plt.gcf()  # Needed because after plt.show() new fig is created and savefig would safe empty fig
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        if savePlot:
            path = 'figures/{}/quality_parameters/'.format(cur_job+1)
            create_dir_if_not_exist(path) 
            
            fname = 'job_{}_{}_success_distribution_acquis.png'.format(cur_job+1, method)
            path += fname
            plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
            plt.close("all")  
        # plt.show()

    ## Combined data -> Calculate 
    success = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        for idx, worker in enumerate(worker_rating[cur_job]):
            if  worker_rating[cur_job][worker]["finalRating"][method]["individual_success"]["EST"] * 100 > 100:
                worker_rating[cur_job][worker]["finalRating"][method]["individual_success"]["EST"] = 1  # cap, possible when workerdetects same car twice, very rare
                # print(worker)
            success.append(worker_rating[cur_job][worker]["finalRating"][method]["individual_success"]["EST"] * 100)
    mean_success = np.mean(success)   # success = indiviudal score / number estimated cluster
    median_success = np.median(success)
    
    success_gt = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        for idx, worker in enumerate(worker_rating[cur_job]):
            if  worker_rating[cur_job][worker]["finalRating"][method]["individual_success"]["GT"] * 100 > 100:
                worker_rating[cur_job][worker]["finalRating"][method]["individual_success"]["GT"] = 1  # cap, possible when workerdetects same car twice, very rare
                # print(worker)
            success_gt.append(worker_rating[cur_job][worker]["finalRating"][method]["individual_success"]["GT"] * 100)
    mean_success_gt = np.mean(success_gt)   # success = indiviudal score / number estimated cluster
    median_success_gt = np.median(success_gt)
    
    # True Success with GT cluster count
    
    # Plot hist
    # Plot: "OK" ratio by worker, density=True, cumulative=True
    #fig, ax = plt.subplots()
    #n, bins, patches = ax.hist(success, bins= len(success), label="OK ratings", color="navajowhite", edgecolor="k", alpha=0.65)
    #ax.axvline(mean_success, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=1)   # mean
    #ax.axvline(median_success, color="y", linestyle="dashed", alpha=0.65, linewidth=1)   # median
    #
    #ax.set_yticks(np.append(np.arange(0, max(n)+1, step=1), max(n)+1)) # 1 tick more than needed to place legend
    #ax.set_title("Rating Distribution:\nVerification with {}interface".format(method))
    #ax.set_xlabel("% of \"OK\" rated Acquisitions")
    #ax.set_ylabel("# of Workers")
    #legend_elements = [ Line2D([0], [0], color="lightseagreen", linestyle="dashed", linewidth=1, alpha=0.65),
    #                    Line2D([0], [0], color="y", linestyle="dashed", linewidth=1, alpha=0.65)]
    #ax.legend(labels=("Mean = {:.2f} %".format(mean_success),"Median = {:.2f} %".format(median_success),), handles=legend_elements, shadow=True, loc="upper left", handlelength=1.5)
    #
    #fig.tight_layout()
    #
    #fig1 = plt.gcf()  # Needed because after plt.show() new fig is created and savefig would safe empty fig
    #
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    #figManager = plt.get_current_fig_manager() # for fullscreen
    ##figManager.window.state("zoomed")
    #figManager.full_screen_toggle()
    
    #if savePlot:
    #    path = 'figures/overall_quality_parameters'
    #    create_dir_if_not_exist(path) 
    #    
    #    fname = '/{}_rating_distribution_acquisitions.png'.format(method)
    #    path += fname
    #    plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
    #    plt.close("all")  
    #
    
    # Cumulative Cost Graph
    # Plot: "OK" ratio by worker    ,density=True, cumulative =True
    try:
        # Same payment & fee for acqui & question campaign
        paymentPerTask = config["campaign-general"]["paymentPerTask"]
        campaignFee = 0.1 # 10 % of complete campaign 
        availablePositions = config["jobs"]["number_of_jobs"] * config["jobs"]["number_of_acquisitions"]
        ymax = 1 * availablePositions * paymentPerTask * (1 + campaignFee)
        ymin = 0
        
        ## Erfolgsquote: Vergleiche Histogramme basierend auf berechneter Clusteranzahl und der Referenz Clusteranzahl
        fig, ax = plt.subplots()
        binCount_est = len(success)
        binCount_gt = len(success_gt)
        #weights_est = np.ones_like(success)/len(success)
        #weights_gt = np.ones_like(success_gt)/len(success_gt)
        x1, bins1, p1 = ax.hist(success, binCount_est, alpha=0.5, label=r'Crowdworker$_{Integriert}$', color="#1f77b4", range=[0, 100])#, weights=weights_est)#, histtype='step', stacked=True, fill=False)
        x2, bins2, p2 = ax.hist(success_gt, binCount_gt, alpha=0.5, label=r'Crowdworker$_{Referenz}$', color="#2ca02c", range=[0, 100])#, weights=weights_gt)#, histtype='step', stacked=True, fill=False)
        ax.set_xlabel("Erfolgsquote [%]")
        ax.set_ylabel("Absolute Häufigkeit")
        ax.legend(fancybox=True, shadow=True, handlelength=1.5)
        
        fig.tight_layout()
        fig1 = plt.gcf()
        savePlot = True
        if savePlot:
            path = 'figures/overall_quality_parameters'
            create_dir_if_not_exist(path)             
            fname = '/{}_erfolgsquote_alle_streifen.png'.format(method)
            path += fname
            plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
            plt.close("all")       
        
        # Save plotted data to file
        __path = "plotted_data_textformat/allgemeine_daten/"
        create_dir_if_not_exist(__path)
        __fname = "erfolgsquoten_{}.txt".format(method)
        with open(__path + __fname,"w") as f:
            f.write("Erfolgsquoten, der Crowdworker aus Erfassung und Überprüfung über aller Streifen basierend auf integrierter Fahrzeuganzahl\n")
            for ele in success:
                f.write("{},".format(ele))
            
            f.write("\n\nErfolgsquoten, der Crowdworker aus Erfassung und Überprüfung über aller Streifen basierend auf der Fahrzeuganzahl der Referenz\n")
            for ele in success_gt:
                f.write("{},".format(ele))
        
        ## Erfolgsquote + Kostenentwicklung
        fig, ax = plt.subplots(1,1, figsize=(14,14))
        hist, bin_edges = np.histogram(success, bins=len(success), density=True)
                
        hist_2, _ = np.histogram(success, bins=len(success), density=False)
        # cumsum = np.cumsum(hist) / np.cumsum(hist)[-1] # normed cumsum [0, ..., 1]
        # reversed
        cumsum = np.cumsum(hist[::-1])[::-1] / np.cumsum(hist[::-1])[::-1][0]
        cumsum *= ymax  # scale to max price   
        ax.grid(True)
        ax.bar(bin_edges[:-1], hist_2, width=np.diff(bin_edges), color="#1f77b4", alpha=0.5)  # align="edge"
        # ax.axvline(mean_success, color="lightseagreen", linestyle="dashed", alpha=0.8, linewidth=1)   # mean
        # ax.axvline(median_success, color="y", linestyle="dashed", alpha=0.8, linewidth=1)   # median
        
        ax2 = ax.twinx()
        _, number_files = calc_sub_it_numb(config)
        positionsQuestionCampaign = number_files * config["interface_questions"]["it_numb"]
        costQuestionCampaign = positionsQuestionCampaign * paymentPerTask * (1 + campaignFee)
        total_cost = np.insert(cumsum, cumsum.size-1, cumsum[-1]) + costQuestionCampaign
        
        # Check if only 1 intersection exists
        idc = np.where(total_cost <= max(cumsum))[0]
        check_idc = np.array([*range(idc[0], idc[-1]+1)])
        if not np.array_equal(idc, check_idc):
            raise Exception("Intersection plot needs patch, does not yet plot multiple intersections")
        y1=[ymax, ymax]
        y2=[0, 0]

        threshold_payment = bin_edges[idc[0]]
        
        #ax2.fill_between([threshold_payment, 100], y1, y2, where=[True, True], color='lightskyblue', alpha=0.2, interpolate=True, label="Payed crowd")
        print("Payed Crowd, at Critical Point (Blue Area) = {} / {} = {:.2f}% of the crowd gets payed".format(np.sum(hist_2[idc[0]:]), availablePositions, np.sum(hist_2[idc[0]:]) / availablePositions * 100))
        
        ax2.plot(bin_edges, total_cost, color="darkgreen", linestyle="-", markersize=1, label="Weighted cumulative histogram (Total cost)")
        ax2.plot(bin_edges, np.insert(cumsum, cumsum.size-1, cumsum[-1]), color="darkorange", linestyle="-", markersize=1, label="Weighted cumulative histogram (1st Cost)")   
        
        ax2.plot([0, 100], [ymax, ymax], color="k", linestyle="-", markersize=1, label="Total cost surpasses cost of 1st campaign")
        #ax2.plot([bin_edges[idc[0]], bin_edges[idc[0]]], [max(total_cost), 0], color="k", linestyle="--", markersize=1, label="Worker Percentage")
        ax2.plot(threshold_payment, max(cumsum), "o", markerfacecolor="red", markeredgecolor="white", markersize="8" , label='Critical Point')
                
        #xcs, ycs = interpolated_intercepts(bin_edges, total_cost, [max(cumsum)] * len(total_cost))
        #for xc, yc in zip(xcs, ycs):
        #    ax2.plot(xc, yc, 'co', ms=5, label='Intersection of Threshold')
        
        #dx = 0
        #dy = 20
        #x = bin_edges[idc[0]]
        #y = ymax
        #ax2.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')
        #bbox = dict(boxstyle="round", fc="lightskyblue", ec="k", pad=0.2)
        ax2.annotate("{} / {} = {:.2f}% der Crowdworker werden bezahlt".format(np.sum(hist_2[idc[0]:]), availablePositions, np.sum(hist_2[idc[0]:]) / availablePositions * 100),
                     xy=(threshold_payment, ymax + 0.5), xycoords="data", xytext=(threshold_payment, ymax + ymax/3), textcoords="data", size=12, va="center", ha="center", arrowprops=dict(arrowstyle="-|>")) #, bbox=bbox
        
        if availablePositions == 150:
            yTop = 30.0
        elif availablePositions == 300:
            yTop = 60.0
            
        ax2.set_ylim(bottom=0, top=yTop)
        ax2.set_ylabel("Kampagnenkosten")
        #ax2.set_ylim(0, max(total_cost)+5)
        #ax.set_xlim(min(bin_edges), max(bin_edges))    
        
        #n, bins, patches = ax.hist(success, bins=len(success), density=True, label="Histogram", color="navajowhite", edgecolor="k", alpha=0.5)
        #ax.axvline(mean_success, color="lightseagreen", linestyle="dashed", alpha=0.65, linewidth=1)   # mean
        #ax.axvline(median_success, color="y", linestyle="dashed", alpha=0.65, linewidth=1)   # median
        
        #ax_bis = ax.twinx()
        #ax_bis.plot(bins, np.cumsum(n) / np.cumsum(n)[-1], color="darkorange", marker="o", linestyle="-", markersize=1, label="Cumulative Histogram")
        
        #ax_bis.hist(success, bins=len(success), density=True, cumulative=True, label="Cumulative Histogram", color="navajowhite", edgecolor="k", alpha=0.5, histtype="step")      
        
        #ax.set_yticks(np.append(np.arange(0, ymax+1, step=1), ymax+1)) # 1 tick more than needed to place legend
        if method == "admin":
            ax.set_title("Erfolgsquote + Kostenentwicklung:\nÜberprüfung mittels Admin")
        if method == "crowd":
            ax.set_title("Erfolgsquote + Kostenentwicklung:\nÜberprüfung mittels Crowd")
            
        ax.set_xlabel("Erfolgsquote")
        ax.set_ylabel("Absolute Anzahl an Crowdworkern", color="#1f77b4")
        ax.tick_params(axis='y', labelcolor="#1f77b4")
        
        def x_fmt(x, y):
            return "{:.0f}%".format(x)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_fmt))
        
        def y_fmt(x, y):
            return "{:.0f}$".format(x)
        ax2.yaxis.set_major_locator(ticker.AutoLocator())
        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
        
        #def align_yaxis(ax, v, ax2, v2):  # https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin 28.02.2021
        #    """adjust ax2 ylimit so that v2 in ax2 is aligned to v in ax"""
        #    _, y1 = ax.transData.transform((0, v))
        #    _, y2 = ax2.transData.transform((0, v2))
        #    inv = ax2.transData.inverted()
        #    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        #    miny, maxy = ax2.get_ylim()
        #    ax2.set_ylim(miny+dy, maxy+dy)        
        #align_yaxis(ax, 0, ax2, 0)
        
        #ax1.set_ylim(bottom=0, top= 0.7)
        
        #align.yaxes(ax, 0, ax2, 0)#, 0.9)
        
        legend_elements = [Patch(facecolor="#1f77b4", label="Crowdworker"),
                           Line2D([0], [0], color="k", linestyle="-", linewidth=1, alpha=0.8),
                           # Line2D([0], [0], color="k", linestyle="--", linewidth=1, alpha=0.8),
                           # Line2D([0], [0], color="lightseagreen", linestyle="--", linewidth=1, alpha=0.8),
                           # Line2D([0], [0], color="y", linestyle="--", linewidth=1, alpha=0.8),
                           Line2D([0], [0], color="darkgreen", linestyle="-", linewidth=1, alpha=0.65),
                           Line2D([0], [0], color="darkorange", linestyle="-", linewidth=1, alpha=0.65),
                           Line2D([0], [0], color="darkgreen", marker="o", markersize=5, markerfacecolor="red", markeredgecolor="white", alpha=0.65),
                           #Patch(facecolor="lightskyblue", edgecolor="k", label="Payed Crowd", alpha=0.2),
                           ]
        ax.legend(labels=("Crowdworker",
                            "Maximale Kosten der Fahrzeugerfassung (Kampagne 1): {}$".format(ymax),
                            #"Percentage at Intersection",
                            #"Mean = {:.2f} %".format(mean_success),
                            #"Median = {:.2f} %".format(median_success),
                            "Gesamtkosten (Fahrzeugerfassung mit Überprüfung $\widehat{=}$ Kampagne 1 und 2)",  
                            "Kosten der Fahrzeugerfassung (Kampagne 1)",                                                      
                            "Schnittpunkt: ({:.1f}%, {:.1f}$)".format(threshold_payment, ymax),
                            #"Bezahlte Crowdworker am Kritischen Punkt",
                           ), handles=legend_elements, loc='best', fancybox=True, shadow=True, handlelength=1.5)#)loc='upper left', bbox_to_anchor=((1.05, 1)), fancybox=True, shadow=True, handlelength=1.5)
        fig.tight_layout()

        plt.subplots_adjust(bottom=0.086, hspace=0.2, top=0.945, right=0.693, wspace=0.2, left=0.054)
        #plt.subplots_adjust(right=0.76)

        #plt.show()
        fig1 = plt.gcf()  # Needed because after plt.show() new fig is created and savefig would safe empty fig
        #figManager = plt.get_current_fig_manager() # for fullscreen
        ##figManager.window.state("zoomed")
        #figManager.full_screen_toggle()
        
        if savePlot:
            path = 'figures/overall_quality_parameters'
            create_dir_if_not_exist(path) 
            
            fname = '/{}_rating_distribution_acquisitions.png'.format(method)
            path += fname
            plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
            plt.close("all")  
        
        # Save plotted data to file
        __path = "plotted_data_textformat/allgemeine_daten/"
        create_dir_if_not_exist(__path)
        __fname = "kampagnenkosten_{}.txt".format(method)
        with open(__path + __fname, "w") as f:
            f.write("Gesamtkosten = Kosten der Kampagne 1 + 2 basierend auf gewaehlter Erfolgsquote: (Erfolgsquote, Gesamtkosten)\n")
            for __idx, ele in enumerate(list(bin_edges)):
                gesamtkosten = np.array(total_cost)[__idx]
                f.write("({},{}),".format(ele, gesamtkosten))
            
            f.write("\n\nKosten der Fahrzeugerfassung: (Erfolgsquote, Kosten der Kampagne 1)\n")
            for __idx, ele in enumerate(list(bin_edges)):
                kosten = np.insert(cumsum, cumsum.size-1, cumsum[-1])[__idx]            
                f.write("({},{}),".format(ele, kosten))
    except:
        print("Error: plotting cumulative histogram plot failed")
    # Final rating with computed median as threshold
    print("Use success score threshold of critical point for payment = {:.1f}%".format(threshold_payment))
    successfulWorker = 0
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        for idx, worker in enumerate(worker_rating[cur_job]):
            individual_success = worker_rating[cur_job][worker]["finalRating"][method]["individual_success"]["EST"]
            if individual_success > threshold_payment/100:
                worker_rating[cur_job][worker]["finalRating"][method]["rating"] = "OK"
                worker_rating[cur_job][worker]["finalRating"][method]["comment"] = "Good Job!"
                successfulWorker += 1
            else:
                worker_rating[cur_job][worker]["finalRating"][method]["rating"] = "NOK"
                worker_rating[cur_job][worker]["finalRating"][method]["comment"] = "Less than {}% of the cars were precisely detected".format(int(threshold_payment))
    
    total_worker = config["jobs"]["number_of_jobs"] * config["jobs"]["number_of_acquisitions"]
    ratio_successful = successfulWorker / total_worker * 100
    print("\n--------------")
    print("Detected Cluster (Final) = ", clusterCount_final)
    print("Ratio successful = {:.2f}%, successful worker = {}, total worker = {}\n".format(ratio_successful, successfulWorker, total_worker))
    
    ## Plot: Distribution "OK"<->"NOK"
    ok_count_final = {}
    nok_count_final = {}    
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        ok_count_final[cur_job] = 0
        nok_count_final[cur_job] = 0        
        for worker in worker_rating[cur_job]:
            worker = worker_rating[cur_job][worker]
            if worker["finalRating"][method]["rating"] == "OK":
                ok_count_final[cur_job] +=1
            elif worker["finalRating"][method]["rating"] == "NOK":
                nok_count_final[cur_job] +=1        
        print("Job {}: OK -> {} / {} (# of Workers) = {}%".format(cur_job, ok_count_final[cur_job], nok_count_final[cur_job] + ok_count_final[cur_job], 100 * ok_count_final[cur_job]/(nok_count_final[cur_job] + ok_count_final[cur_job])))
    
    ok_count_final_total = sum(ok_count_final.values())
    nok_count_final_total = sum(nok_count_final.values())
    
    fig, ax = plt.subplots()    #, density=True
    bar_x = [1,2]
    bar_height = [nok_count_final_total, ok_count_final_total]
    bar_tick_label = ["NOK", "OK"]
    ok_ratio = ok_count_final_total / (nok_count_final_total + ok_count_final_total) * 100
    nok_ratio = nok_count_final_total / (nok_count_final_total + ok_count_final_total) * 100
    bar_label = [ "{}/{}={:.2f}".format(nok_count_final_total, (nok_count_final_total + ok_count_final_total), nok_ratio)+"%", "{}/{}={:.2f}".format(ok_count_final_total, (nok_count_final_total + ok_count_final_total),ok_ratio)+"%" ]
    
    bar_plt = plt.bar(x=bar_x, height=bar_height, tick_label=bar_tick_label, color="navajowhite", edgecolor="k", alpha=0.65)
    title_str = "Final Rating Distribution: Correct Labeled Vehicles/Number of Estimated Cars"
    title_str +=" > {:.2f}% => \"OK\"\n(Verification with {}interface)".format(threshold_payment, method)
    plt.title(title_str)       
    plt.ylabel("# of Workers")
    
    def autolabel(rects):   # https://moonbooks.org/Articles/How-to-add-text-on-a-bar-with-matplotlib-/ 30.01.2021
        for idx,rect in enumerate(bar_plt):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height / 2,
                    bar_label[idx],
                    ha='center', va='bottom', rotation=0)
    autolabel(bar_plt)
    plt.ylim(0,max([nok_count_final_total, ok_count_final_total])+2)        
    fig.tight_layout()
    
    fig1 = plt.gcf()  # Needed because after plt.show() new fig is created and savefig would safe empty fig
    
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    # figManager = plt.get_current_fig_manager() # for fullscreen
    # figManager.window.state("zoomed")
    # figManager.full_screen_toggle()
    if savePlot:
        path = "figures/overall_quality_parameters"
        create_dir_if_not_exist(path) 
        
        fname = "/{}_total_rating_distribution_ok_nok.png".format(method)
        path += fname
        plt.savefig(path, format="png", dpi=300, bbox_inches="tight")
        plt.close("all")  
    
    return [OK_int_cluster_idc, approvedCluster_weak_idc, worker_rating]  # , cars_in_shds, worker_rating ]


def delete_txt_bplaced(ftp):
    ls = []
    ftp.retrlines('MLSD', ls.append)
    if len(ls) > 2: # standard file ".", ".."
        for f in ftp.nlst():
            if f[-4:] == ".txt":    # Delete all textfiles
                try:
                    ftp.delete(f)
                except:
                    print("Error deleting file={}".format(f))


def upload_rating_files(config):
    print("Uploading rating input files to bplaced server")
    # Get data from server via ftp
    ftp = ftplib.FTP()
    ftp.connect(config["ftp"]["url"])
    ftp.login(user=config["ftp"]["user"], passwd=config["ftp"]["passwd"])
    print("\nConnected to bplaced: " + ftp.getwelcome() + "\n")

    # Clear old campaign data
    root_dir_server = "/www/Admininterface/Post Rating"
    ftp.cwd(root_dir_server)  # change dir to subdir
    delete_txt_bplaced(ftp)

    root_dir_server = "/www/Admininterface/Pre Rating/"
    ftp.cwd(root_dir_server)  # change dir to subdir
    delete_txt_bplaced(ftp)

    # Upload new data
    root_dir_local = "Admininterface/Pre Rating/"
    flist = glob.glob(root_dir_local + "*.txt")
    try:
        for path in flist:
            fname = path.split("\\")[1]
            with open(path, "rb") as f:
                ftp.storlines("STOR {}".format(fname), f)
        print("Success uploading rating data for admininterface to bplaced server")
    except:
        print("Error uploading rating data for admininterface to bplaced server")

    # Clear old Crowdinterface data
    root_dir_server = "/www/Crowdinterface_Questions/fb/"
    ftp.cwd(root_dir_server) # change dir to subdir
    delete_txt_bplaced(ftp)

    root_dir_server = "/www/Crowdinterface_Questions/results/"
    ftp.cwd(root_dir_server)
    delete_txt_bplaced(ftp)

    root_dir_server = "/www/Crowdinterface_Questions/time_clicks/"
    ftp.cwd(root_dir_server)
    delete_txt_bplaced(ftp)

    root_dir_server = "/www/Crowdinterface_Questions/Data/"
    ftp.cwd(root_dir_server)
    delete_txt_bplaced(ftp)

    # Upload Crowd rating
    root_dir_local = "Crowdinterface/Pre Rating/"
    flist = glob.glob(root_dir_local + "*.txt")
    try:
        for path in flist:
            fname = path.split("\\")[1]
            with open(path, "rb") as f:
                ftp.storlines("STOR {}".format(fname), f)
        print("Success uploading questions for crowdinterface to bplaced server")
    except:
        print("Error uploading questions for crowdinterface to bplaced server")


def ftp_download_ratings(config, method):
    print("Downloading {} rating files from bplaced server (answers to questions)".format(method))
    # Get data from server via ftp
    ftp = ftplib.FTP()
    ftp.connect(config["ftp"]["url"])
    ftp.login(user=config["ftp"]["user"], passwd=config["ftp"]["passwd"])
    print("\nConnected to bplaced: " + ftp.getwelcome() + "\n")

    if method == "admin":   # car acquisitions
        directory = set_dir("admin")

        clearOldValues = True
        if clearOldValues == True:
            for f in glob.glob(directory["rootDir_local"] + "/*.txt"):
                os.remove(f)

        ftp.cwd(directory["rootDir_server"])

        for fname in ftp.nlst():
            try:
                print("Downloading admin rating file {}".format(fname))
                local_filename = os.path.join(os.getcwd() + "/" + directory["rootDir_local"] + "/" + fname)
                local_file = open(local_filename, "wb")
                ftp.retrbinary("RETR "+fname, local_file.write)
                local_file.close()
            except:
                print("Error downloading admin rating file \"{}\" from bplaced server".format(fname))

    if method == "crowd":   # questions
        directory = set_dir(method="questions")

        fetch_data(directory["rootDir_server"], directory["rootDir_local"], directory["subDir"], config, pre_rating=directory["pre_rating"])

        # Check data (is it complete or is something missing)
        root = directory["rootDir_local"]
        subDir = directory["subDir"]

        # Remove empty files
        path = root + subDir[0] + "/*.txt"
        flist = glob.glob(path)
        for path in flist:
            # If file is empty -> delete
            try:
                fsize = os.path.getsize(path)
                if fsize == 0:
                    print("File in dir: {} is empty -> delete".format(path))
                    os.remove(path)
                    continue
            except:
                print("Error: deleting file {} failed".format(fname))

        # Check if data is complete
        batch_numb = [*range(1,config["jobs"]["number_of_jobs"]+1)]

        sub_it_numb, _ = calc_sub_it_numb(config)
        # sub_it_numb = [ [*range(x)] for x in config["interface_questions"]["sub_it_numb"]]
        it_numb = [*range(1, config["interface_questions"]["it_numb"]+1)]

        missing_questions = {}
        for cur_subDir in subDir[0:2]:
            missing_questions[cur_subDir] = []
            # createDir(subDir_matched + cur_subDir, ["/OK", "/NOK"])
            for cur_batch in batch_numb:
                for cur_sub_it in sub_it_numb[cur_batch-1]:
                    for cur_it in it_numb:
                        fname = "{}-{}-{}.txt".format(cur_batch, cur_sub_it, cur_it)
                        path = root + cur_subDir + "/" + fname
                        if os.path.exists(path):
                            continue
                        else:
                            missing_questions[cur_subDir].append((cur_batch, cur_sub_it, cur_it))
                            print("File missing: batch={}, sub_it={}, it={}".format(cur_batch, cur_sub_it, cur_it))
        if not set(missing_questions[subDir[0]]) - set(missing_questions[subDir[1]]):
            print("Results and time files match")
        else:
            print("Mismatch between /result and /time_clicks data")

    ftp.quit


def get_active_campaign(mw_api, method, config):
    """ 
    Get active microworkers campaign with status "PAUSED_SYSTEM" and slots "notRated"
    
    Parameters:
    ----------  
        mw_api: class instance
        method: string
            "acquisitions", "questions"
    
    Returns:
    ----------
        camp_info: dict
            campaign info for listbox
        camp_running
        camp_to_rate

    """
    UserInformation = mw_api.do_request("get", "/accounts/me") # test api connection

    # List all hire-group-campaigns
    HR_camp_info_all = mw_api.do_request("get", "/hire-group-campaigns")

    # Get specific Campaign ! Only 1 campaign
    HR_campaign = {}
    for cur_camp, camp_data in enumerate(HR_camp_info_all["value"]["items"]):
        HR_campaign[cur_camp] = {
            "title": camp_data["title"],
            "status": camp_data["status"],
            "campaignId": camp_data["id"],
            "availablePositions": camp_data["availablePositions"],
            "paymentPerTask": camp_data["paymentPerTask"],
            "slotsCount": {
                "locked": camp_data["slotsCount"]["locked"],
                "ok": camp_data["slotsCount"]["ok"],
                "nok": camp_data["slotsCount"]["nok"],
                "notRated": camp_data["slotsCount"]["notRated"],
                "unTaken": camp_data["slotsCount"]["unTaken"]
            }
        }

    # !!! Only works for 1 active campaign at a time
    camp_to_rate = []; camp_running = []
    for camp_idx in HR_campaign:
        camp_data = HR_campaign[camp_idx]
        try: 
            if method == "acquisitions":
                if camp_data["title"] == "TTV-"+config["campaign-acquisitions"]["title"]:
                    if camp_data["status"] == "RUNNING":
                        print("\nCampaign running currently:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                        print("\n--> Wait for campaign to finish.")
                        camp_running.append(camp_idx)
                    elif camp_data["status"] == "PAUSED_SYSTEM" and camp_data["availablePositions"] == camp_data["slotsCount"]["notRated"]:   # Campaign is ready for rating (Data collection finished)
                        print("\nCampaign READY for rating:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                        camp_to_rate.append(camp_idx)  # --> for to be rated campaign
                    elif camp_data["status"] == "FINISHED" and camp_data["slotsCount"]["notRated"] > 0:
                        print("\nCampaign READY for rating:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                        camp_to_rate.append(camp_idx)  # --> for to be rated campaign
                    else:
                        print("\nCampaign NOT READY for rating:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                else:
                    print("\n No Campaign found with title {}".format(config["campaign-acquisitions"]["title"]))
            elif method == "questions":
                if camp_data["title"] == "TTV-"+config["campaign-questions"]["title"]:
                    if camp_data["status"] == "RUNNING":
                        print("\nCampaign running currently:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                        print("\n--> Wait for campaign to finish.")
                        camp_running.append(camp_idx)
                    elif camp_data["status"] == "PAUSED_SYSTEM" and camp_data["availablePositions"] == camp_data["slotsCount"]["notRated"]:   # Campaign is ready for rating (Data collection finished)
                        print("\nCampaign READY for rating:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                        camp_to_rate.append(camp_idx)  # --> for to be rated campaign
                    elif camp_data["status"] == "FINISHED" and camp_data["slotsCount"]["notRated"]  > 0:
                        print("\nCampaign READY for rating:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                        camp_to_rate.append(camp_idx)  # --> for to be rated campaign
                    elif camp_data["status"] == "PENDING_REVIEW":
                        print("\nCampaign Pending Review:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                        camp_running.append(camp_idx)
                    else:
                        print("\nCampaign NOT READY for rating:\nTitle = {}, Campaign Id = {}, Status = {}".format(camp_data["title"], camp_data["campaignId"], camp_data["status"]))
                else:
                    print("\n No Campaign found with title {}".format(config["campaign-questions"]["title"]))
        except:
            print("ERROR: selection of to be rated campaign failed")

    # If multiple campaigns -> exit program
    # if len(camp_to_rate) > 1:
    #    print("More than 1 campaign ready for rating, not expexted -> Exit")
    #    exit()
    
    camp_info = []
    for camp in camp_running:
        info = HR_campaign[camp]
        camp_info.append("Title=" + info["title"] + ", status="+ info["status"] + ", campaignId=" + info["campaignId"] + ", slots unTaken=" + str(info["slotsCount"]["unTaken"]) + ", slots notRated=" + str(info["slotsCount"]["notRated"]))
    
    for camp in camp_to_rate:
        info = HR_campaign[camp]
        camp_info.append("Title=" + info["title"] + ", status="+ info["status"] + ", campaignId=" + info["campaignId"] + ", slots unTaken=" + str(info["slotsCount"]["unTaken"]) + ", slots notRated=" + str(info["slotsCount"]["notRated"]))
    
    if camp_running:
        camp_running = HR_campaign[camp_running[0]]
    else:
        camp_running = None
        
    if camp_to_rate:
        camp_to_rate = HR_campaign[camp_to_rate[0]]
    else:
        camp_to_rate = None
        
    if camp_running is None and camp_to_rate is None:
        camp_info = ["Currently No Active Campaign.", ]
    
    return camp_info, camp_running, camp_to_rate
    
    # HR_campaign = HR_campaign[ camp_to_rate[0] ]
    # return HR_campaign


def move_existing(root_dir_local, sub_dir):
    """ 
    Move existing files to backup folder
    
    Parameters:
    ----------
        rootDir_local: string
        subDir: list
            ["fb", "results", "time"] 
    
    """
    # Current time -> new backup folder time coded 
    timestamp = datetime.now()
    timestamp = str(timestamp.strftime("%y%m%d_%H-%M-%S"))
    
    for cur_subDir in sub_dir:
        # Move existing data in local file system to backup folder
        src = os.path.join(os.getcwd() + "/" + root_dir_local + "/" + cur_subDir[1:] + "/")
        existing_data = len(glob.glob(src + "*")) > 0
        if existing_data:
            print("\nFolder=\"{}\" is not empty, copy data to backup folder".format(cur_subDir))
            dst = os.path.join(os.getcwd() + "/backup/" + root_dir_local + "/" + timestamp + "/" + cur_subDir[1:] + "/")
            try:
                copy_tree(src, dst)
                try:
                    files = glob.glob(src + "*")
                    for f in files:
                        os.remove(f)                    
                except:
                    print(" Error trying to delete old files")
            except:
                print("Error trying to copy to backup folder")
        # Same for matched folder
        src2 = os.path.join(os.getcwd() + "/" + root_dir_local + "/matched" + "/" + cur_subDir[1:] + "/")
        existing_data2 = len(glob.glob(src2 + "*")) > 0
        if existing_data2:
            dst2 = os.path.join(os.getcwd() + "/backup/" + root_dir_local + "/" + timestamp + "/matched/" + cur_subDir[1:] + "/")
            try:
                copy_tree(src2, dst2)
                try:
                    files = glob.glob(src2 + "*")
                    for f in files:
                        os.remove(f)                    
                except:
                    print(" Error trying to delete old files")
            except:
                print("Error trying to copy to backup folder")

    return timestamp


def match_data(HR_campaign, mw_api, root_dir_local, sub_dir, method, save_country):
    """ 
    Match data on bplaced server with not rated slots on microworkers end
    (Compare data located on bplaced server with data on microworkers end)
    
    Parameters:
    ----------
        HR_campaign: dict
        mw_api:
        rootDir_local:
        subDir:
        method:
        saveCountry: bool
    """
    # Make directories if not existing
    subDir_matched = root_dir_local + "/matched"
    if not os.path.exists(subDir_matched):
        os.makedirs(subDir_matched)
    for cur_subDir in sub_dir:
        if not os.path.exists(root_dir_local + "/matched" + cur_subDir):
            os.makedirs(root_dir_local + "/matched" + cur_subDir)
    
    # Match data on server with notRated slots on microworkers end
    # state = "PAUSED_SYSTEM"
    # state = "RUNNING"
    # if HR_campaign["status"] == state:# and HR_campaign["availablePositions"] == HR_campaign["slotsCount"]["notRated"]:
    # Get notRated slots
    params = { 
        "pageSize": HR_campaign["slotsCount"]["notRated"],
        # "sort": "status"
        "status": "NOTRATED"          
    }
    HR_camp_slots = mw_api.do_request("get", "/hire-group-campaigns/"+HR_campaign["campaignId"]+"/slots", params=params)["value"]["items"]
    # Create list with all slotId's and workerId's
    mw_slotId_list = []; mw_workerId_list = []; mw_country = []
    for slot in HR_camp_slots:
        mw_slotId_list.append(slot["id"])
        mw_workerId_list.append(slot["workerId"])
        mw_country.append(slot["country"])

    # Compare with downloaded result files
    local_workerId_list = []; local_slotId_list = []
    
    path = root_dir_local + "/" + "results/*.txt"         # os.getcwd() + "\\" + rootDir_local + "\\"
    for fname in glob.glob(path):
        # If file is empty -> delete
        try:
            fsize = os.path.getsize(fname)
            if fsize == 0:
                print("File in dir: {} is empty -> delete".format(fname))
                os.remove(fname)
                continue
        except: 
            print("Error: deleting file {} failed".format(fname))
        
        # Retrieve batch_num, cur_it, workerId & slotId from fname string 
        path = root_dir_local + "/results\\\\"
        fname = re.sub(root_dir_local + "/results\\\\", "", fname)
        fname = fname.split("-") 
        batch_num = fname[0]
        if method == "questions":
            sub_it = fname[1]
            fname = fname[2].split("_")
        elif method == "acquisitions":
            fname = fname[1].split("_")
        cur_it = fname[0]
        workerId = fname[1]       
        slotId = re.sub(".txt", "", fname[2])
        
        local_workerId_list.append(workerId)
        local_slotId_list.append(slotId)

    # Find workerId in workerId_list
    missing_slots = []; matching_slots = []; mismatch_slots = []; matching_country = []
    for idx, mw_slotId in enumerate(mw_slotId_list):
        try:
            local_idc = [i for i, x in enumerate(local_slotId_list) if x == mw_slotId]
            if not local_idc:
                print("Attention: mw_slot not found in local data")               
                try:
                    mw_workerId = mw_workerId_list[idx]
                    missing_slots.append([mw_workerId, mw_slotId])
                except:
                    print("Error: missing slots")
            # print(local_idc)
            for local_idx in local_idc:               
                # print(local_idx)
                local_workerId = local_workerId_list[local_idx]
                if local_workerId == mw_workerId_list[idx]:    # Slot & WorkerId match
                    # Move 
                    matching_slots.append([local_workerId, mw_slotId])
                    matching_country.append(mw_country[idx])
                    # print("Slot matches")
                else:  # mismatch -> delete, too much data (Data on own server but not registered on microworkers server -> submit failed after writing data or other error)
                    mismatch_slots.append([local_workerId, mw_slotId])
                    print("Worker {} with Slot {} to be deleted, failed submit on microworkers side".format(mw_workerId_list[idx], mw_slotId))
                    
        except ValueError:
            print("Error matching slots")
        
    if save_country:
        path = subDir_matched + "/countries.txt"
        matching_country = list(zip([*Counter(matching_country).keys()], [*Counter(matching_country).values()]))
        with open(path, "w") as f:
            f.write("Country    Number of Worker")
            for country in matching_country:
                f.write("\n{}\t{}".format(country[0], country[1]))
    
    # Move matched data to folder "matched"
    for ids in matching_slots:
        for cur_subDir in sub_dir:
            dstDir = os.path.join(root_dir_local + "/matched" + "/" + cur_subDir[1:] + "/")
            
            try:
                if method =="acquisitions":
                    src = glob.glob(root_dir_local + "/" + cur_subDir + "/*_" + ids[0] + "_" + ids[1] + ".txt")[0]
                    fname = re.sub(root_dir_local + "/" + cur_subDir + "\\\\", "", src)
                elif method == "questions":
                    src = glob.glob(root_dir_local + cur_subDir + "/*_" + ids[0] + "_" + ids[1] + ".txt")[0]
                    fname = re.sub(root_dir_local + cur_subDir + "\\\\", "", src)
                dst = dstDir + fname
                
                copyfile(src, dst)
            except ValueError:
                print(" Error moving files to \"matched\" folder ")
    print()
    # else:
    #    print("Error: Not all slots are ready for rating process!")

    # Slot info of missing
    if method == "questions":
        for idx, slot in enumerate(missing_slots):
            slot_info = mw_api.do_request("get", "/slots/"+slot[1])
            
            batch_numb = slot_info["value"]["tasksAnswers"][0]["questionsAnswers"][0]["answer"]
            cur_sub_it = slot_info["value"]["tasksAnswers"][0]["questionsAnswers"][1]["answer"]
            cur_it = slot_info["value"]["tasksAnswers"][0]["questionsAnswers"][2]["answer"]
            delta_t = slot_info["value"]["tasksAnswers"][0]["questionsAnswers"][3]["answer"]

            missing_slots[idx].extend([batch_numb, cur_sub_it, cur_it, delta_t])

    return


def load_ground_truth(cur_job, config, plot, save_plot):
    """ 
    Load ground truth data
    
    Parameters
    ----------
        cur_job: int
            Index of current job
        config: dict
            Current config
        plot: boolean
            True: Acquisitions get plotted on top of shd
            False: No plots
        save_plot: boolean
            True: Save plots in standard folder defined in config file
    Returns
    ----------
        loadedAcquisition: dict
            coordinates, mean, car axis length, car axis angle
    """
    loadedGroundTruth = dict()
    loadedGroundTruth = {
        "start": {"x": [], "y": []},
        "end": {"x": [], "y": []},
        "mean": {"x": [], "y": []},
        "car_axis_len": [],
        "car_axis_angle": []
    }    
    
    path = config["directories"]["GT_Folder"] + "\\\\" + str(cur_job+1) + "-groundtruth.txt"
    with open(path) as f:
        for line in f:
            (startX, startY, endX, endY) = [float(x) for x in line.split()]  # line.split()
                
            loadedGroundTruth["start"]["x"].append(startX)
            loadedGroundTruth["start"]["y"].append(startY)
            loadedGroundTruth["end"]["x"].append(endX)
            loadedGroundTruth["end"]["y"].append(endY)
            loadedGroundTruth["mean"]["x"].append((startX + endX) / 2)
            loadedGroundTruth["mean"]["y"].append((startY + endY) / 2)
            loadedGroundTruth["car_axis_len"].append(np.sqrt((startX - endX) ** 2 + (startY - endY) ** 2))
            
            # angle calc
            loadedGroundTruth["car_axis_angle"].append(np.arctan2((endY- startY), (endX - startX)))
        
    # Plot
    if plot:        
        cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
        
        f1 = plt.figure(1)
        ax = plt.subplot(111)
        
        height, width = cur_img.shape    
        # extent = [0.5, width+0.5, height+0.5, 0.5]      # Account for different coordinate origin Html Canvas(0,0) == upper left corner, Matlab(1,1)==Pixel Center Upper left corner, Python(0,0) Pixel Center Upper left corner
        extent = [-1, width-1, height-1, -1]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-1, -1) via addition of constants in original webinterface
        
        ax.imshow(cur_img, cmap="gray", origin="upper", extent=extent, interpolation='none')
        
        ax.plot([loadedGroundTruth["start"]["x"], 
                loadedGroundTruth["end"]["x"]],
                [loadedGroundTruth["start"]["y"],
                loadedGroundTruth["end"]["y"]], color="green", linewidth=1)
        
        plt.title('Streifen {}, Fahrzeuganzahl={}'.format(cur_job+1, len(loadedGroundTruth["start"]["x"])))
       
        gt_patch = Line2D([0], [0], linestyle='-', color='green', label='Referenz',linewidth=1)               
        ax.legend(handles=[gt_patch], fancybox=True, shadow=True, handlelength=1, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis("off")
        
        fig1 = plt.gcf()
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

        # figManager = plt.get_current_fig_manager() # for fullscreen
        # figManager.window.state("zoomed")
        # plt.show()
        if save_plot:
            
            path = config["directories"]["Figures"] + "{}/".format(cur_job+1)
            create_dir_if_not_exist(path) 
            
            fname = 'job_{}_clusterCount_{}.png'.format(cur_job+1, len(loadedGroundTruth["start"]["x"]))
            path += fname
            
            fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")
            plt.close("all")  
        
        # Plot zoomed in version
        #if cur_job == 0:
        cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
        f1, (ax1, ax2) = plt.subplots(2)
                
        # Plot image
        height, width = cur_img.shape
        extent = [-1, width-1, height-1, -1 ]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-
        ax2.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')  
        
        # Plot acquis
        ax2.plot([loadedGroundTruth["start"]["x"], loadedGroundTruth["end"]["x"]],
                 [loadedGroundTruth["start"]["y"], loadedGroundTruth["end"]["y"]], 
                 color="green", linewidth= 1, label="Referenz")
                    
        # Plot image
        if cur_job == 0:
            x_min = 300; x_max = 550
            y_min = 350; y_max = 499
            #x_min = 1365; x_max = 1410
            #y_min = 0; y_max = 45
        if cur_job == 1:
            x_min = 778; x_max = 778+163
            y_min = 95; y_max = 95+150
        if cur_job == 2:
            x_min = 1283; x_max = 1283+185
            y_min = 4; y_max = 4+150
        if cur_job == 3:
            x_min = 2332; x_max = 2332+209
            y_min = 54; y_max = 4+209
        if cur_job == 4:
            x_min = 1812; x_max = 1812+172
            y_min = 247; y_max = 247+141
        if cur_job == 5:
            x_min = 421; x_max = 421+217
            y_min = 252; y_max = 252+200
        
        ax1.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
        ax1.plot([loadedGroundTruth["start"]["x"], loadedGroundTruth["end"]["x"]],
                 [loadedGroundTruth["start"]["y"], loadedGroundTruth["end"]["y"]], 
                color="green", linewidth= 1, label="Referenz")
                    
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_max, y_min)
        
        ax1.axis("off")
        ax2.axis("off")
        
        acqui_patch = Line2D([0], [0], linestyle='-', color='green', label='Referenz',linewidth=1)               
        ax1.legend(handles=[acqui_patch], fancybox=True, shadow=True, handlelength=1, loc='center left', bbox_to_anchor=(1, 0.5))

        ax1.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
        ax2.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
        
        plt.title('Streifen {}, Fahrzeuganzahl={}'.format(cur_job+1, len(loadedGroundTruth["start"]["x"])))
        
        fig1 = plt.gcf()
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        if save_plot:
            path = config["directories"]["Figures"] +'{}/'.format(cur_job+1)
            create_dir_if_not_exist(path) 
            
            fname = 'job_{}_clusterCount_{}_ZOOMED.png'.format(cur_job+1, len(loadedGroundTruth["start"]["x"]))
            path += fname
            
            fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")       # , bbox_inches="tight" -> save legend to plot
            plt.close("all") 
    return loadedGroundTruth


def load_acquisitions(cur_job, acqui_numbr, config, plot, save_plot, counter_len_null):
    # config.number_of_acquisitions = acqui_numbr

    """
    Load acquisitions
    
    Parameters
    ----------
        cur_job: int
            Index of current job
        acqui_numbr: int
            Number of acquisitions per job
        plot: boolean
            True: Acquisitions get plotted on top of shd
            False: No plots
        save_plot: boolean
            True: Save plots in standard folder defined in config file
    
    Returns
    ----------
        loadedAcquisition: dict
            coordinates, integration results, idc, workerId, removed outliers    
    """
    # Init dict to return later on
    loadedAcquisition = dict()    
    loadedAcquisition = {
        "workerId_list": [],
        "all": {
            "start": { "x": [], "y": [] },
            "end": { "x": [], "y": [] },
        },
        "start": { "x": [], "y": [] },
        "end": { "x": [], "y": [] },
        "mean": { "x": [], "y": [] },
        "car_axis_len": [],
        "car_axis_angle": [],
        "job_idc": [],
        "job_idc_remaining_1D": [],
        "job_idc_remaining": [], 
        "job_idc_remaining_orig": [],   #!! soll job_idc_remaining ersetzen
        "job_idc_orig":[],
        "job_idc_orig_1D": [],
        "acqui_count": [],
        "workerId": [],
        "slotId": [],
        "axis_too_short": {
            "acquisition": [],
            "idc": []
        },
        "removed": {
            "axis_too_short": {
                "workerId": [],
                "start": { "x": [], "y": [] },
                "end": { "x": [], "y": [] },
                "mean": { "x": [], "y": [] },
                "axis_len": [],
                "idc": [],
                "idc_orig": [], # !! soll idc ersetzen
                "acquisition": {}
            }
        }
    }
    
    # Init local variables
    start_idc = 0
    start_idc_ori = 0
    startX = []
    startY = []
    endX = []
    endY = []

    # Mirrored acquisitions: jobs 1-7 normal, 8-13 mirrored not implemented
    acqui_total_numbr = acqui_numbr
        
    path = config["directories"]["Res_Folder"] + str(cur_job+1) + "-*.txt"
    path_list = glob.glob(path)
    acqui_total_numbr = len(path_list)
    acqui_numbr = acqui_total_numbr
    #for cur_acq in range(acqui_total_numbr): #config.number_of_acquisitions * 2):

    for cur_acq in range(acqui_total_numbr): #config.number_of_acquisitions * 2):
        if cur_acq <= acqui_numbr - 1:
            #print("current acquisition --> {}".format(cur_acq))
            
            #path = config.directories["Res_Folder"] + "/" + str(cur_job + 1) + "-" + str(cur_acq + 1) + ".txt"            
            
            #path = config.directories["Res_Folder"] + "\\\\" + str(cur_job+1) +"-*.txt"
            #if not glob.glob(path):
            #    print("missing acuqisition={} for job={}".format(cur_acq, cur_job))
            #    loadedAcquisition["job_idc"].append([])  # To do
            #    continue
            #path = glob.glob(path)[0]   # Complete path with "asterisk" part (workerId)
            path = path_list[cur_acq]
            
            
            # Retrieve workerId from fname string            
            #workerId = re.sub(config.directories["Res_Folder"] + "\\\\" + str(cur_job + 1) + "-" + str(cur_acq + 1) + "_", "", path)            
            #workerId = re.sub("\.txt$", "", workerId)
            # string = re.sub(config.directories["Res_Folder"] , "", path)
            string = re.sub("\.txt$", "", path).split("\\")
            #string = re.sub("\.txt$", "", string).split("-")
            string = string[1].split("-")
            
            cur_batch = string[0]
            string = string[1].split("_")    
            cur_it = string[0]     
            # WorkerId
            workerId = string[1]
            # SlotId  
            slotId = string[2]
            
            # Read Result File
            cars_acquired = 0
            linecount = 0
            removed_idc = []
            remaining_idc = []           
            
            with open(path) as f:
                for line in f:
                    # Linecount
                    linecount += 1
                    cars_acquired += 1
                    # Split line to columns
                    (startX, startY, endX, endY) = [float(x) for x in line.split()]  # line.split()
                    # Remove acquisitions, where car axis == 0
                    cur_axis_len = np.sqrt((startX - endX) ** 2 + (startY - endY) ** 2)
                    
                    # List with start/end for all submitted acquisitions
                    loadedAcquisition["all"]["start"]["x"].append(startX)
                    loadedAcquisition["all"]["start"]["y"].append(startY)
                    loadedAcquisition["all"]["end"]["x"].append(endX)
                    loadedAcquisition["all"]["end"]["y"].append(endY)
                    
                    if cur_axis_len > config["integration"]["minimal_length"]:    
                        remaining_idc.append(linecount - 1)
                             
                        loadedAcquisition["workerId_list"].append(workerId)
                        loadedAcquisition["start"]["x"].append(startX)
                        loadedAcquisition["start"]["y"].append(startY)
                        loadedAcquisition["end"]["x"].append(endX)
                        loadedAcquisition["end"]["y"].append(endY)
                        loadedAcquisition["mean"]["x"].append((startX + endX) / 2)
                        loadedAcquisition["mean"]["y"].append((startY + endY) / 2)
                        loadedAcquisition["car_axis_len"].append(cur_axis_len)
                            
                        # Calculate car axis angle
                        LP = []
                        RP = []
                        if startX < endX:
                            LP.extend([startX, startY])
                            RP.extend([endX, endY])
                        else:
                            LP.extend([endX, endY])
                            RP.extend([startX, startY])
                            
                        loadedAcquisition["car_axis_angle"].append(np.arctan2(RP[1] - LP[1], RP[0] - LP[0]))
                            
                    else:
                        removed_idc.append(linecount - 1)
                        loadedAcquisition["removed"]["axis_too_short"]["start"]["x"].append(startX)
                        loadedAcquisition["removed"]["axis_too_short"]["start"]["y"].append(startY)
                        loadedAcquisition["removed"]["axis_too_short"]["end"]["x"].append(endX)
                        loadedAcquisition["removed"]["axis_too_short"]["end"]["y"].append(endY)
                        loadedAcquisition["removed"]["axis_too_short"]["mean"]["x"].append((startX + endX) / 2)
                        loadedAcquisition["removed"]["axis_too_short"]["mean"]["y"].append((startY + endY) / 2)
                        loadedAcquisition["removed"]["axis_too_short"]["axis_len"].append(cur_axis_len)
                        continue
                    
            # Job indice
            end_idc = start_idc + linecount - len(removed_idc)                      
            loadedAcquisition["job_idc"].append([*range(start_idc, end_idc, 1)])
            
            loadedAcquisition["acqui_count"].append(linecount)
            
            # Job indice for all acquisitions
            end_idc_ori = start_idc_ori + linecount
            idc_list = [*range(start_idc_ori, end_idc_ori, 1)]  # list of indices for current worker            
            loadedAcquisition["job_idc_orig"].append(idc_list)
            loadedAcquisition["job_idc_orig_1D"].extend(idc_list)
            
            # 
            if removed_idc:               
                rest_idc = [i for j, i in enumerate(loadedAcquisition["job_idc_orig"][-1]) if j not in removed_idc]
                loadedAcquisition["job_idc_remaining_1D"].extend(rest_idc)
            else:
                loadedAcquisition["job_idc_remaining_1D"].extend([*range(start_idc_ori, end_idc_ori, 1)])
                
            loadedAcquisition["removed"]["axis_too_short"]["idc_orig"].append(list(np.array(idc_list)[removed_idc]))
            
            # worker with only axislen of 0
            if (linecount - len(removed_idc)) == 0:
                counter_len_null += 1
                print("All acquis of worker {} have len 0! Total worker with len 0 = {}".format(workerId, counter_len_null))
            
            # Worker ID
            loadedAcquisition["workerId"].append(workerId)
            loadedAcquisition["slotId"].append(slotId)
            
            # Save indice of removed acquisition
            loadedAcquisition["removed"]["axis_too_short"]["idc"].append(removed_idc)  # -1 to start idc at 0 #####TESTSTSTEST
            if removed_idc:                    
                # "acquisition" is irrelevant. "idc" + "workerId" are enough 
                # cars_in_shds[cur_job]["removed"]["axis_too_short"]["acquisition"][cur_acq] = removed_idc #.append(cur_acq) # [1-50] per Worker 1 Acquisition file with multiple vehicle acquisitions
                # loadedAcquisition["removed"]["axis_too_short"]["idc"].append(removed_idc) # -1 to start idc at 0
                loadedAcquisition["removed"]["axis_too_short"]["workerId"].append(workerId)   # ToDo: wie oben
            
            # Save indice of remaining acquisition
            loadedAcquisition["job_idc_remaining"].append(remaining_idc)
            loadedAcquisition["job_idc_remaining_orig"].append(list(np.array(idc_list)[remaining_idc]))   # !!!! soll job_idc_remaining ersetzen
            
            print("Acquisition {}: --> {} cars acquired".format(cur_acq, linecount))
            
            # Update start_idc
            if loadedAcquisition["job_idc"][cur_acq]:
                start_idc += loadedAcquisition["job_idc"][cur_acq][-1] - start_idc + 1
                
            start_idc_ori = end_idc_ori

         #    # else:
        #    #     path = config.directories["Res_Folder"] + "/" + str(cur_job + 1 + int(np.fix(config.number_of_jobs / 2))) + "-" + str(cur_acq - (config.number_of_acquisitions - 1)) + ".txt"
        #    #     cur_res_file = np.loadtxt(path)
        #    #     cur_axis_len = np.sqrt((cur_res_file[:,0] - cur_res_file[:,2]) ** 2 + (cur_res_file[:,1] - cur_res_file[:,3]) ** 2)
        #    #     cur_res_file = cur_res_file[cur_axis_len > config.minimal_length,:]
        #
        #    #     print("Acquisition {}: --> {} cars acquired".format(cur_acq, np.size(cur_res_file, 0)))
        #
        #    #     # Plot acquired cars
        #    #     for c in range(np.size(cur_res_file, 0)):
        #    #         #print(c)
        #    #         plt.plot([cur_res_file[:,0],cur_res_file[:,2]],[cur_res_file[:,1],cur_res_file[:,3]], color='green', linewidth= 2)
        #        
        #    #     startX.append(cur_res_file[:,0])
        #    #     startY.append(cur_res_file[:,1])
        #    #     endX.append(cur_res_file[:,2])
        #    #     endY.append(cur_res_file[:,3])
        #
        ## Plot acquired Vehicles with large enough axis length
        #plt.plot([cars_in_shds[cur_job]["start"]["x"],
        #          cars_in_shds[cur_job]["end"]["x"]],
        #         [cars_in_shds[cur_job]["start"]["y"],
        #          cars_in_shds[cur_job]["end"]["y"]], color="red", linewidth= 2)        
        ##plt.show()
        
    if plot:        
        cur_img = plt.imread(config["directories"]["Img_Folder"] + 'job' + str(cur_job + 1) + '/' + 'shd.png')
        
        f1 = plt.figure(1)
        ax = plt.subplot(111)
        
        height, width = cur_img.shape    
        extent = [-1, width-1, height-1, -1 ]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-1, -1) via addition of constants in original webinterface
        
        ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
        
        # Plot all acquis
        ax.plot([loadedAcquisition["all"]["start"]["x"], 
          loadedAcquisition["all"]["end"]["x"]],
        [loadedAcquisition["all"]["start"]["y"],
         loadedAcquisition["all"]["end"]["y"]], color="red", linewidth=1, label="Acquisition")
        
        # Plot acquis with len > threshold
        #ax.plot([loadedAcquisition["start"]["x"], 
        #          loadedAcquisition["end"]["x"]],
        #        [loadedAcquisition["start"]["y"],
        #         loadedAcquisition["end"]["y"]], color="red", linewidth=1, label="Acquisition")
        
        plt.title('Streifen {}, Crowdworker={}, Gesamte Erfassungen={}'.format(cur_job+1, cur_acq+1, len(loadedAcquisition["job_idc_orig_1D"])))
        acqui_patch = Line2D([0], [0], linestyle='-', color='red', label='Erfassung', linewidth=1)
        ax.legend(handles=[acqui_patch], fancybox=True, shadow=True, handlelength=1, loc='upper center', bbox_to_anchor=(0.5, 0))
        plt.axis("off")
        fig1 = plt.gcf()
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        # figManager = plt.get_current_fig_manager() # for fullscreen
        # figManager.window.state("zoomed")
        # plt.show()
        if save_plot:
            path = config["directories"]["Figures"] +'{}/'.format(cur_job+1)
            create_dir_if_not_exist(path) 
            
            fname = 'job_{}_crwd_wrkr_nbr_{}_total_acquisitions_{}.png'.format(cur_job+1, cur_acq+1, len(loadedAcquisition["job_idc_orig_1D"]))
            path += fname
            
            fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")       # , bbox_inches="tight" -> save legend to plot
            plt.close("all") 
        
        # Plot strip + zoomed in part
        if cur_job == 0:
            x_min = 1365; x_max = 1410
            y_min = 0; y_max = 45
        if cur_job == 1:
            x_min = 778; x_max = 778+163
            y_min = 95; y_max = 95+150
        if cur_job == 2:
            x_min = 1283; x_max = 1283+185
            y_min = 4; y_max = 4+150
        if cur_job == 3:
            x_min = 2332; x_max = 2332+209
            y_min = 54; y_max = 4+209
        if cur_job == 4:
            x_min = 1812; x_max = 1812+172
            y_min = 247; y_max = 247+141
        if cur_job == 5:
            x_min = 421; x_max = 421+217
            y_min = 252; y_max = 252+200
            
        #if cur_job == 0:
        cur_img = plt.imread(config["directories"]["Img_Folder"] + 'job' + str(cur_job + 1) + '/' + 'shd.png')
        
        f1, (ax1, ax2) = plt.subplots(2)
                
        # Plot image
        height, width = cur_img.shape
        extent = [-1, width-1, height-1, -1 ]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-
        ax2.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')  
        
        # Plot acquis
        ax2.plot([loadedAcquisition["all"]["start"]["x"], loadedAcquisition["all"]["end"]["x"]],
                 [loadedAcquisition["all"]["start"]["y"], loadedAcquisition["all"]["end"]["y"]], 
                 color="red", linewidth=1, label="Erfassung")
        
        # Plot image            
        #x_min = 1365; x_max = 1410
        #y_min = 0; y_max = 45
        
        ax1.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
        ax1.plot([loadedAcquisition["all"]["start"]["x"], loadedAcquisition["all"]["end"]["x"]],
                [loadedAcquisition["all"]["start"]["y"], loadedAcquisition["all"]["end"]["y"]], 
                color="red", linewidth=1, label="Erfassung")
                    
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_max, y_min)
        
        ax1.axis("off")
        ax2.axis("off")
        
        acqui_patch = Line2D([0], [0], linestyle='-', color='red', label='Erfassung',linewidth=1)               
        ax1.legend(handles=[acqui_patch], fancybox=True, shadow=True, handlelength=1, loc='center left', bbox_to_anchor=(1, 0.5))

        ax1.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
        ax2.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
        
        # plt.title('Streifen {}, Crowdworker={}, Gesamte Erfassungen={}'.format(cur_job+1, cur_acq+1, len(loadedAcquisition["job_idc_orig_1D"])))
        
        fig1 = plt.gcf()
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        if save_plot:
            path = config["directories"]["Figures"] + '{}/'.format(cur_job+1)
            create_dir_if_not_exist(path) 
            
            fname = 'job_{}_crwd_wrkr_nbr_{}_total_acquisitions_{}_ZOOMED.png'.format(cur_job+1, cur_acq+1, len(loadedAcquisition["job_idc_orig_1D"]))
            path += fname
            
            fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")       # , bbox_inches="tight" -> save legend to plot
            plt.close("all") 
        
        return(loadedAcquisition, counter_len_null)


def zooming_box(fig, ax1, roi, ax2, color='orange', linewidth=1):  # Source: https://stackoverflow.com/questions/24477220/use-subplots-to-zoom-into-timeseries-or-how-i-can-draw-lines-outside-of-axis-bor (accessed 30.03.2021)
    """
    **Notes (for reasons unknown to me)**
    1. Sometimes the zorder of the axes need to be adjusted manually...
    2. The figure fraction is accurate only with qt backend but not inline...
    """
    roiKwargs = dict([('fill', False), ('linestyle', '-'), ('color', color), ('linewidth', linewidth)])
    ax1.add_patch(Rectangle([roi[0], roi[2]], roi[1]-roi[0], roi[3]-roi[2], **roiKwargs))


def integrate_main(config):
    """ 
    Main integration script:
        integrates acquisitions, creates .txt files for admin & crowdinterface 
        resulting variables saved with pickle as .dat file in /backup folder
    """
    
    # Skip integration and use stored variables
    load1 = False
    
    if load1 == True:
        try:
            with open(config["backup"]["cars_in_shds_pre_verification"], "rb") as file:
                cars_in_shds = pickle.load(file)
        except:
            print("Error trying to open backup file")
    
    # Start integrate 
    if load1 == False:
        clearFigures = True
        if clearFigures:
            for cur_job in range(config["jobs"]["number_of_jobs"]):        
                # Remove existing figures (clear folders)
                figBaseDir = "figures" + "\\\\" + str(cur_job + 1) + "\\\\"
                
                # Cluster + ellipse
                path = figBaseDir + "*.png"             
                files = glob.glob(path)
                for f in files:
                    os.remove(f) 
                    
                # Raw acquisitions + DBSCAN  
                path = figBaseDir + "*.png"
                files = glob.glob(path)
                for f in files:
                    os.remove(f)
                
                # "OK" and "NOK" rating results 
                path = figBaseDir + "Verify_Rating"
                files = glob.glob(path + "\\\\" + "OK" + "\\\\" + "*.png")
                for f in files:
                    os.remove(f)
                files = glob.glob(path + "\\\\" + "NOK" + "\\\\" + "*.png")
                for f in files:
                    os.remove(f)     
                files = glob.glob(path + "\\\\" + "diff_reference_acqui" + "\\\\" + "OK" + "\\\\" + "" + "*.png")
                for f in files:
                    os.remove(f)  
                files = glob.glob(path + "\\\\" + "diff_reference_acqui" + "\\\\" + "NOK" + "\\\\" + "" + "*.png")
                for f in files:
                    os.remove(f)  
        
        # Create Directories
        for directory in config["directories"]:
            # print(config.directories[directory])
            create_dir_if_not_exist(config["directories"][directory])
        cars_in_shds = {}  # create dict to store all data
        dataToCheck = {}
        
        counter_len_null = 0
        
        for cur_job in range(config["jobs"]["number_of_jobs"]):  # 1:number_of_jobs
            print("current job --> {}".format(cur_job)) 
            
            # Init dict
            cars_in_shds[cur_job] = {
                "gt": {},
                "all": {
                    "start": {"x": [], "y": []},
                    "end": {"x": [], "y": []},
                },            
                "workerId_list": [],
                "start": {"x": [], "y": []},
                "end": {"x": [], "y": []},
                "mean": {"x": [], "y": []},
                "car_axis_len": [],
                "car_axis_angle": [],
                "job_idc": [],
                "job_idc_remaining": [], 
                "job_idc_orig": [],
                "workerId": [],
                "slotId": [],
                "removed": {
                    "axis_too_short": {
                        "workerId": [],
                        "start": {"x": [], "y": []},
                        "end": {"x": [], "y": []},
                        "mean": {"x": [], "y": []},
                        "axis_len": [],
                        "idc": [],
                        "acquisition": {}
                    },
                    "dbscan_outlier": {
                        "noise_mask": []
                    },
                    "dbscan_integrated_axis_len": {
                        "idc": [],
                        "cluster_idc": []
                    },
                    "kmeans_outlier": {
                        "idc": [],
                        "kmeans_mask": []
                    },
                },
                "dbscan": {
                    "label": [],
                    "cluster_core_idc": [],
                    "cluster_all_idc": [],
                    "center": {"x": [], "y": []},     # Contains mean cluster centers of core samples
                    "integrated_cluster_idc": [],
                    "integrated_params": {
                        "center": {"x": [], "y": []},
                        "axis_len": []
                    }
                },
                "kmeans": {
                    "label": [],
                    "center": [],
                    "cluster_idc": [],
                    "int_line_start": {"x": [], "y": []},
                    "int_line_end": {"x": [], "y": []},
                    "mean": {"x": [], "y": []},
                    "uncertain": {
                        "cluster_idc": []
                    },
                    "uncertain_acqui": {    # Method 1: acquis which are outside of error ellipse
                        "cluster_idc": [],
                        "acqui_idc": []
                    },
                    "uncertain_cluster": {  # Method 2: all acquis of clusters which have pearson<0.2 and std>2.0
                        "cluster_idc": [],
                        "acqui_idc": []
                    },
                    "OK_int_cluster_idc": [],    # For quality calculation
                    "FP": [],
                    "FN": [],
                    "TP": [],
                    "TN": []
                },
                "dbscanWEAK": {     # Übergangsweise
                    "label": [],
                    "cluster_core_idc": [],
                    "cluster_all_idc": [],
                    "center": {"x": [], "y": []},
                    "integrated_cluster_idc": [],
                    "integrated_params": {
                        "center": {"x": [], "y": []},
                        "axis_len": []
                    }
                },
                "Input4Interfaces": {
                    "ellResult": {
                        "cluster_idc": [],
                        "acqui_idc": []
                    },
                    "weakResult": {
                        # "cluster_idc": [],
                        "acqui_idc": [],
                        "mean": {"x": [], "y": []}
                    }
                },
            }

            # Load groundtruth
            cars_in_shds[cur_job]["gt"] = load_ground_truth(cur_job, config, plot=True, save_plot=True)
            
            # Load acquisitions (result files with crowdworker acquisitions)
            acqui_numbr = config["jobs"]["number_of_acquisitions"]
            mirrored = False
            plot = True
            savePlot = True
            
            loadedAcquisitions, counter_len_null = load_acquisitions(cur_job, acqui_numbr, config, plot, savePlot, counter_len_null)
            
            ## Übergangslösung, funktioniert aber ist etwas wild und unschööön            
            cars_in_shds[cur_job]["all"]["start"]["x"] = loadedAcquisitions["all"]["start"]["x"]
            cars_in_shds[cur_job]["all"]["start"]["y"] = loadedAcquisitions["all"]["start"]["y"]
            cars_in_shds[cur_job]["all"]["end"]["x"] = loadedAcquisitions["all"]["end"]["x"]
            cars_in_shds[cur_job]["all"]["end"]["y"] = loadedAcquisitions["all"]["end"]["y"]
            
            cars_in_shds[cur_job]["workerId_list"] = loadedAcquisitions["workerId_list"]
            cars_in_shds[cur_job]["start"]["x"] = loadedAcquisitions["start"]["x"]
            cars_in_shds[cur_job]["start"]["y"] = loadedAcquisitions["start"]["y"]
            cars_in_shds[cur_job]["end"]["x"] = loadedAcquisitions["end"]["x"]
            cars_in_shds[cur_job]["end"]["y"] = loadedAcquisitions["end"]["y"]
            cars_in_shds[cur_job]["mean"]["x"] = loadedAcquisitions["mean"]["x"]
            cars_in_shds[cur_job]["mean"]["y"] = loadedAcquisitions["mean"]["y"]
            cars_in_shds[cur_job]["car_axis_len"] = loadedAcquisitions["car_axis_len"]
            cars_in_shds[cur_job]["car_axis_angle"] = loadedAcquisitions["car_axis_angle"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["start"]["x"] = loadedAcquisitions["removed"]["axis_too_short"]["start"]["x"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["start"]["y"] = loadedAcquisitions["removed"]["axis_too_short"]["start"]["y"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["end"]["x"] = loadedAcquisitions["removed"]["axis_too_short"]["end"]["x"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["end"]["y"] = loadedAcquisitions["removed"]["axis_too_short"]["end"]["y"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["mean"]["x"] = loadedAcquisitions["removed"]["axis_too_short"]["mean"]["x"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["mean"]["y"] = loadedAcquisitions["removed"]["axis_too_short"]["mean"]["y"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["axis_len"] = loadedAcquisitions["removed"]["axis_too_short"]["axis_len"]
            cars_in_shds[cur_job]["job_idc"] = loadedAcquisitions["job_idc"]
            
            cars_in_shds[cur_job]["job_idc_orig"] = loadedAcquisitions["job_idc_orig"]
            cars_in_shds[cur_job]["acqui_count"] = loadedAcquisitions["acqui_count"]
            
            len(loadedAcquisitions["job_idc_remaining_1D"])    # 1D list with indices of remaining acquisitions
            # print(len(loadedAcquisitions["job_idc_orig_1D"]))
            
            cars_in_shds[cur_job]["workerId"] = loadedAcquisitions["workerId"]
            cars_in_shds[cur_job]["slotId"] = loadedAcquisitions["slotId"]
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["idc"] = loadedAcquisitions["removed"]["axis_too_short"]["idc"]
            
            # print(loadedAcquisitions["removed"]["axis_too_short"]["idc_orig"])
            # print(loadedAcquisitions["job_idc_remaining_orig"])
            
            cars_in_shds[cur_job]["removed"]["axis_too_short"]["workerId"] = loadedAcquisitions["removed"]["axis_too_short"]["workerId"]
            cars_in_shds[cur_job]["job_idc_remaining"] = loadedAcquisitions["job_idc_remaining"]
            
            ## Clustering with DBSCAN using optimized parameters    
            plt.close("all")            
            meanX = cars_in_shds[cur_job]["mean"]["x"]
            meanY = cars_in_shds[cur_job]["mean"]["y"]
            plot = True
            savePlot = True
            
            # eps = config["integration"]["DBSCAN-1"]["epsilon"]
            # minpts = config["integration"]["DBSCAN-1"]["minpts"]
            # db_optimized = dbscan(meanX, meanY , cur_job, eps, minpts, config, plot, savePlot)
            db_optimized = dbscan(meanX, meanY , cur_job, config["integration"]["DBSCAN-1"]["epsilon"], config["integration"]["DBSCAN-1"]["minpts"], config, plot, savePlot)
            
            # Write outlier and clusterpoints to file for CD      
            __path = "plotted_data_textformat/bestehende_integration__dbscan_1/"
            create_dir_if_not_exist(__path)
            __fname = "job_{}_ausreisser.txt".format(cur_job+1)  
            with open(__path + __fname, "w") as f:
                # 1. DBSCAN
                f.write("Job {} - Ausreisser der ersten DBSCAN Clusteranalyse: (x,y)\n".format(cur_job+1))
                
                __noiseX = np.array(meanX)[db_optimized["noise_mask"]]
                __noiseY = np.array(meanY)[db_optimized["noise_mask"]]
                
                for _idx, _x in enumerate(__noiseX):
                    _y = __noiseY[_idx]
                    f.write("\n{},{}".format(_x, _y))
            
            __path = "plotted_data_textformat/bestehende_integration__dbscan_1/"
            create_dir_if_not_exist(__path)
            __fname = "job_{}_clusterpunkte.txt".format(cur_job+1)
            with open(__path + __fname, "w") as f:
                # 1. DBSCAN
                f.write("Job {} - Clusterpunkte der ersten DBSCAN Clusteranalyse: (x,y)\n".format(cur_job+1))
                
                __clusterX = np.array(meanX)[~db_optimized["noise_mask"]]
                __clusterY = np.array(meanY)[~db_optimized["noise_mask"]]
                
                for _idx, _x in enumerate(__clusterX):
                    _y = __clusterY[_idx]
                    f.write("\n{},{}".format(_x, _y))
            
            # Store results in main dict
            cars_in_shds[cur_job]["dbscan"]["noise_mask"] = db_optimized["noise_mask"]     
            cars_in_shds[cur_job]["dbscan"]["label"] = db_optimized["labels"]   
            # cars_in_shds[cur_job]["removed"]["dbscan_outlier"]["noise_mask"] = db_optimized["noise_mask"]
            cars_in_shds[cur_job]["dbscan"]["center"]["x"] = db_optimized["centerX"]    
            cars_in_shds[cur_job]["dbscan"]["center"]["y"] = db_optimized["centerY"]
            cars_in_shds[cur_job]["dbscan"]["cluster_core_idc"] = db_optimized["cluster_core_idc"]
            cars_in_shds[cur_job]["dbscan"]["cluster_all_idc"] = db_optimized["cluster_all_idc"]
            cars_in_shds[cur_job]["dbscan"]["n_clusters_"] = db_optimized["n_clusters_"]
            cars_in_shds[cur_job]["dbscan"]["n_noise_"] = db_optimized["n_noise_"]
            
            ## Clustering with DBSCAN using weaker parameters with outliers, of first DBSCAN result, as inputs
            # Filter Uncertain Cluster: Need to be checked by an admin
            noise_mask = cars_in_shds[cur_job]["dbscan"]["noise_mask"]
            
            meanX = np.array(cars_in_shds[cur_job]["mean"]["x"])[noise_mask]
            meanY = np.array(cars_in_shds[cur_job]["mean"]["y"])[noise_mask]
            plot = True
            savePlot = True  
                      
            # eps = 10
            # minpts = 2
            # db_weak = dbscan(meanX, meanY, cur_job, eps, minpts, config, plot, savePlot)
            db_weak = dbscan(meanX, meanY, cur_job, config["integration"]["DBSCAN-2"]["epsilon"], config["integration"]["DBSCAN-2"]["minpts"], config, plot, savePlot)    
            # db_weak["noise_mask"]   # Outlier of weak dbscan -> Trash -> "NOK" Rating
            
            # Write outlier and clusterpoints to file for CD        
            __path = "plotted_data_textformat/ausreisser_nachuntersuchung__dbscan_2/"
            create_dir_if_not_exist(__path)
            __fname = "job_{}_ausreisser.txt".format(cur_job+1)
            with open(__path + __fname, "w") as f:
                # 2. DBSCAN
                f.write("Job {} - Ausreisser der zweiten DBSCAN Clusteranalyse: (x, y)\n".format(cur_job+1))
                
                __noiseX = np.array(meanX)[db_weak["noise_mask"]]
                __noiseY = np.array(meanY)[db_weak["noise_mask"]]
                
                for _idx, _x in enumerate(__noiseX):
                    _y = __noiseY[_idx]
                    f.write("\n{},{}".format(_x, _y))
                    
            __path = "plotted_data_textformat/ausreisser_nachuntersuchung__dbscan_2/"
            create_dir_if_not_exist(__path)
            __fname = "job_{}_clusterpunkte.txt".format(cur_job+1)
            with open(__path + __fname, "w") as f:
                # 2. DBSCAN
                f.write("Job {} - Clusterpunkte der zweiten DBSCAN Clusteranalyse: (x, y)\n".format(cur_job+1))
                                
                __clusterX = np.array(meanX)[~db_weak["noise_mask"]]
                __clusterY = np.array(meanY)[~db_weak["noise_mask"]]

                for _idx, _x in enumerate(__clusterX):
                    _y = __clusterY[_idx]
                    f.write("\n{},{}".format(_x, _y))
            
            # Cluster of weak dbscan -> Check by Admin -> "OK" & "NOK" possible
            cars_in_shds[cur_job]["dbscanWEAK"]["noise_mask"] = db_weak["noise_mask"]     
            cars_in_shds[cur_job]["dbscanWEAK"]["label"] = db_weak["labels"]
            cars_in_shds[cur_job]["dbscanWEAK"]["center"]["x"] = db_weak["centerX"]    
            cars_in_shds[cur_job]["dbscanWEAK"]["center"]["y"] = db_weak["centerY"]
            cars_in_shds[cur_job]["dbscanWEAK"]["cluster_core_idc"] = db_weak["cluster_core_idc"]   # -> Check again by admin
            cars_in_shds[cur_job]["dbscanWEAK"]["cluster_all_idc"] = db_weak["cluster_all_idc"]
            cars_in_shds[cur_job]["dbscanWEAK"]["n_clusters_"] = db_weak["n_clusters_"]
            cars_in_shds[cur_job]["dbscanWEAK"]["n_noise_"] = db_weak["n_noise_"]
            
            # Data to create .txt for interfaces
            
            cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"] = db_weak["cluster_all_idc"]
            # cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"] = db_weak["cluster_core_idc"]
            cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["mean"]["x"] = db_weak["centerX"]
            cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["mean"]["y"] = db_weak["centerY"]
                        
            plt.close("all")      
        
        print("\n---------------------------------------")
        print("Anzahl Crowdworker mit Achsenlänge von 0 = {}".format(counter_len_null))
        
        print("Zu kurze Achse -> Ausreißer: ")
        axis_too_short = []        
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            axis_too_short.append(len(cars_in_shds[cur_job]["removed"]["axis_too_short"]["start"]["x"]))
            print("Streifen {}: Gelöschte Ausreißer = {}".format(cur_job+1, len(cars_in_shds[cur_job]["removed"]["axis_too_short"]["start"]["x"])))
        print("Gesamt: gelöschte Erfassungen (länge<={}px) = {}".format(config["integration"]["minimal_length"], sum(axis_too_short)))
        
        print("\n-------------------------")
        print("1. DBSCAN result:")
        sum_cluster = []; sum_ausreißer=[]; sum_acquiCount = []
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            
            print("Streifen {}: Clusteranzahl = {}, Ausreißer/Rauschen = {}".format(cur_job+1, cars_in_shds[cur_job]["dbscan"]["n_clusters_"], cars_in_shds[cur_job]["dbscan"]["n_noise_"]))
            sum_cluster.append(cars_in_shds[cur_job]["dbscan"]["n_clusters_"])
            sum_ausreißer.append(cars_in_shds[cur_job]["dbscan"]["n_noise_"])
            sum_acquiCount.append(np.sum(cars_in_shds[cur_job]["acqui_count"]))
        print("Gesamt: Erfassungen = {}, Cluster = {}, Ausreißer = {}".format(np.sum(sum_acquiCount), np.sum(sum_cluster), np.sum(sum_ausreißer)))
        
        print("\n2. DBSCAN result:")
        sum_cluster = []; sum_ausreißer=[]; sum_acquiCount = []
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            print("Streifen {}: Clusteranzahl = {}, Ausreißer/Rauschen = {}".format(cur_job+1, cars_in_shds[cur_job]["dbscanWEAK"]["n_clusters_"], cars_in_shds[cur_job]["dbscanWEAK"]["n_noise_"]))
            sum_cluster.append(cars_in_shds[cur_job]["dbscanWEAK"]["n_clusters_"])
            sum_ausreißer.append(cars_in_shds[cur_job]["dbscanWEAK"]["n_noise_"])            
        print("Cluster = {}, Ausreißer = {}".format(np.sum(sum_cluster), np.sum(sum_ausreißer)))
        
        ## Integrate Car Parameters
        print("\n----------------------------------------------------------------\n")
        print("Integrating axis length")
        max_len_deviation = config["integration"]["max_len_deviation"]        
        
        print("max_len_deviation = {}".format(max_len_deviation))
        
        for cur_job in range(config["jobs"]["number_of_jobs"]):  # 1:number_of_jobs
            iteration = 0
            for cur_cluster in range(cars_in_shds[cur_job]["dbscan"]["n_clusters_"]):            
                abort = 0
                # cur_idcs = cars_in_shds[cur_job]["dbscan"]["cluster_core_idc"][cur_cluster]
                cur_idcs = cars_in_shds[cur_job]["dbscan"]["cluster_all_idc"][cur_cluster]
                while (abort != 1):
                    cur_center_x = cars_in_shds[cur_job]["dbscan"]["center"]["x"][cur_cluster]
                    cur_center_y = cars_in_shds[cur_job]["dbscan"]["center"]["y"][cur_cluster]
                    
                    cur_axis_len_list = [cars_in_shds[cur_job]["car_axis_len"][i] for i in cur_idcs]
                    
                    cur_axis_len = np.mean(cur_axis_len_list)
                    diff_axis_len = np.abs(cur_axis_len - cur_axis_len_list)
                    
                    log1 = diff_axis_len < max_len_deviation    # logical list: True when axis length difference is smaller than the allowed max value
                    
                    if (len(log1) > 2):   # Update cur_idcs
                        cur_idcs = np.array(cur_idcs)[log1].tolist()    # reduce idc list
                        # removed_idcs = np.array(cur_idcs)[~log1].tolist()
                    
                    # print(log1)
                    # print("np.min(log1)", np.min(log1))
                    if ((np.min(log1)) or (len(log1) <= 2)):     # log1: min==false, max==true --> When min(log1) == true
                        abort = 1
                        
                        # Store in main dict
                        cars_in_shds[cur_job]["dbscan"]["integrated_cluster_idc"].append(cur_idcs)
                        cars_in_shds[cur_job]["dbscan"]["integrated_params"]["center"]["x"].append(cur_center_x)
                        cars_in_shds[cur_job]["dbscan"]["integrated_params"]["center"]["y"].append(cur_center_y)
                        cars_in_shds[cur_job]["dbscan"]["integrated_params"]["axis_len"].append(cur_axis_len)
                        
                        # integrated_removed_idc = [ele for ele in cars_in_shds[cur_job]["dbscan"]["cluster_core_idc"][cur_cluster] if ele not in cur_idcs]
                        integrated_removed_idc = [ele for ele in cars_in_shds[cur_job]["dbscan"]["cluster_all_idc"][cur_cluster] if ele not in cur_idcs]
                        cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["idc"].extend(integrated_removed_idc)
                        cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["cluster_idc"].append(cur_cluster)
                        
                        # print("ABRUCH")
                    iteration += 1
                    
        # Clustering within DBSCAN-Clusters using robustly adaptive KMEANS
        print("\n----------------------------------------------------------------\n")
        print("Iterative clustering within DBSCAN clusters and detecting outliers\n")
        print("max_dist_2_integrated_line = {}\n".format(config["integration"]["max_dist_2_integrated_line"]))
        
        for cur_job in range(config["jobs"]["number_of_jobs"]): # 1:number_of_jobs
            iteration = 0
            for cur_cluster in range(len(cars_in_shds[cur_job]["dbscan"]["integrated_cluster_idc"])):
                abort = 0
                
                cur_idcs = cars_in_shds[cur_job]["dbscan"]["integrated_cluster_idc"][cur_cluster]
                orig_idcs = cur_idcs[:]
                
                while (abort != 1):
                    sx = [cars_in_shds[cur_job]["start"]["x"][i] for i in cur_idcs]                
                    sy = [cars_in_shds[cur_job]["start"]["y"][i] for i in cur_idcs]
                                    
                    ex = [cars_in_shds[cur_job]["end"]["x"][i] for i in cur_idcs]                
                    ey = [cars_in_shds[cur_job]["end"]["y"][i] for i in cur_idcs]
                    
                    sxex = sx[:]
                    sxex.extend(ex)                
                    syey = sy[:]
                    syey.extend(ey)
                    
                    # KMeans with 2 Clusters -> 1 Cluster car start, 1 Cluster car end
                    km = KMeans(n_clusters=2, max_iter=300, random_state=42)
                    km.fit(np.transpose([sxex, syey]))
                    
                    labels = km.labels_
                    cluster_center = km.cluster_centers_
                    n_iter_conv = km.n_iter_
                    
                    # print("Number of iterations till convergence = {}".format(n_iter_conv))
                    
                    mean_cluster_center = np.mean(cluster_center, axis=0)
                    
                    x1 = cluster_center[0,0]
                    x2 = cluster_center[1,0]
                    y1 = cluster_center[0,1]
                    y2 = cluster_center[1,1]
                    
                    # Check start point
                    x0 = np.array(sx)
                    y0 = np.array(sy)
                    
                    # Perpendicular distance of points (start/end point of acquisitions in cluster) to line between the 2 kmeans cluster center   https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html  28.01.2021
                    D_integrated_line = []
                    D_integrated_line.append(np.abs((x2-x1) * (y1-y0) - (x1-x0) * (y2-y1)) / np.sqrt((x2-x1)**2 + (y2-y1)**2))
                    
                    # Check end point
                    x0 = np.array(ex)
                    y0 = np.array(ey)
                    
                    D_integrated_line.append(np.abs((x2-x1) * (y1-y0) - (x1-x0) * (y2-y1)) / np.sqrt((x2-x1)**2 + (y2-y1)**2))
                    D_integrated_line = np.amax(D_integrated_line, axis=0)
                    
                    # Logical array: 1: < max_dist; 0: > max_dist
                    logD = D_integrated_line < config["integration"]["max_dist_2_integrated_line"]
                    
                    # Update active acquis by idcs
                    cur_idcs = np.array(cur_idcs)[logD].tolist()
                    if iteration > 100:
                        abort = 1
                        
                        cars_in_shds[cur_job]["kmeans"]["label"] = []
                        cars_in_shds[cur_job]["kmeans"]["center"] = []
                        cars_in_shds[cur_job]["kmeans"]["cluster_idc"] = []
                        cars_in_shds[cur_job]["kmeans"]["mean"]["x"] = mean_cluster_center[0]
                        cars_in_shds[cur_job]["kmeans"]["mean"]["y"] = mean_cluster_center[1]     
                        
                        # Remove idcs
                        removed_idc = list(set(orig_idcs)-set(cur_idcs))
                        if removed_idc:
                            cars_in_shds[cur_job]["removed"]["kmeans_outlier"]["idc"].extend(removed_idc)                    
                                                
                        print("Canceled")
                    
                    if (min(logD)>0):
                        abort = 1
                        cars_in_shds[cur_job]["kmeans"]["label"].append(labels)
                        cars_in_shds[cur_job]["kmeans"]["center"].append(cluster_center)
                        cars_in_shds[cur_job]["kmeans"]["cluster_idc"].append(cur_idcs)
                        
                        cars_in_shds[cur_job]["kmeans"]["int_line_start"]["x"].append(x1)
                        cars_in_shds[cur_job]["kmeans"]["int_line_start"]["y"].append(y1)
                        cars_in_shds[cur_job]["kmeans"]["int_line_end"]["x"].append(x2)
                        cars_in_shds[cur_job]["kmeans"]["int_line_end"]["y"].append(y2)
                        
                        cars_in_shds[cur_job]["kmeans"]["mean"]["x"].append(mean_cluster_center[0])
                        cars_in_shds[cur_job]["kmeans"]["mean"]["y"].append(mean_cluster_center[1])
                                        
                        # Remove idcs
                        removed_idc = list(set(orig_idcs)-set(cur_idcs))
                        if removed_idc:
                            cars_in_shds[cur_job]["removed"]["kmeans_outlier"]["idc"].extend(removed_idc)

                iteration += 1

        # Test check difference integrated before <-> after kmeans !?!?!? does nothing
        # for cur_job in range(config["jobs"]["number_of_jobs"]):
        #     # Cluster count
        #     len_km = len(cars_in_shds[cur_job]["kmeans"]["cluster_idc"])
        #     len_db = len(cars_in_shds[cur_job]["dbscan"]["integrated_cluster_idc"])
        #     try:
        #         len_km != len_db
        #     except ValueError:
        #         print("diff to integrated line is not useless")
        #     # Elements
        #     for idx, cur_km in enumerate(cars_in_shds[cur_job]["kmeans"]["cluster_idc"]):
        #         cur_db = cars_in_shds[cur_job]["dbscan"]["integrated_cluster_idc"][idx]
        #         try:
        #             cur_db == cur_km
        #         except ValueError:
        #             print("diff to integrated line is not useless")

        ## Filter Uncertain Cluster: Need to be checked by an admin    
        print("------------------------------------")     
        print("\nStart filtering with minpts and ellipse parameters")  
        for cur_job in range(config["jobs"]["number_of_jobs"]): # 1:number_of_jobs
            # if cur_job<4:
            #    continue
            print("\nCurrent Job {}".format(cur_job))  
            print("-------------")
            # cur_img = plt.imread(config.directories["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
            # height, width = cur_img.shape
            # extent = [0.5, width+0.5, height+0.5, 0.5]  # Adjust coordinate origin (HTML Canvas = Matlab vs Python)
            
            # fig, ax = plt.subplots(111)
            # plt.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
            
            # Calculate error ellipse
            n_std = 3.0  # 3.0 # Number of standard deviation to determine the ellipse's radiuses -> 1.0 -> 68%, 2.0 -> 95%, 3.0 -> 98%
            # distance from mean value mu e.g. 
            # mu - 1 sigma
            # mu - 2 sigma
            # mu - 3 sigma

            # minpts_threshold = 5
            config["integration"]["minpts_threshold_ellipse_1"]
            plot = True
            savePlot = True     
            method_contains_points = False      # True: uncertain acquisitions == outside of the ellipsoid; False: uncertain acquisitions/whole clusters == pearson~0 and std >> 0

            for cur_cluster_idx, cur_acqui_idc in enumerate(cars_in_shds[cur_job]["kmeans"]["cluster_idc"]):
                # print("Plot next error ellipse")
                # cur_acqui_idc # acqui idcs for current cluster
                
                # else:
                #     print(">{} Acquisitions, Cluster {} is most likely a cluster".format(minpts_threshold, cur_cluster_idx))
                
                meanX = np.array(cars_in_shds[cur_job]["mean"]["x"])[cur_acqui_idc]   # Mean value of acquisitions of current cluster
                meanY = np.array(cars_in_shds[cur_job]["mean"]["y"])[cur_acqui_idc]
                
                ## https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
                # if cur_job == 4:
                #     print("cluster_idx = {}, cur_acqui_idc = {}".format(cur_cluster_idx, cur_acqui_idc))
                #     print("meanX = {}, meanY = {}".format(meanX, meanY))
                    
                cov = np.cov(meanX, meanY) # Covariance
                std = np.sqrt(cov.diagonal()) # Standard deviation
                
                std_mean = np.sqrt(sum(cov.diagonal()))  # mean standard deviation https://de.wikipedia.org/wiki/Fehlerellipse -> Mittlerer Maximalfehler
                
                mu_x = np.mean(meanX) # Center of ellipse
                mu_y = np.mean(meanY)
                
                if plot:
                    fig, (ax1, ax2) = plt.subplots(1,2)   # ax1 -> acquisitions, ax2 -> error ellipse
                #functionLibrary.confidence_ellipse(meanX, meanY, ax, n_std=1,
                #    label=r'$1\sigma$', edgecolor='firebrick')
                #functionLibrary.confidence_ellipse(meanX, meanY, ax, n_std=2,
                #                label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
                #functionLibrary.confidence_ellipse(meanX, meanY, ax, n_std=3,
                #                label=r'$3\sigma$', edgecolor='blue', linestyle=':')    
                
                # Compute ellipse
                ellipse, pearson = confidence_ellipse(meanX, meanY, ax2, n_std=n_std, label="{:.0f}".format(n_std)+r'$\sigma$',  edgecolor="red", alpha=0.3, facecolor='pink', zorder=0)
                # Get mask of points located inside of the ellipse
                mask = ellipse.contains_points(ax2.transData.transform(np.transpose([meanX, meanY])))
                
                if plot:
                    title = "Streifen {}, Cluster {}, Korr(X,Y) = {:.2f}, $\sigma_x$ = {:.2f}, $\sigma_y$ = {:.2f}".format(cur_job+1, cur_cluster_idx+1, pearson, std[0], std[1])
                    title += ", $\sigma_{max} = \sqrt{\sigma_x^2 + \sigma_y^2} $" + " = {:.2f}".format(std_mean)
                    fig.suptitle(title)
                    # Acquisitions
                    cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
                    
                    height, width = cur_img.shape    
                    extent = [-1, width-1, height-1, -1, ]
                    # extent = [-1, width-1, height-1, -1]  # Adjust coordinate origin (HTML Canvas = Matlab vs Python)
                    ax1.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none', zorder=0) #, vmin=0, vmax=255)                    
                    
                    xmin = 50000
                    ymin = 50000
                    xmax = 0
                    ymax = 0
                    for cur_acqui in cur_acqui_idc:
                        x = np.array(cars_in_shds[cur_job]["start"]["x"])[cur_acqui]
                        y = np.array(cars_in_shds[cur_job]["start"]["y"])[cur_acqui]
                        
                        x2 = np.array(cars_in_shds[cur_job]["end"]["x"])[cur_acqui]
                        y2 = np.array(cars_in_shds[cur_job]["end"]["y"])[cur_acqui]
                        
                        if x < xmin:
                            xmin = x
                        if x2 < xmin:
                            xmin = x2
                        if x > xmax:
                            xmax = x
                        if x2 > xmax:
                            xmax = x2
                        if y < ymin:
                            ymin = y
                        if y2 < ymin:
                            ymin = y2
                        if y > ymax:
                            ymax = y
                        if y2 > ymax:
                            ymax = y2                        
                        ax1.plot([x, x2],
                                 [y, y2], color="red", linewidth=2, zorder=1)
                    
                    ax1.scatter(meanX, meanY, c="blue", s=25, zorder=2)   # Middle points of acquis
                    ax1.plot([], [], color="red", label=r'Erfassung')  # empty only legend label
                    ax1.plot([], [], c="blue", label=r'Mittelpunkt einer Erfassung')  # empty only legend label
                    # ax1.scatter([],[], color="blue", s=25, label=r'Mittelpunkt der Erfassung')
                    
                    ax1.axis([xmin-20, xmax+20, ymax+20, ymin-20])          # flipped    , origin upper left corner
                    ax1.set_title("Erfassung")
                    ax1.set_xlabel("x [px]")
                    ax1.set_ylabel("y [px]")
                    ax1.legend()
                    
                    # Error ellipse
                    ax2.add_patch(ellipse)
                    ax2.scatter(meanX, meanY, c='blue', s=25, label=r'Mittelpunkt einer Erfassung')   # r'$\overline{X_i}, \overline{Y_i}$')
                    ax2.scatter(mu_x, mu_y, c='red', s=25, marker="D", label=r'Mittelpunkt der Ellipse')
                    
                    ax2.invert_yaxis()
                    ax2.set_xlabel("x [px]")
                    ax2.set_ylabel("y [px]")
                    ax2.legend()                
                    ax2.set_title("Fehlerellipse")
                    
                    # plt.show()
                    fig = plt.gcf()
                    #manager = plt.get_current_fig_manager()
                    #manager.resize(*manager.window.maxsize())
                    figManager = plt.get_current_fig_manager() # for fullscreen
                    figManager.window.state("zoomed")
                    
                    if savePlot:
                        fig.savefig('figures/{}/job_{}_cluster_{}_ellipse_acquisitions.png'.format(cur_job+1, cur_job+1, cur_cluster_idx+1))
                        plt.close("all")
                    
                    # Savedata
                    # Write outlier and clusterpoints to file for CD       
                    __path =  "plotted_data_textformat/cluster_nachuntersuchung__ellipse_params/"
                    create_dir_if_not_exist(__path)
                    __fname = "job_{}_cluster_{}_bestimmte_ellipse_parameter.txt".format(cur_job+1, cur_cluster_idx+1)
                    with open(__path + __fname, "w") as f:
                        # 2. DBSCAN
                        f.write("Job {} - Parameter der Fehlerellipse und Koordinaten der Erfassungen, die Teil von Clustern sind\n".format(cur_job+1))
                        
                        f.write("mu_x = {}, mu_y = {}".format(mu_x, mu_y))
                        f.write("\n\nKoordinaten der Mittelpunkte der Erfassungen des Clusters (x_mean, y_mean)\n")
                        for __idx, __acqui_meanX in enumerate(meanX):
                            __acqui_meanY = meanY[__idx]                            
                            f.write("\n{},{}".format(__acqui_meanX, __acqui_meanY))
                        
                        f.write("\n\nAnfangs- und Endpunktkoordinaten der Erfassungen des Clusters (x_start, y_start, x_ende, y_ende)")
                        for cur_acqui in cur_acqui_idc:
                            x = np.array(cars_in_shds[cur_job]["start"]["x"])[cur_acqui]
                            y = np.array(cars_in_shds[cur_job]["start"]["y"])[cur_acqui]
                            
                            x2 = np.array(cars_in_shds[cur_job]["end"]["x"])[cur_acqui]
                            y2 = np.array(cars_in_shds[cur_job]["end"]["y"])[cur_acqui]
                            
                            f.write("\n{},{},{},{}".format(x, y, x2, y2))
                
                if method_contains_points:  # Not fully implemented yet
                    ## Outside Acquisitions
                    fig2, (ax3, ax4) = plt.subplots(1,2)
                    fig2.suptitle("Job {}, Cluster {}, Korr(X,Y) = {:.2f}, $\sigma_x$ = {:.2f}, $\sigma_y$ = {:.2f}".format(cur_job+1, cur_cluster_idx+1, pearson, std[0], std[1]))
                    
                    ## Compute ellipse
                    ellipse, pearson = confidence_ellipse(meanX, meanY, ax3, n_std=n_std, label=str(n_std)+r'$\sigma$', edgecolor="red", alpha=0.3, facecolor='pink', zorder=0)
                    # Get mask of points located inside of the ellipse
                    mask = ellipse.contains_points(ax3.transData.transform(np.transpose([meanX, meanY])))
                    
                    # Plot mean x,y of acquisitions
                    if plot:                    
                        ax3.add_patch(ellipse)
                        ax3.scatter(meanX[mask], meanY[mask], c='blue', s=25, label=r'Inside: $\overline{X_i}, \overline{Y_i}$')
                    
                    if ~mask.min(): # Points outside of ellipse
                        if plot:
                            ax3.scatter(meanX[~mask], meanY[~mask], marker="o", facecolors="blue", edgecolors='red', c='blue', s=25, label=r'Outside: $\overline{X}, \overline{Y}$')
                            
                            # Create subplot for acquisitions outside of ellipse
                            # Acquisitions
                            cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
                            
                            height, width = cur_img.shape    
                            extent = [-1, width-1, height-1, -1]   # Adjust coordinate origin (HTML Canvas = Matlab vs Python)
                            ax4.imshow(cur_img, cmap="gray", origin="upper", extent=extent, interpolation='none') #, vmin=0, vmax=255)
                            
                            xmin = 50000
                            ymin = 50000
                            xmax = 0
                            ymax = 0
                            for cur_acqui in np.array(cur_acqui_idc)[~mask]:
                                    
                                x = np.array(cars_in_shds[cur_job]["start"]["x"])[cur_acqui]
                                y = np.array(cars_in_shds[cur_job]["start"]["y"])[cur_acqui]
                                
                                x2 = np.array(cars_in_shds[cur_job]["end"]["x"])[cur_acqui]
                                y2 = np.array(cars_in_shds[cur_job]["end"]["y"])[cur_acqui]
                                
                                if x < xmin:
                                    xmin = x
                                if x2 < xmin:
                                    xmin = x2
                                if x > xmax:
                                    xmax = x
                                if x2 > xmax:
                                    xmax = x2
                                if y < ymin:
                                    ymin = y
                                if y2 < ymin:
                                    ymin = y2
                                if y > ymax:
                                    ymax = y
                                if y2 > ymax:
                                    ymax = y2
                                ax4.plot([x, x2],
                                         [y, y2], color="red", linewidth=2)
                            ax4.plot([], [], color="red", label=r'Acquisition')  # empty only legend label
                            ax4.axis([xmin-20, xmax+20, ymin-20, ymax+20])
                            ax4.set_title("Acquisitions outside of ellipse")
                            ax4.legend()
                        
                        # Save
                        # cars_in_shds[cur_job]["kmeans"]["uncertain_acqui"]["cluster_idc"].append(cur_cluster_idx)
                        # cars_in_shds[cur_job]["kmeans"]["uncertain_acqui"]["acqui_idc"].append(np.array(cur_acqui_idc)[~mask])
                        
                        cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"].append(cur_cluster_idx)
                        cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["acqui_idc"].append(np.array(cur_acqui_idc)[~mask])
                        
                        print("Outside of ellipse, Cluster {}, acquisition {} needs to be checked by admin".format(cur_cluster_idx, np.array(cur_acqui_idc)[~mask]))
                    
                    if plot:
                        ax3.scatter(mu_x, mu_y, c='red', s=25, marker="D", label=r'$\mu_x, \mu_y$')
                        ax3.set_title("Job {}, Fehlerellipse zu Cluster {}, Korr(X,Y) = {:.2f}, $\sigma_x$ = {:.2f}, $\sigma_y$ = {:.2f}".format(cur_job+1, cur_cluster_idx+1, pearson, std[0], std[1]))
                        ax3.set_xlabel("x [px]")
                        ax3.set_ylabel("y [px]")
                        ax3.legend() 
                        
                        fig = plt.gcf()
                        
                        # manager = plt.get_current_fig_manager()
                        # manager.resize(*manager.window.maxsize())
                        # figManager = plt.get_current_fig_manager() # for fullscreen
                        # figManager.window.state("zoomed")
                        
                        if savePlot:
                            fig2.savefig('figures/{}/job_{}_cluster_{}_ellipse_contains.png'.format(cur_job+1, cur_job+1, cur_cluster_idx+1))
                            plt.close("all")
                else:   # Threshold method: whole cluster to be checked by the admin
                    # print("Cluster {}: pearson = {}, std_mean = {}".format(cur_cluster_idx, pearson, std_mean))
                    
                    # Check if there are enough acquisitions in specific cluster
                    pts_in_cluster = len(cur_acqui_idc)
                    if pts_in_cluster <= config["integration"]["minpts_threshold_ellipse_1"]:
                        print("<{} Acquisitions, Cluster {} needs to be checked by admin".format(config["integration"]["minpts_threshold_ellipse_1"], cur_cluster_idx))
                        # cars_in_shds[cur_job]["kmeans"]["uncertain"]["cluster_idc"].append(cur_cluster_idx)
                        cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"].append(cur_cluster_idx)
                        cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["acqui_idc"].append(np.array(cur_acqui_idc)) # Whole cluster with all acquisitions to be checked
                        continue
                    
                    # Check "size" of ellipse                    
                    minpts_thresh = config["integration"]["minpts_threshold_ellipse_2"]
                    std_thresh = config["integration"]["std_threshold"]
                    if pts_in_cluster < minpts_thresh and std_mean > std_thresh:    # abs(pearson) < 0.1 or (abs(std) > 2).max():  # pearson == 0 -> circle -> no correlation, or stdx, stdy > 2

                        print("cluster uncertain -> check cluster {}, pts_in_cluster={}, std_mean={:.2f}".format(cur_cluster_idx, pts_in_cluster, std_mean))
                        # cars_in_shds[cur_job]["kmeans"]["uncertain_cluster"]["cluster_idc"].append(cur_cluster_idx)
                        # cars_in_shds[cur_job]["kmeans"]["uncertain_cluster"]["acqui_idc"].append(np.array(cur_acqui_idc))
                        
                        cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"].append(cur_cluster_idx)
                        cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["acqui_idc"].append(np.array(cur_acqui_idc))
                        
                    else: # Rest "OK" clusters
                        cars_in_shds[cur_job]["kmeans"]["OK_int_cluster_idc"].append(cur_cluster_idx)

        print("\n--------------")
        print("Ausreißer durch Mittelbildung (bestehende Integration):")
        ausreisser = []; ausreisser_kmeans = []
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            ausreisser.append(len(cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["idc"]))
            print("Streifen {}: {} Ausreißer (Abstand zur mittleren Fahrzeuglänge zu groß)".format(cur_job+1, len(cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["idc"])))

            ausreisser_kmeans.append(len(cars_in_shds[cur_job]["removed"]["kmeans_outlier"]["idc"]))
            print("Streifen {}: {} Ausreißer (Abstand zur integrierten Fahrzeugachse zu groß (KMeans))".format(cur_job+1,len(cars_in_shds[cur_job]["removed"]["kmeans_outlier"]["idc"])))

        print("Durch Mittelbildung eliminierte Ausreißer = {} (Abstand zur mittleren Fahrzeuglänge zu groß)".format(np.sum(ausreisser)))
        print("Durch Mittelbildung eliminierte Ausreißer = {} (Abstand zur integrierten Fahrzeugachse zu groß (KMeans))".format(np.sum(ausreisser_kmeans)))
        ## Visualize integrated Cars    
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            # Result of kmeans integration
            cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
            height, width = cur_img.shape    
            extent = [-1, width-1, height-1, -1]   # Adjust coordinate origin (HTML Canvas = Matlab vs Python)

            fig, ax = plt.subplots(1,1)
            ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
            
            for cur_car in range(len(cars_in_shds[cur_job]["kmeans"]["center"])):
                cur_center = cars_in_shds[cur_job]["kmeans"]["center"][cur_car]                
                ax.plot([ cur_center[0][0], cur_center[1][0] ], [ cur_center[0][1], cur_center[1][1] ], linestyle="-", color="blue", linewidth=1)

            #ax.set_title("Streifen {}, integrierte Fahrzeuge".format(cur_job+1))
            ax.axis("off")
            
            ax.plot([],[],  linestyle='-', color='blue', label='Integriertes Fahrzeug', linewidth=1)
            ax.legend(fancybox=True, shadow=True, handlelength=1, loc='upper center', bbox_to_anchor=(0.5, 0)) #loc='center left', bbox_to_anchor=(1, 0.5))
            #Integrierte_Erfassung = Line2D([0], [0], linestyle='-', color='red', label='Erfassung',linewidth=1)  
            #ax.legend(handles=[Integrierte_Erfassung], loc='upper center', bbox_to_anchor=(0.5, 0)) #loc='center left', bbox_to_anchor=(1, 0.5))

            fig = plt.gcf()
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            path = 'figures/{}/job_{}_kmeans_integrated_cars.png'.format(cur_job+1, cur_job+1)
            fig.savefig(path, format="png" , dpi=300, bbox_inches="tight")
            
            plt.close("all")  
            
            # Write integrated cars to file for CD       
            __path =  "plotted_data_textformat/integrierte_fahrzeuge/"
            create_dir_if_not_exist(__path)
            __fname = "job_{}_integrierte_fahrzeuge.txt".format(cur_job+1)
            with open(__path + __fname, "w") as f:
                # Integrierte Fahrzeuge
                f.write("Job {} - Integrierte Fahrzeugkoordinaten (x_start, y_start, x_ende, y_ende)\n".format(cur_job+1))
                for __cur_car in range(len(cars_in_shds[cur_job]["kmeans"]["center"])):
                    __cur_center = cars_in_shds[cur_job]["kmeans"]["center"][__cur_car]           
                    f.write("\n{},{},{},{}".format(__cur_center[0][0], __cur_center[1][0],  __cur_center[0][1], __cur_center[1][1]))
            
            ## DBSCAN
            ##plt.close("all")
            #fig2 = plt.figure()    
            #plt.subplot(2,1,1)
            #plt.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')    
            #x_noise = np.array(cars_in_shds[cur_job]["mean"]["x"])[~core_samples_mask]    # only noise
            #y_noise = np.array(cars_in_shds[cur_job]["mean"]["y"])[~core_samples_mask]
            #plt.plot(x_noise, y_noise , 'xr', markersize=12, linewidth=3)
            #    
            #x_cluster = np.array(cars_in_shds[cur_job]["mean"]["x"])[core_samples_mask]     # All Cluster points without noise
            #y_cluster = np.array(cars_in_shds[cur_job]["mean"]["y"])[core_samples_mask]
            #plt.plot(x_cluster, y_cluster, "xg", markersize=12, linewidth=3)
            #    
            #plt.plot(cars_in_shds[cur_job]["dbscan"]["center"]["x"], cars_in_shds[cur_job]["dbscan"]["center"]["y"], "*c", markersize=4, linewidth=3)  # Cluster center
            #
            #plt.axis('off')
            #plt.title("Visu 1: Outliers detected with DBSCAN")    
            ## All DBSCAN clusters
            #plt.subplot(2,1,2)
            #plt.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
            #
            #plt.plot(x_cluster, y_cluster, "xg", markersize=12, linewidth=3)
            #plt.plot(cars_in_shds[cur_job]["dbscan"]["center"]["x"], cars_in_shds[cur_job]["dbscan"]["center"]["y"], "*c", markersize=4, linewidth=3)  # Cluster center
            #plt.axis('off')
            #plt.title("Visu 2: DBSCAN Clusters")
            #plt.show(block=False)
        
            # Outliers removed through other means
            # Axis too short
            #fig2 = plt.figure()
            fig, ax = plt.subplots(1,1)
            ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none') 
            
            #t = cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["idc"]
            #integrated_axis_idc = [item for sublist in t for item in sublist] # Flatten list to 1D        
            integrated_axis_idc = cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["idc"]
            
            removed_mean_x = [cars_in_shds[cur_job]["mean"]["x"][i] for i in integrated_axis_idc]
            removed_mean_y = [cars_in_shds[cur_job]["mean"]["y"][i] for i in integrated_axis_idc] 
            
            removed_x = [cars_in_shds[cur_job]["start"]["x"][i] for i in integrated_axis_idc]
            removed_y = [cars_in_shds[cur_job]["start"]["y"][i] for i in integrated_axis_idc]
            removed_x2 = [cars_in_shds[cur_job]["end"]["x"][i] for i in integrated_axis_idc]
            removed_y2 = [cars_in_shds[cur_job]["end"]["y"][i] for i in integrated_axis_idc]  
            
            integrated_x = []; integrated_y = []; integrated_x2 = []; integrated_y2 = []
            for cur_car in cars_in_shds[cur_job]["kmeans"]["cluster_idc"]:
                integrated_x.extend([cars_in_shds[cur_job]["start"]["x"][i] for i in cur_car])
                integrated_y.extend([cars_in_shds[cur_job]["start"]["y"][i] for i in cur_car])
                integrated_x2.extend([cars_in_shds[cur_job]["end"]["x"][i] for i in cur_car])
                integrated_y2.extend([cars_in_shds[cur_job]["end"]["y"][i] for i in cur_car])

                #cur_center = cars_in_shds[cur_job]["kmeans"]["center"][cur_car]                            
                #ax.plot([ cur_center[0][0], cur_center[1][0] ], [ cur_center[0][1], cur_center[1][1] ], "r", color="red", linewidth=2)
            
            col_ausreißer = "red"; col_integriert = "blue"; linewidth = 1            
            ax.plot([integrated_x, integrated_x2], [integrated_y, integrated_y2], '-', color=col_integriert, linewidth=linewidth)
            ax.plot([removed_x, removed_x2], [removed_y, removed_y2], '-', color=col_ausreißer, linewidth=linewidth)
            #ax.plot(removed_mean_x, removed_mean_y , 'xr', markersize=6, linewidth=linewidth)
            
            ax.plot([],[], '-', color=col_ausreißer, linewidth=linewidth, label="Ausreißer")
            ax.plot([],[], '-', color=col_integriert, linewidth=linewidth, label="Integrierte Erfassung")
            #core_patch = Line2D([0], [0], marker='xr', linestyle="", color='cyan', label='Kernpunkt', markersize=15)            
            ax.legend(fancybox=True, shadow=True, handlelength=1, loc='upper center', bbox_to_anchor=(0.5, 0))    #loc='center left', bbox_to_anchor=(1, 0.5))
            
            ax.axis('off')
            ax.set_title("Streifen {}, eliminierte Ausreißer={}".format(cur_job+1, len(cars_in_shds[cur_job]["removed"]["dbscan_integrated_axis_len"]["idc"])+len(cars_in_shds[cur_job]["removed"]["kmeans_outlier"]["idc"])))
            fig = plt.gcf()
            fig.savefig('figures/{}/job_{}_dbscan_integrated_axis_len.png'.format(cur_job+1, cur_job+1), dpi=300, bbox_inches="tight")
            plt.close("all")  
            
            # Write filtered outlier in clusters to file for CD
            __path =  "plotted_data_textformat/cluster_nachuntersuchung__eliminierte_ausreisser_integrierte_erfassungen/"
            create_dir_if_not_exist(__path)
            __fname = "job_{}_eliminierte_ausreisser.txt".format(cur_job+1)
            with open(__path + __fname, "w") as f:
                # Eliminierte Ausreisser
                f.write("Job {} \nEliminierte Ausreisser innerhalb der mit dem ersten DBSCAN gefundenen Cluster (x_start, y_start, x_ende, y_ende)\n".format(cur_job+1))
                for __idx, __removed_x in enumerate(removed_x):
                    __removed_y  = np.array(removed_y)[__idx]
                    __removed_x2 = np.array(removed_x2)[__idx] 
                    __removed_y2 = np.array(removed_y2)[__idx]    
                    f.write("\n{},{},{},{}".format(__removed_x, __removed_x2, __removed_y, __removed_y2))
            
            __fname = "job_{}_integrierte_erfassungen.txt".format(cur_job+1)
            with open(__path + __fname, "w") as f:
                # Integrierte Erfassungen/Fahrzeuge
                f.write("\n\nIntegrierte Erfassungen/Fahrzeuge (x_start, y_start, x_ende, y_ende)\n".format(cur_job+1))
                for __idx, __integrated_x in enumerate(integrated_x):
                    __integrated_y = np.array(integrated_y)[__idx]
                    __integrated_x2 = np.array(integrated_x2)[__idx] 
                    __integrated_y2 = np.array(integrated_y2)[__idx]
                    f.write("\n{},{},{},{}".format(__integrated_x, __integrated_x2, __integrated_y, __integrated_y2))
            
            fig, ax = plt.subplots(1,1)
            ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none') 
            
            integrated_cluster_center_x = cars_in_shds[cur_job]["kmeans"]["mean"]["x"]
            integrated_cluster_center_y = cars_in_shds[cur_job]["kmeans"]["mean"]["y"]
            colorlist = ["r","g","b","c","m","y"]    
            
            count = len(integrated_cluster_center_x)
            idx = 0
            for c in range(count):            
                col = colorlist[idx]                
                ax.plot(integrated_cluster_center_x[c], integrated_cluster_center_y[c], color=col, marker='o', label="Mittlere Cluster Zentren")
                idx += 1
                if idx >= len(colorlist):
                    idx = 0   
            ax.axis('off')            
            ax.set_title("Streifen {}, integrierte Clusterzentren".format(cur_job+1))
            
            fig = plt.gcf()
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            fig.savefig('figures/{}/job_{}_integrated_cluster_center.png'.format(cur_job+1, cur_job+1), dpi=300, bbox_inches="tight")
            #plt.show(block=False)     
                   
            ################################ Zoomed in plots
            # Zoom Areas
            if cur_job == 0:
                x_min = 300; x_max = 550       
                y_min = 350; y_max = 499                
                #x_min = 1365; x_max = 1410  # Small car top center strip 1
                #y_min = 0; y_max = 45
            if cur_job == 1:
                x_min = 778; x_max = 778+163
                y_min = 95; y_max = 95+150
            if cur_job == 2:
                x_min = 1283; x_max = 1283+185
                y_min = 4; y_max = 4+150
            if cur_job == 3:
                x_min = 2332; x_max = 2332+209
                y_min = 54; y_max = 4+209
            if cur_job == 4:
                x_min = 1812; x_max = 1812+172
                y_min = 247; y_max = 247+141
            if cur_job == 5:
                x_min = 421; x_max = 421+217
                y_min = 252; y_max = 252+200
            
            # Zoomed in for strip 1
            #if cur_job == 0:               

            ## Integrated Cars
            try:
                cur_img = plt.imread(config["directories"]["Img_Folder"] + 'job' + str(cur_job + 1) + '/' + 'shd.png')
                
                f1, (ax1, ax2) = plt.subplots(2)
                        
                # Plot image
                col_integrated = "blue"
                height, width = cur_img.shape
                extent = [-1, width-1, height-1, -1 ]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-
                ax2.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')  
                
                # Plot integrated acquis
                for cur_car in range(len(cars_in_shds[cur_job]["kmeans"]["center"])):
                    cur_center = cars_in_shds[cur_job]["kmeans"]["center"][cur_car]                
                    ax2.plot([ cur_center[0][0], cur_center[1][0] ], [ cur_center[0][1], cur_center[1][1] ], "-", color=col_integrated, linewidth=1)
                
                # Plot image
                ax1.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
                
                # Plot integrated acquis
                for cur_car in range(len(cars_in_shds[cur_job]["kmeans"]["center"])):
                    cur_center = cars_in_shds[cur_job]["kmeans"]["center"][cur_car]                
                    ax1.plot([ cur_center[0][0], cur_center[1][0] ], [ cur_center[0][1], cur_center[1][1] ], "-", color=col_integrated, linewidth=1)

                ax1.set_xlim(x_min, x_max)
                ax1.set_ylim(y_max, y_min)

                ax1.axis("off")
                ax2.axis("off")

                ax1.plot([],[], '-', color=col_integriert, linewidth=linewidth, label="Integriertes Fahrzeug")
                ax1.legend(fancybox=True, shadow=True, handlelength=1, loc='upper center', bbox_to_anchor=(0.5, 0)) #loc='center left', bbox_to_anchor=(1, 0.5))
                #acqui_patch = Line2D([0], [0], linestyle='-', color='red', label='Integrierte Erfassung',linewidth=1)               
                #ax1.legend(handles=[acqui_patch], fancybox=True, shadow=True, handlelength=1, loc='center left', bbox_to_anchor=(1, 0.5))

                ax1.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
                ax2.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
                
                fig1 = plt.gcf()
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                if savePlot:
                    path = config["directories"]["Figures"] +'{}/'.format(cur_job+1)
                    create_dir_if_not_exist(path)
                    
                    fname = 'job_{}_kmeans_integrated_cars_ZOOMED.png'.format(cur_job+1)
                    path += fname
                    
                    fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")       # , bbox_inches="tight" -> save legend to plot
                    plt.close("all") 
            except:
                print("Error plotting integrated cars with zoom")
                
            ## Eliminated acquis by other means
            try:
                cur_img = plt.imread(config["directories"]["Img_Folder"] + 'job' + str(cur_job + 1) + '/' + 'shd.png')
                
                f1, (ax1, ax2) = plt.subplots(2)
                        
                # Plot image
                height, width = cur_img.shape
                extent = [-1, width-1, height-1, -1 ]    # Account for different coordinate origin: HTML Canvas(0,0) -> set to (-
                ax2.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')  
                
                # Plot integrated acquis
                col_ausreißer = "red"; col_integriert = "blue"; linewidth = 1            
                ax2.plot([integrated_x, integrated_x2], [integrated_y, integrated_y2], '-', color=col_integriert, linewidth=linewidth)       
                ax2.plot([removed_x, removed_x2], [removed_y, removed_y2], '-', color=col_ausreißer, linewidth=linewidth)
                #ax2.plot(removed_mean_x, removed_mean_y , 'xr', markersize=6, linewidth=linewidth)
                
                # Plot image
                ax1.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
                
                # Plot integrated acquis
                ax1.plot([integrated_x, integrated_x2], [integrated_y, integrated_y2], '-', color=col_integriert, linewidth=linewidth)       
                ax1.plot([removed_x, removed_x2], [removed_y, removed_y2], '-', color=col_ausreißer, linewidth=linewidth)
                #ax1.plot(removed_mean_x, removed_mean_y , 'xr', markersize=6, linewidth=linewidth)
                ax1.plot([],[], '-', color=col_ausreißer, linewidth=linewidth, label="Ausreißer")
                ax1.plot([],[], '-', color=col_integriert, linewidth=linewidth, label="Integrierte Erfassung")
                
                ax1.set_xlim(x_min, x_max)
                ax1.set_ylim(y_max, y_min)

                ax1.axis("off")
                ax2.axis("off")
                              
                ax1.legend(fancybox=True, shadow=True, handlelength=1, loc='upper center', bbox_to_anchor=(0.5, 0))

                ax1.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
                ax2.add_patch(Rectangle([x_min, y_max], x_max-x_min, y_min-y_max, fill=False, linestyle="-", linewidth=1, color="orange"))
                
                fig1 = plt.gcf()
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                if savePlot:
                    path = config["directories"]["Figures"] + '{}/'.format(cur_job+1)
                    create_dir_if_not_exist(path)
                    
                    fname = 'job_{}_dbscan_integrated_axis_len_ZOOMED.png'.format(cur_job+1)
                    path += fname
                    
                    fig1.savefig(path, format='png', dpi=300, bbox_inches="tight")       # , bbox_inches="tight" -> save legend to plot
                    plt.close("all") 
            except:
                print("Error plotting integrated cars with zoom")

        #plt.show()
        plt.close("all")    

        ## PLOT END ##############################################################################

        # Save variables to .dat file
        SAVE_OUTPUT = True
        if SAVE_OUTPUT is True:
            print("Saving Output in backup folder")

            config["backup"]["cars_in_shds_pre_verification"]

            # Write cars_in_shds to file
            with open(config["backup"]["cars_in_shds_pre_verification"], 'wb') as file:  
                obj = (cars_in_shds)
                pickle.dump(obj, file)            

    ## Output
    print("\n---------------------------------")
    print("Korrekte Cluster VOR Ellipsenkriterium")
    
    cluster_korrekt = []; erfassung_korrekt_sum = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        cluster_korrekt.append(len(cars_in_shds[cur_job]["kmeans"]["cluster_idc"]))
        erfassung_korrekt = []
        for ok_cluster in cars_in_shds[cur_job]["kmeans"]["cluster_idc"]:
            erfassung_korrekt.append(len(ok_cluster))
        erfassung_korrekt_sum.append(np.sum(erfassung_korrekt))
        print("Streifen {}: Cluster = {}, Erfassungen = {}".format(cur_job +1, cluster_korrekt[cur_job], np.sum(erfassung_korrekt)))
    
    print("Gesamt: Korrekte Cluster  = {}, Korrekte Erfassungen = {}".format(np.sum(cluster_korrekt), np.sum(erfassung_korrekt_sum)))
    
    print("\n---------------------------------")
    print("Korrekte Cluster nach Ellipsenkriterium")
    cluster_korrekt = []; erfassung_korrekt_sum = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        cluster_korrekt.append(len(cars_in_shds[cur_job]["kmeans"]["OK_int_cluster_idc"]))
        
        erfassung_korrekt = []
        for ok_cluster in cars_in_shds[cur_job]["kmeans"]["OK_int_cluster_idc"]:
            erfassung_korrekt.append(len(cars_in_shds[cur_job]["kmeans"]["cluster_idc"][ok_cluster]))
        erfassung_korrekt_sum.append(np.sum(erfassung_korrekt))
        print("Streifen {}: Cluster = {}, Erfassungen = {}".format(cur_job, cluster_korrekt[cur_job], np.sum(erfassung_korrekt)))
    
    print("Gesamt: Korrekte Cluster  = {}, Korrekte Erfassungen = {}".format(np.sum(cluster_korrekt), np.sum(erfassung_korrekt_sum)))
    
    # ------------------------------------------------------------------------------------------------------
    #
    # Organize results and write to .txt: Admininterface -> db.txt & ell.txt;   Crowdinterface -> db+ell combined.txt 
    #
    # ------------------------------------------------------------------------------------------------------
    try:
        # .txt for the Admininterface
        path = "Admininterface/Pre Rating/"
        create_dir_if_not_exist(path)    
        
        # Clear directory
        print("Clear admininterface pre rating folder")
        for f in glob.glob("Admininterface/Pre Rating/*.txt"):                
            try:
                os.remove(f)
            except:
                print("couldnt delete file: {}".format(f))
        print("Clear admininterface post rating folder")
        for f in glob.glob("Admininterface/Post Rating/*.txt"):                
            try:
                os.remove(f)
            except:
                print("couldnt delete file: {}".format(f))
    
        print("Clear crowdinterface pre rating folder")
        for f in glob.glob("Crowdinterface/Pre Rating/*.txt"):    
            try:
                os.remove(f)
            except:
                print("couldnt delete file: {}".format(f))
        
        dataToCheck = {}
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            dataToCheck[cur_job] = {
                "ell": {
                    "start": {"x": [], "y": []},
                    "end": { "x": [], "y": [] },
                    "mean": { "x": [], "y": [] },
                    "workerId": []
                },
                "weak": {
                    "start": {"x": [], "y": []},
                    "end": { "x": [], "y": [] },
                    "mean": { "x": [], "y": [] },
                    "workerId": []
                },
                "combined": {
                    "clusterIdx": [],
                    "start": {"x": [], "y": []},
                    "end": { "x": [], "y": [] },
                    "mean": { "x": [], "y": [] },
                    "workerId": [],
                    "method": []
                }
            }
            # Uncertain: ellipse result
            #path = os.getcwd() + "\\Admininterface\\Pre Rating\\{}\\".format(cur_job)
            #if not os.path.exists(path):
            #    os.makedirs(path)        
        
            if cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"] != []:
                fname = "{}_uncertain_cluster_ellipse.txt".format(cur_job)
                fname = path + fname
                with open(fname, "w") as f_uncertain:
                    f_uncertain.write("Detected uncertain cluster (ellipse criteria) to be checked by the admin\nclusterIdx | x | y | x2 | y2 | xMean | yMean | workerId")            
                    for count, clusterIdx in enumerate(cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["cluster_idc"]):
                        
                        acquiIdc = cars_in_shds[cur_job]["Input4Interfaces"]["ellResult"]["acqui_idc"][count]   # works for both methods (only acquis outside ellipse and whole cluster)

                        x = np.array(cars_in_shds[cur_job]["start"]["x"])[acquiIdc]
                        y = np.array(cars_in_shds[cur_job]["start"]["y"])[acquiIdc]
                        x2 = np.array(cars_in_shds[cur_job]["end"]["x"])[acquiIdc]
                        y2 = np.array(cars_in_shds[cur_job]["end"]["y"])[acquiIdc]
                        x_mean = cars_in_shds[cur_job]["kmeans"]["mean"]["x"][clusterIdx]
                        y_mean = cars_in_shds[cur_job]["kmeans"]["mean"]["y"][clusterIdx]
                        
                        # Plot and save "uncertain" clusters detected by clusterexamination (Ellipse ...)
                        fig42, ax = plt.subplots(1,1)
                        # Acquisitions
                        cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
                        
                        height, width = cur_img.shape    
                        extent = [-1, width-1, height-1, -1, ]
                        #extent = [-1, width-1, height-1, -1]  # Adjust coordinate origin (HTML Canvas = Matlab vs Python)
                        ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none', zorder=0) #, vmin=0, vmax=255)                    
                        
                        xmin = np.min([np.min(x), np.min(x2)])
                        xmax = np.max([np.max(x), np.max(x2)])
                        
                        ymin = np.min([np.min(y), np.min(y2)])
                        ymax = np.max([np.max(y), np.max(y2)])
                        
                        ax.plot([x, x2], [y, y2], color="red", linewidth= 2, zorder=1)
                        
                        ax.plot([],[], color="red", label=r'Erfassung')  # empty only legend label
                        
                        ax.axis([xmin-20,xmax+20,ymax+20,ymin-20])          #flipped    , origin upper left corner 
                        
                        title = "Unsicheres Cluster in Streifen {} (Clusteruntersuchung)\nAnzahl unsicherer Erfassungen = {}".format(cur_job+1, len(acquiIdc))
                        ax.set_title(title)
                        ax.set_xlabel("x [px]")
                        ax.set_ylabel("y [px]")
                        ax.legend(fancybox=True, shadow=True)
                        fig42.tight_layout()

                        fig42 = plt.gcf()
                        figManager = plt.get_current_fig_manager() # for fullscreen
                        figManager.resize(*figManager.window.maxsize())
                        
                        figpath = 'figures/{}/unsichere_Cluster/job_{}_uncertain_cluster_by_clusterexamination_idx_{}.png'.format(cur_job+1, cur_job+1, clusterIdx+1)
                        create_dir_if_not_exist("figures/{}/unsichere_Cluster/".format(cur_job+1))
                        
                        fig42.savefig(figpath, format="png", dpi=300, bbox_inches="tight")
                        plt.close("all")  

                        for cur_acqui, cur_acqui_idx in enumerate(acquiIdc):
                            # Find corresponding workerid
                            for job_idx, acqui_idc in enumerate(cars_in_shds[cur_job]["job_idc"]):
                                try:
                                    acqui_idc.index(cur_acqui_idx) # check if acqui idx is in job idc list                            
                                    workerId = cars_in_shds[cur_job]['workerId'][job_idx]
                                    break
                                except:
                                    continue
                                    #print("Skip to next worker, acqui idx not part of current workers job")
                            
                            f_uncertain.write("\n%s,%.2f %.2f %.2f %.2f,%.2f %.2f,%s" % (clusterIdx, x[cur_acqui], y[cur_acqui], x2[cur_acqui], y2[cur_acqui], x_mean, y_mean, workerId))
                    
                            dataToCheck[cur_job]["ell"]["start"]["x"].append(x[cur_acqui])
                            dataToCheck[cur_job]["ell"]["start"]["y"].append(y[cur_acqui])
                            dataToCheck[cur_job]["ell"]["end"]["x"].append(x2[cur_acqui])
                            dataToCheck[cur_job]["ell"]["end"]["y"].append(y2[cur_acqui])
                            dataToCheck[cur_job]["ell"]["mean"]["x"].append(x_mean)
                            dataToCheck[cur_job]["ell"]["mean"]["y"].append(y_mean)
                            dataToCheck[cur_job]["ell"]["workerId"].append(workerId)
    
                            dataToCheck[cur_job]["combined"]["method"].append("ellipse")
                            dataToCheck[cur_job]["combined"]["clusterIdx"].append(clusterIdx)
                            dataToCheck[cur_job]["combined"]["start"]["x"].append(x[cur_acqui])
                            dataToCheck[cur_job]["combined"]["start"]["y"].append(y[cur_acqui])
                            dataToCheck[cur_job]["combined"]["end"]["x"].append(x2[cur_acqui])
                            dataToCheck[cur_job]["combined"]["end"]["y"].append(y2[cur_acqui])
                            dataToCheck[cur_job]["combined"]["mean"]["x"].append(x_mean)
                            dataToCheck[cur_job]["combined"]["mean"]["y"].append(y_mean)
                            dataToCheck[cur_job]["combined"]["workerId"].append(workerId)
            
            # Uncertain: 2nd weak dbscan result
            if cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"] != []:
                fname = "{}_uncertain_cluster_db_weak.txt".format(cur_job)
                fname = path + fname
                with open(fname, "w") as f_uncertain:
                    f_uncertain.write("Detected uncertain cluster (2nd dbscan with weak parameters)to be checked by the admin\nclusterIdx | x | y | x2 | y2 | xMean | yMean | workerId")     
                    for clusterIdx, acquiIdc in enumerate(cars_in_shds[cur_job]["Input4Interfaces"]["weakResult"]["cluster_idc"]):
                        
                        fst_noise_mask = cars_in_shds[cur_job]["dbscan"]["noise_mask"]
                        #snd_noise_mask = cars_in_shds[cur_job]["dbscanWEAK"]["noise_mask"]
                        #acqui_idc = cars_in_shds[cur_job]["kmeans"]["cluster_idc"][clusterIdx]
                        
                        x = np.array(cars_in_shds[cur_job]["start"]["x"])[fst_noise_mask][acquiIdc]
                        y = np.array(cars_in_shds[cur_job]["start"]["y"])[fst_noise_mask][acquiIdc]
                        x2 = np.array(cars_in_shds[cur_job]["end"]["x"])[fst_noise_mask][acquiIdc]
                        y2 = np.array(cars_in_shds[cur_job]["end"]["y"])[fst_noise_mask][acquiIdc]
                        x_mean = cars_in_shds[cur_job]["dbscanWEAK"]["center"]["x"][clusterIdx]
                        y_mean = cars_in_shds[cur_job]["dbscanWEAK"]["center"]["y"][clusterIdx]    
                        
                        # Plot and save "uncertain" clusters detected by clusterexamination (Ellipse ...)
                        fig42, ax = plt.subplots(1,1)
                        
                        # Acquisitions
                        cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
                        
                        height, width = cur_img.shape    
                        extent = [-1, width-1, height-1, -1, ]
                        # extent = [-1, width-1, height-1, -1]  # Adjust coordinate origin (HTML Canvas = Matlab vs Python)
                        ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none', zorder=0) #, vmin=0, vmax=255)                    
                        
                        xmin = np.min([np.min(x), np.min(x2)])
                        xmax = np.max([np.max(x), np.max(x2)])
                        
                        ymin = np.min([np.min(y), np.min(y2)])
                        ymax = np.max([np.max(y), np.max(y2)])
                                                                
                        ax.plot([x, x2], [y, y2], color="red", linewidth= 2, zorder=1)
                        
                        ax.plot([],[], color="red", label=r'Erfassung')  # empty only legend label
                        
                        ax.axis([xmin-20,xmax+20,ymax+20,ymin-20])          #flipped    , origin upper left corner 
                        title = "Unsicheres Cluster in Streifen {} (Ausreißeruntersuchung)\nAnzahl unsicherer Erfassungen = {}".format(cur_job+1, len(acquiIdc))
                        ax.set_title(title)
                        ax.set_xlabel("x [px]")
                        ax.set_ylabel("y [px]")
                        ax.legend(fancybox=True, shadow=True)
                        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True)
                        fig42.tight_layout()
                        fig42 = plt.gcf()
                        # manager = plt.get_current_fig_manager()
                        # manager.resize(*manager.window.maxsize())
                        figManager = plt.get_current_fig_manager() # for fullscreen
                        figManager.resize(*figManager.window.maxsize())

                        figpath = 'figures/{}/unsichere_Cluster/job_{}_uncertain_cluster_by_outlierexamination_idx_{}.png'.format(cur_job+1, cur_job+1, clusterIdx+1)
                        create_dir_if_not_exist("figures/{}/unsichere_Cluster/".format(cur_job+1))
                        fig42.savefig(figpath, format="png", dpi=300, bbox_inches="tight")
                        plt.close("all")
                        
                        workerId = np.array(cars_in_shds[cur_job]["workerId_list"])[fst_noise_mask]         
                    
                        for cur_acqui, cur_acqui_idx in enumerate(acquiIdc):
                            f_uncertain.write("\n%s,%.2f %.2f %.2f %.2f,%.2f %.2f,%s" % (clusterIdx, x[cur_acqui], y[cur_acqui], x2[cur_acqui], y2[cur_acqui], x_mean, y_mean, workerId[cur_acqui_idx]))
                            dataToCheck[cur_job]["weak"]["start"]["x"].append(x[cur_acqui])
                            dataToCheck[cur_job]["weak"]["start"]["y"].append(y[cur_acqui])
                            dataToCheck[cur_job]["weak"]["end"]["x"].append(x2[cur_acqui])
                            dataToCheck[cur_job]["weak"]["end"]["y"].append(y2[cur_acqui])
                            dataToCheck[cur_job]["weak"]["mean"]["x"].append(x_mean)
                            dataToCheck[cur_job]["weak"]["mean"]["y"].append(y_mean)
                            dataToCheck[cur_job]["weak"]["workerId"].append(workerId[cur_acqui_idx])
                            
                            dataToCheck[cur_job]["combined"]["method"].append("db_weak")
                            dataToCheck[cur_job]["combined"]["clusterIdx"].append(clusterIdx)
                            dataToCheck[cur_job]["combined"]["start"]["x"].append(x[cur_acqui])
                            dataToCheck[cur_job]["combined"]["start"]["y"].append(y[cur_acqui])
                            dataToCheck[cur_job]["combined"]["end"]["x"].append(x2[cur_acqui])
                            dataToCheck[cur_job]["combined"]["end"]["y"].append(y2[cur_acqui])
                            dataToCheck[cur_job]["combined"]["mean"]["x"].append(x_mean)
                            dataToCheck[cur_job]["combined"]["mean"]["y"].append(y_mean)
                            dataToCheck[cur_job]["combined"]["workerId"].append(workerId[cur_acqui_idx])
        
        print("\n---------------------------------------")
        print("Ergebnis der Integration vor der Überprüfung (Unsichere Cluster):") 
        sum_cluster_ell = []; sum_cluster_db = []; sum_acqui_ell = []; sum_acqui_db = []
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            uncertain_count_ell = []; uncertain_count_db = []
            for idx, cluster in enumerate(dataToCheck[cur_job]["combined"]["method"]):
                if cluster == "ellipse":
                    uncertain_count_ell.append(dataToCheck[cur_job]["combined"]["clusterIdx"][idx])
                if cluster == "db_weak":
                    uncertain_count_db.append(dataToCheck[cur_job]["combined"]["clusterIdx"][idx])
            
            clusterCount_ell = len(set(uncertain_count_ell))
            clusterCount_db = len(set(uncertain_count_db))
            
            sum_cluster_ell.append(clusterCount_ell)
            sum_cluster_db.append(clusterCount_db)
            sum_acqui_ell.append(len(uncertain_count_ell))
            sum_acqui_db.append(len(uncertain_count_db))
            # np.sum(uncertain_count_ell), np.sum(uncertain_count_db)
            print("Streifen {}: Clusteruntersuchung: {} Cluster = {} Erfassungen, Ausreißeruntersuchung: {} Cluster = {} Erfassungen".format(
                cur_job+1, clusterCount_ell, len(uncertain_count_ell), clusterCount_db, len(uncertain_count_db)))
        print("Gesamt: Clusteruntersuchung = {} Cluster = {} Erfassungen, Ausreißeruntersuchung: {} Cluster = {} Erfassungen".format(
                                        np.sum(sum_cluster_ell), np.sum(sum_acqui_ell), np.sum(sum_cluster_db), np.sum(sum_acqui_db)))
        print("Unsichere Cluster: {} = {} Erfassungen\n".format(np.sum(sum_cluster_ell) + np.sum(sum_cluster_db), np.sum(sum_acqui_ell) + np.sum(sum_acqui_db)))
        
        # .txt for the Crowdinterface
        number_needed_crowdworker = []
        number_file_iteration = 5   # Number of crowdworkers rating 1 file consisting of up to 10 acquisitions
        path = "Crowdinterface/Pre Rating/"
        create_dir_if_not_exist(path)
        
        # overwrite_crowd_input = False   # !! Important, because crowd input is created using randomizer, unique id shouldve been added
        
        overwrite_crowd_input = config["integration"]["overwrite_crowd_input"]
        
        if overwrite_crowd_input:
            for cur_job in range(config["jobs"]["number_of_jobs"]):
                # path = os.getcwd() + "\\Crowdinterface\\Pre Rating\\"

                ## Calculate amount of files needed by job        
                total_uncertain = len(dataToCheck[cur_job]["combined"]["start"]["x"])
                
                if not total_uncertain: # If no uncertain acquisitions for current batch go to next
                    print("No uncertain acquisition for batch {}".format(cur_job + 1))
                    continue
                
                max_acqui = 10  # 1min Ref Guide, 20s each Acqui -> 1min + 10*20s = 4min, 20s total
                total_ref = 1   # 1 Reference Acqui to simplify rating of verification acquisitions (Correct Answers are known) -> Autorate NOK if worker fails ref question
                free_acqui = max_acqui - total_ref
                
                number_file = math.ceil(total_uncertain / free_acqui)   # Needed file count, math.ceil -> Round up to next integer eg. 3.2 -> 4, 3.0 -> 3, 0.5 -> 1
                
                number_needed_crowdworker.append(number_file * number_file_iteration)
                
                # Gather data and shuffle (So worker will have to rate acquis from different clusters and not only the same)     
                random_list = [*range(0, total_uncertain)]  # Idx list
                random.shuffle(random_list)   # Random ordered
                
                sub_list = np.array_split(np.array(random_list), number_file)  # Spread idc to files
                
                # Load reference data
                #path_ref = os.getcwd() + "\\Crowdinterface\\Reference\\"
                path_ref = "Crowdinterface/Reference/"
                
                fname = "{}.txt".format(cur_job+1)
                fname = path_ref + fname
                print("fname:", fname)
                ref_startX = []
                ref_startY = []
                ref_endX = []
                ref_endY = []
                with open(fname) as f_ref:
                    for line in f_ref:
                        (ref_startX, ref_startY, ref_endX, ref_endY) = [float(x) for x in line.split()]     
                        ref_meanX = ((ref_startX + ref_endX) / 2)
                        ref_meanY = ((ref_startY + ref_endY) / 2)
                
                # Write to .txt
                for fnumber in range(number_file):            
                    # Combine reference and uncertain data and write to file
                    fname = "{}_{}_uncertain_crowd.txt".format(cur_job+1, fnumber)
                    fname = path + fname
                    with open(fname, "w") as f_uncertain:
                        f_uncertain.write("Detected uncertain cluster (combination of ellipse and 2nd dbscan) to be checked by the admin\nclusterIdx | x | y | x2 | y2 | xMean | yMean | workerId")
                        f_uncertain.write("\n%s,%.2f %.2f %.2f %.2f,%.2f %.2f,%s" % (400, ref_startX, ref_startY, ref_endX, ref_endY, ref_meanX, ref_meanY, "Admin"))    
                        for idx in sub_list[fnumber]:                    
                            x = dataToCheck[cur_job]["combined"]["start"]["x"][idx]
                            y = dataToCheck[cur_job]["combined"]["start"]["y"][idx]
                            x2 = dataToCheck[cur_job]["combined"]["end"]["x"][idx]
                            y2 = dataToCheck[cur_job]["combined"]["end"]["y"][idx]
                            x_mean = dataToCheck[cur_job]["combined"]["mean"]["x"][idx]
                            y_mean = dataToCheck[cur_job]["combined"]["mean"]["y"][idx]
                            workerId = dataToCheck[cur_job]["combined"]["workerId"][idx]
                            clusterIdx = dataToCheck[cur_job]["combined"]["clusterIdx"][idx]
                            method = dataToCheck[cur_job]["combined"]["method"][idx]
                        
                            f_uncertain.write("\n%s,%.2f %.2f %.2f %.2f,%.2f %.2f,%s,%s" % (clusterIdx, x, y, x2, y2, x_mean, y_mean, workerId, method))    
    except:
        print("Error: Organizing results.")
    total_needed_crowdworker = sum(number_needed_crowdworker)


def generate_links(config):
    """ 
    Generate links (query url) for uncertain clusters where admin needs to place reference. Links pointing to the admininterface on the bplaced server
    
    Parameters:
    ----------
        config: dict
    
    Results:
    ----------  
        links: list
    
    """
    
    path = "Admininterface/Pre Rating"
    pathend = "/*.txt"
    pathend = path + pathend
    #baseURL = "https://geoinf-rs.bplaced.net/Admininterface"
    baseURL = config["jobs"]["url_admin"]
    
    links = []
    for f in glob.glob(pathend):
        f = re.sub(path + "\\\\", "", f)
        f = re.sub(".txt", "", f)
        
        f = f.split("_")
        # umständlich
        batchIdx = str(int(f[0]) + 1)
        
        if f[-1] == "weak":  
            method = "db"
        else:
            method = "ell"
        
        links.append(baseURL + "?batchIdx=" + batchIdx + "&method=" + method)
    
    return links


def open_browser(links):
    """ 
    Open links as tabs in new browser instance (chrome)
    
    Parameters:
    ----------
    
        links: list
            urls pointing to admininterface
    
    """
    
    chromePath_NW = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s --new-window"
    chromePath = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"    
    
    chrome_NW = webbrowser.get(chromePath_NW)
    chrome = webbrowser.get(chromePath)
    
    first = True
    new = 0
    for link in links: 
        if first:
            chrome_NW.open(link, new=0)
            first = False
        else:
            new += 1
            chrome.open(link, new=new)


def plot_performance(method, quality_params1, quality_params2, config):
    ## Plot Precision, Recall, F1-Score
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py 2.2.2021
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    labels = [ str(ele) for ele in [*range(1,config["jobs"]["number_of_jobs"]+1)] ]

    x = np.arange(len(labels)) # label location
    barWidth = 0.25 # bar width

    # Set position of bar on x axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth + 0.01 for x in r1]
    r3 = [x + barWidth + 0.01 for x in r2]

    fig, ax = plt.subplots(figsize=(14,7))
    
    prec_no_diff = []; rec_no_diff = []; f1_no_diff = []
    precision = []; recall = []; f1_score = []; 
    performance = {"precision": {"lower": [], "diff": [], "increase": []}, "recall": {"lower": [], "diff": [], "increase": []}, "f1-score": {"lower": [], "diff": [], "increase": []}}
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        
        diff_prec = quality_params1[cur_job]["precision"] - quality_params2[cur_job]["precision"]        
        if diff_prec > 0:
            performance["precision"]["lower"].append(round(quality_params2[cur_job]["precision"] * 100, 1))
            performance["precision"]["diff"].append(round(abs(diff_prec) * 100, 1))
            performance["precision"]["increase"].append(False)
            
        if diff_prec <= 0:
            performance["precision"]["lower"].append(round(quality_params1[cur_job]["precision"] * 100, 1))
            performance["precision"]["diff"].append(round(abs(diff_prec) * 100, 1))
            performance["precision"]["increase"].append(True)
        
        if diff_prec == 0:
            prec_no_diff.append(True)
        else:
            prec_no_diff.append(False)
            
        diff_rec = quality_params1[cur_job]["recall"] - quality_params2[cur_job]["recall"]     
        if diff_rec > 0:
            performance["recall"]["lower"].append(round(quality_params2[cur_job]["recall"] * 100, 1))
            performance["recall"]["diff"].append(round(abs(diff_rec) * 100, 1))
            performance["recall"]["increase"].append(False)
            
        if diff_rec <= 0:
            performance["recall"]["lower"].append(round(quality_params1[cur_job]["recall"] * 100, 1))
            performance["recall"]["diff"].append(round(abs(diff_rec * 100), 1))
            performance["recall"]["increase"].append(True)
        
        if diff_rec == 0:
            rec_no_diff.append(True)
        else:
            rec_no_diff.append(False)
        
        diff_f1 = quality_params1[cur_job]["f1-score"] - quality_params2[cur_job]["f1-score"]     
        if diff_f1 > 0:
            performance["f1-score"]["lower"].append(round(quality_params2[cur_job]["f1-score"] * 100, 1))
            performance["f1-score"]["diff"].append(round(abs(diff_f1) * 100, 1))
            performance["f1-score"]["increase"].append(False)
            
        if diff_f1 <= 0:
            performance["f1-score"]["lower"].append(round(quality_params1[cur_job]["f1-score"] * 100, 1))
            performance["f1-score"]["diff"].append(round(abs(diff_f1) * 100, 1))
            performance["f1-score"]["increase"].append(True)

        if diff_f1 == 0:
            f1_no_diff.append(True)
        else:
            f1_no_diff.append(False)
    # colors = iter([plt.cm.Pastel1(i) for i in range(9)])
    # next(colors)
    
    color_prec = (0.7019607843137254, 0.803921568627451, 0.8901960784313725, 1.0)
    color_rec = (0.8705882352941177, 0.796078431372549, 0.8941176470588236, 1.0)
    color_f1 = (0.996078431372549, 0.8509803921568627, 0.6509803921568628, 1.0)
    color_inc = (0.8, 0.9215686274509803, 0.7725490196078432, 1.0) #"tab:green"
    color_dec = (0.984313725490196, 0.7058823529411765, 0.6823529411764706, 1.0) #"tab:red"
    
    def autolabel(rect_lower, rect_upper, increase):
        """ 
        Attach a text label above each bar in *rects*, displaying its height.
        """
        
        height_lower = rect_lower.get_height()
        height_upper = rect_upper.get_height()
        
        # Lower bar stack
        #if height_upper == 0:
        #    new_height = height_lower
        #else:
        #    if increase:
        #        new_height = round(height_upper + height_lower, 1)
        #    else:
        #        new_height = round(height_upper - height_lower, 1)
        #ax.annotate("{}".format(new_height),
        ax.annotate("{}".format(height_lower),
            xy=(rect_lower.get_x() + rect_lower.get_width() / 2, height_lower /2), #- 5),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')
            
        # Upper bar stack       
        if height_upper == 0:
            return
            #text = "$\pm$ 0"
        else:
            if increase:
                text = "+{}".format(round(height_upper, 1))
            else:
                text = "-{}".format(round(height_upper, 1))
        
        ax.annotate(text,
            xy=(rect_upper.get_x() + rect_upper.get_width() / 2, height_lower + height_upper - 0.5),  # - 5),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')
    
    for i, r in enumerate(r1):
        if performance["precision"]["increase"][i] == True:
            color_diff = color_inc
            label_diff = "increase"
        else:
            color_diff = color_dec
            label_diff = "decrease"

        rect1_lower = ax.bar(r, height=performance["precision"]["lower"][i], bottom=0, width=barWidth, label = "Precision", color=color_prec)
        rect1_upper = ax.bar(r, height=performance["precision"]["diff"][i], bottom=performance["precision"]["lower"][i], width=barWidth, color=color_diff, label=label_diff) #hatch='xxx', edgecolor="white")
        autolabel(rect1_lower[0], rect1_upper[0], performance["precision"]["increase"][i])
    
    for i, r in enumerate(r2):
        if performance["recall"]["increase"][i] == True:
            color_diff = color_inc
            label_diff = "increase"
        else:
            color_diff = color_dec
            label_diff = "decrease"
        
        rect2_lower = ax.bar(r, height=performance["recall"]["lower"][i], bottom=0, width=barWidth, label = "Recall", color=color_rec)
        rect2_upper = ax.bar(r, height=performance["recall"]["diff"][i], bottom=performance["recall"]["lower"][i], width=barWidth, color=color_diff, label=label_diff) #, hatch='xxx', edgecolor="white")
        autolabel(rect2_lower[0], rect2_upper[0], performance["recall"]["increase"][i])
        
    for i, r in enumerate(r3):
        if performance["f1-score"]["increase"][i] == True:
            color_diff = color_inc
            label_diff = "increase" 
        else:
            color_diff = color_dec
            label_diff = "decrease" 
            
        rect3_lower = ax.bar(r, height=performance["f1-score"]["lower"][i], bottom=0, width=barWidth, label = "F1-Score", color=color_f1)
        rect3_upper = ax.bar(r, height=performance["f1-score"]["diff"][i], bottom=performance["f1-score"]["lower"][i], width=barWidth, color=color_diff, label=label_diff) #, hatch='xxx', edgecolor="white")
        autolabel(rect3_lower[0], rect3_upper[0], performance["f1-score"]["increase"][i])
    
    ax.set_xticks([r + barWidth for r in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Streifen")
    ax.set_ylabel("%")
    if method == "pre<->post-admin":
        ax.set_title("Entwicklung der Genauigkeitsmaße:\nVor- vs nach der Überprüfung mittels Admin")
    if method == "pre<->post-crowd":
        ax.set_title("Entwicklung der Genauigkeitsmaße:\nVor- vs nach der Überprüfung mittels Crowd")
    if method == "post-admin<->post-crowd":
        ax.set_title("Unterschied der Genauigkeitsmaße\nNach der Überprüfung mittels Admin vs Crowd")
    
    # Put a legend to the right of the current axis    
    additional_patches = []
    # Increase 
    try:
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            for idx, bool_ in enumerate(performance["precision"]["increase"]):
                if bool_ and not prec_no_diff[idx]:
                    additional_patches.append(Patch(color=color_inc, label="Steigerung"))
                    raise StopIteration
            for idx, bool_ in enumerate(performance["recall"]["increase"]):
                if bool_ and not rec_no_diff[idx]:
                    additional_patches.append(Patch(color=color_inc, label="Steigerung"))
                    raise StopIteration
            for idx, bool_ in enumerate(performance["f1-score"]["increase"]):
                if bool_ and not f1_no_diff[idx]:
                    additional_patches.append(Patch(color=color_inc, label="Steigerung"))
                    raise StopIteration
    except StopIteration:
        pass
    # Decrease
    try:
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            for idx, bool_ in enumerate(performance["precision"]["increase"]):
                if not bool_ and not prec_no_diff[idx]:
                    additional_patches.append(Patch(color=color_dec, label="Verringerung"))
                    raise StopIteration
            for idx, bool_ in enumerate(performance["recall"]["increase"]):
                if not bool_ and not rec_no_diff[idx]:
                    additional_patches.append(Patch(color=color_dec, label="Verringerung"))
                    raise StopIteration
            for idx, bool_ in enumerate(performance["f1-score"]["increase"]):
                if not bool_ and not f1_no_diff[idx]:
                    additional_patches.append(Patch(color=color_dec, label="Verringerung"))
                    raise StopIteration
    except StopIteration:
        pass
            
    handles = [Patch(color=color_prec, label="Precision"),
               Patch(color=color_rec, label="Recall"),
               Patch(color=color_f1, label="F1-Score")]
    handles.extend(additional_patches)
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    fig.tight_layout()

    fig1 = plt.gcf()
    figManager = plt.get_current_fig_manager()  # for fullscreen
    # figManager.full_screen_toggle()
    # figManager.window.state("zoomed")
    # plt.show()
    savePlot = True
    if savePlot:
        path = "figures/overall_quality_parameters/"
        create_dir_if_not_exist(path) 

        if method == "pre<->post-admin":
            fname = "quality_developement_pre_post-admin.png"
        if method == "pre<->post-crowd":
            fname = "quality_developement_pre_post-crowd.png"
        if method == "post-admin<->post-crowd":
            fname = "quality_diff_post-admin_post-crowd.png"

        path += fname

        fig1.savefig(path, format="png", dpi=300)
        plt.close("all")  


def plot_overlap(pre, post_admin, post_crowd, config):
    
    ## Plot Combined Error Distributions    
    pos_dst_PRE =[]; len_dst_PRE = []; ori_dst_PRE = []; hausdorff_dst_PRE = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):      
        pos_dst_PRE.extend(pre[cur_job]["err"]["pos"])
        len_dst_PRE.extend(pre[cur_job]["err"]["len"])
        ori_dst_PRE.extend(pre[cur_job]["err"]["ori"])  # * 180 / math.pi
        hausdorff_dst_PRE.extend(pre[cur_job]["err"]["hausdorff"])
    
    pos_dst_PRE = np.array(pos_dst_PRE) * config["integration"]["cellSize"]
    len_dst_PRE = np.array(len_dst_PRE) * config["integration"]["cellSize"]
    hausdorff_dst_PRE = np.array(hausdorff_dst_PRE) * config["integration"]["cellSize"]
    
    pos_dst_ADMIN =[]; len_dst_ADMIN = []; ori_dst_ADMIN = []; hausdorff_dst_ADMIN = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):      
        pos_dst_ADMIN.extend(post_admin[cur_job]["err"]["pos"])
        len_dst_ADMIN.extend(post_admin[cur_job]["err"]["len"])
        ori_dst_ADMIN.extend(post_admin[cur_job]["err"]["ori"])  # * 180 / math.pi
        hausdorff_dst_ADMIN.extend(post_admin[cur_job]["err"]["hausdorff"])
    
    pos_dst_ADMIN = np.array(pos_dst_ADMIN) * config["integration"]["cellSize"]
    len_dst_ADMIN = np.array(len_dst_ADMIN) * config["integration"]["cellSize"]
    hausdorff_dst_ADMIN = np.array(hausdorff_dst_ADMIN) * config["integration"]["cellSize"]
    
    pos_dst_CROWD =[]; len_dst_CROWD = []; ori_dst_CROWD = []; hausdorff_dst_CROWD = []
    for cur_job in range(config["jobs"]["number_of_jobs"]):      
        pos_dst_CROWD.extend(post_crowd[cur_job]["err"]["pos"])
        len_dst_CROWD.extend(post_crowd[cur_job]["err"]["len"])
        ori_dst_CROWD.extend(post_crowd[cur_job]["err"]["ori"])  # * 180 / math.pi
        hausdorff_dst_CROWD.extend(post_crowd[cur_job]["err"]["hausdorff"])
    
    pos_dst_CROWD = np.array(pos_dst_CROWD) * config["integration"]["cellSize"]
    len_dst_CROWD = np.array(len_dst_CROWD) * config["integration"]["cellSize"]
    hausdorff_dst_CROWD = np.array(hausdorff_dst_CROWD) * config["integration"]["cellSize"]
    
    # Plot overlapping histograms 
    # Position error, Length error, Orientation error, Hausdorff error
    plot = True
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(10,14))
        
        plt.suptitle('Überlagerte Histogramme der Fehler: Vor und nach der Überprüfung mittels Admin & Crowd')
        
        binCount = 20
        ################### POS
        min_ = 0  # np.min([np.min(pos_dst_PRE), np.min(pos_dst_ADMIN), np.min(pos_dst_CROWD)])
        max_ = np.max([np.max(pos_dst_PRE), np.max(pos_dst_ADMIN), np.max(pos_dst_CROWD)])
        weights1 = np.ones_like(pos_dst_PRE)/len(pos_dst_PRE)
        weights2 = np.ones_like(pos_dst_ADMIN)/len(pos_dst_ADMIN)
        weights3 = np.ones_like(pos_dst_CROWD)/len(pos_dst_CROWD)
        x1, bins1, p1 = ax1.hist(pos_dst_PRE, binCount, alpha=0.5, label="Vor der Überprüfung", range=[ min_, max_], weights=weights1, histtype='step', stacked=True, fill=False)
        x2, bins2, p2 = ax1.hist(pos_dst_ADMIN, binCount, alpha=0.5, label="Nach der Überprüfung mittels Admin", range=[ min_, max_], weights=weights2, histtype='step', stacked=True, fill=False)
        x3, bins3, p3 = ax1.hist(pos_dst_CROWD, binCount, alpha=0.5, label="Nach der Überprüfung mittels Crowd", range=[ min_, max_], weights=weights3, histtype='step', stacked=True, fill=False)
        ax1.set_xlabel("Positionsfehler [m]")
        ax1.set_ylabel("Relative Häufigkeit")
        ax1.legend(shadow=True, loc="upper right", handlelength=1.5)
        ################### LEN
        min_ = 0 #np.min([np.min(len_dst_PRE), np.min(len_dst_ADMIN), np.min(len_dst_CROWD)])
        max_ = np.max([np.max(len_dst_PRE), np.max(len_dst_ADMIN), np.max(len_dst_CROWD)])
        weights1 = np.ones_like(len_dst_PRE)/len(len_dst_PRE)
        weights2 = np.ones_like(len_dst_ADMIN)/len(len_dst_ADMIN)
        weights3 = np.ones_like(len_dst_CROWD)/len(len_dst_CROWD)
        x1, bins1, p1 = ax2.hist(len_dst_PRE, binCount, alpha=0.5, label="Vor der Überprüfung", range=[ min_, max_], weights=weights1, histtype='step', stacked=True, fill=False)
        x2, bins2, p2 = ax2.hist(len_dst_ADMIN, binCount, alpha=0.5, label="Nach der Überprüfung mittels Admin", range=[ min_, max_], weights=weights2, histtype='step', stacked=True, fill=False)
        x3, bins3, p3 = ax2.hist(len_dst_CROWD, binCount, alpha=0.5, label="Nach der Überprüfung mittels Crowd", range=[ min_, max_], weights=weights3, histtype='step', stacked=True, fill=False)
        ax2.set_xlabel("Längenfehler [m]")
        ax2.set_ylabel("Relative Häufigkeit")
        ax2.legend(shadow=True, loc="upper right", handlelength=1.5)
        ################### ORI
        min_ = 0 #np.min([np.min(ori_dst_PRE), np.min(ori_dst_ADMIN), np.min(ori_dst_CROWD)])
        max_ = np.max([np.max(ori_dst_PRE), np.max(ori_dst_ADMIN), np.max(ori_dst_CROWD)])
        weights1 = np.ones_like(ori_dst_PRE)/len(ori_dst_PRE)
        weights2 = np.ones_like(ori_dst_ADMIN)/len(ori_dst_ADMIN)
        weights3 = np.ones_like(ori_dst_CROWD)/len(ori_dst_CROWD)
        x1, bins1, p1 = ax3.hist(ori_dst_PRE, binCount, alpha=0.5, label="Vor der Überprüfung", range=[ min_, max_], weights=weights1, histtype='step', stacked=True, fill=False)
        x2, bins2, p2 = ax3.hist(ori_dst_ADMIN, binCount, alpha=0.5, label="Nach der Überprüfung mittels Admin", range=[ min_, max_], weights=weights2, histtype='step', stacked=True, fill=False)
        x3, bins3, p3 = ax3.hist(ori_dst_CROWD, binCount, alpha=0.5, label="Nach der Überprüfung mittels Crowd", range=[ min_, max_], weights=weights3, histtype='step', stacked=True, fill=False)
        ax3.set_xlabel("Orientierungsfehler [°]")
        ax3.set_ylabel("Relative Häufigkeit")
        ax3.legend(shadow=True, loc="upper right", handlelength=1.5)
        ################### Hausdorff
        min_ = 0 #np.min([np.min(hausdorff_dst_PRE), np.min(hausdorff_dst_ADMIN), np.min(hausdorff_dst_CROWD)])
        max_ = np.max([np.max(hausdorff_dst_PRE), np.max(hausdorff_dst_ADMIN), np.max(hausdorff_dst_CROWD)])
        weights1 = np.ones_like(hausdorff_dst_PRE)/len(hausdorff_dst_PRE)
        weights2 = np.ones_like(hausdorff_dst_ADMIN)/len(hausdorff_dst_ADMIN)
        weights3 = np.ones_like(hausdorff_dst_CROWD)/len(hausdorff_dst_CROWD)
        x1, bins1, p1 = ax4.hist(hausdorff_dst_PRE, binCount, alpha=0.5, label="Vor der Überprüfung", range=[ min_, max_], weights=weights1, histtype='step', stacked=True, fill=False)
        x2, bins2, p2 = ax4.hist(hausdorff_dst_ADMIN, binCount, alpha=0.5, label="Nach der Überprüfung mittels Admin", range=[ min_, max_], weights=weights2, histtype='step', stacked=True, fill=False)
        x3, bins3, p3 = ax4.hist(hausdorff_dst_CROWD, binCount, alpha=0.5, label="Nach der Überprüfung mittels Crowd", range=[ min_, max_], weights=weights3, histtype='step', stacked=True, fill=False)
        ax4.set_xlabel("Hausdorff-Metrik [m]")
        ax4.set_ylabel("Relative Häufigkeit")
        ax4.legend(shadow=True, loc="upper right", handlelength=1.5)
        
        fig.tight_layout()
        
        plt.subplots_adjust(bottom=0.045, hspace=0.27, top=0.95, right=0.975, wspace=0.202, left=0.079)
        
        # plt.show()
        
        fig1 = plt.gcf()
        # manager = plt.get_current_fig_manager()
        # manager.resize(*manager.window.maxsize())
        
        savePlot = True
        if savePlot:
            path = 'figures/overall_quality_parameters/'.format(cur_job+1)
            create_dir_if_not_exist(path) 
            
            fname = 'combined_jobs_error_distribution.png'.format()
            path += fname            
            fig1.savefig(path, format='png', dpi=300)
            plt.close("all") 


def calculate_ratings(config):
    """
    Calculate rating for each acquisition based on admin- & crowdinterface
    Combine ratings
    Check if all worker acquisitions receive rating
    Plot every acquisition and evaluated rating
    Compute final rating for crowdworker
    """
    
    # Load varibles from integration pre verification
    try:
        with open(config["backup"]["cars_in_shds_pre_verification"], "rb") as file:
            cars_in_shds = pickle.load(file) 
    except:
        raise Exception ("Error trying to load variables from backupfile")

    ## Admininterface: Calculate difference between placed reference and given acquisitions
    print("\n --------------------------------------------------")
    print("Calculating difference between newly set reference and acquisitions of corresponding cluster")
    print("--------------------------------------------------")
    ellRatingResult = dict()
    dbRatingResult = dict()
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        # Ellipse input
        ratingPath = "Admininterface/Post Rating/{}_ell.txt".format(cur_job+1)
        inputPath = "Admininterface/Pre Rating/{}_uncertain_cluster_ellipse.txt".format(cur_job)
        
        try:
            ellRatingResult[cur_job] = diff_reference_acqui(cur_job, ratingPath, inputPath, config)
        except:
            print("ERROR: calc difference ref-acqui (ellipse)")
            raise Exception("Error calc difference ref-acqui (ellipse)")
        # 2nd dbscan input
        ratingPath = "Admininterface/Post Rating/{}_db.txt".format(cur_job+1)
        inputPath = "Admininterface/Pre Rating/{}_uncertain_cluster_db_weak.txt".format(cur_job)
        
        try:
            dbRatingResult[cur_job] = diff_reference_acqui(cur_job, ratingPath, inputPath, config)
        except:
            print("ERROR: calc difference ref-acqui (2nd db)")
            raise Exception("Error calc difference ref-acqui (2nd db)")
        
    ## Crowdinterface
    skipCrowd = True  # True: -> calc admin & crowd rating, False: -> calc only admin
    if skipCrowd:
        directory = set_dir(method ="questions")
        [returnDict_crowd, dbRatingResult_crowd, ellRatingResult_crowd, count_clear_majority, id_no_clear_majority] = rate_questions(directory, config)

        print("\nStep 3: Majority vote not clear in {} / {} = {:.1f}%".format(len(count_clear_majority[3]) - sum(count_clear_majority[3]) , len(count_clear_majority[3]), (len(count_clear_majority[3]) - sum(count_clear_majority[3])) / len(count_clear_majority[3]) * 100))
        print("Step 5: Majority vote not clear in {} / {} = {:.1f}%".format(len(count_clear_majority[5]) - sum(count_clear_majority[5]) , len(count_clear_majority[5]), (len(count_clear_majority[5]) - sum(count_clear_majority[5])) / len(count_clear_majority[5]) * 100))

    ## Print rating results to console
    print("\n-------------------------------")
    print("Adminrating result:")
    ## Ell rating result
    def diffAdminCrowdRating(ratingType, adminRating, adminCoord, adminWorkerId):
        # Compare to rating from crowd
        noDiff = {}
        for item in ratingType:
            for acqui in ratingType[item]:
                if acqui["workerId"] == adminWorkerId[0] and acqui["acquiCoord"]["x"] == adminCoord[0] and acqui["acquiCoord"]["y"] == adminCoord[1]:
                    for step in [1,3,5]:
                        noDiff[step] = []
                        if acqui["finalRating"]["steps"][step] != adminRating:
                            noDiff[step] = False
                        else:
                            noDiff[step] = True

                    return noDiff

        if diff != True or diff != False:
            print("ERROR: Not matching (admin-crowd ratings)")

    admin_korr_ell = 0; admin_inkorr_ell = 0; ratingNoDiff = []; ratingNoDiff_1 = []; ratingNoDiff_3 = []; ratingNoDiff_5=[]
    for f in ellRatingResult:
        for item in ellRatingResult[f]:
            for acqui in ellRatingResult[f][item]:
                if acqui["finalRating"] == "OK":
                    admin_korr_ell+=1
                if acqui["finalRating"] == "NOK":
                    admin_inkorr_ell +=1
                ratingNoDiff = diffAdminCrowdRating(ellRatingResult_crowd[f], acqui["finalRating"], acqui["acquiCoord"], acqui["workerId"])
                ratingNoDiff_1.append(ratingNoDiff[1]); ratingNoDiff_3.append(ratingNoDiff[3]); ratingNoDiff_5.append(ratingNoDiff[5])

    print("Admin ell rating --> korrekt={}, inkorrekt={}".format(admin_korr_ell, admin_inkorr_ell))
    
    totalc = admin_korr_ell + admin_inkorr_ell
    print("Diff between ELLIPSE rating of Admin vs Crowd: Step 1:{}/{}={:.2f}% identical,         Step3:{}/{}={:.2f}% identical,        Step5:{}/{}={:.2f}% identical".format(
        np.sum(ratingNoDiff_1), totalc,  np.sum(ratingNoDiff_1)/totalc * 100,
        np.sum(ratingNoDiff_3), totalc,  np.sum(ratingNoDiff_3)/totalc * 100,
        np.sum(ratingNoDiff_5), totalc,  np.sum(ratingNoDiff_5)/totalc * 100))

    ## DB rating result
    admin_korr_db = 0; admin_inkorr_db = 0; ratingNoDiff_db = []; ratingNoDiff_1_db = []; ratingNoDiff_3_db = []; ratingNoDiff_5_db=[]
    for f in dbRatingResult:
        for item in dbRatingResult[f]:
            for acqui in dbRatingResult[f][item]:
                if acqui["finalRating"] == "OK":
                    admin_korr_db+= 1
                if acqui["finalRating"] == "NOK":
                    admin_inkorr_db += 1
                ratingNoDiff_db = diffAdminCrowdRating(dbRatingResult_crowd[f], acqui["finalRating"], acqui["acquiCoord"], acqui["workerId"])
                ratingNoDiff_1_db.append(ratingNoDiff_db[1]); ratingNoDiff_3_db.append(ratingNoDiff_db[3]); ratingNoDiff_5_db.append(ratingNoDiff_db[5])
                
    print("Admin db rating --> korrekt={}, inkorrekt={}".format(admin_korr_db, admin_inkorr_db))
    
    totalc_db = admin_korr_db + admin_inkorr_db
    print("Diff between ELLIPSE rating of Admin vs Crowd: Step 1:{}/{}={:.2f}% identical,         Step3:{}/{}={:.2f}% identical,        Step5:{}/{}={:.2f}% identical".format(
        np.sum(ratingNoDiff_1_db), totalc_db,  np.sum(ratingNoDiff_1_db)/totalc_db * 100,
        np.sum(ratingNoDiff_3_db), totalc_db,  np.sum(ratingNoDiff_3_db)/totalc_db * 100,
        np.sum(ratingNoDiff_5_db), totalc_db,  np.sum(ratingNoDiff_5_db)/totalc_db * 100))
    
    print("Gesamt: Adminrating: korrekt = {}, inkorrekt = {} ---> korrekt = {:.2f}%".format(admin_korr_db + admin_korr_ell, admin_inkorr_db + admin_inkorr_ell,  100 * (admin_korr_db + admin_korr_ell) / (admin_korr_db + admin_korr_ell + admin_inkorr_db + admin_inkorr_ell)))
    
    print("Gesamt: Übereinstimmung crowd-admin: step1: {}/{}={:.2f}%,      step3: {}/{}={:.2f}%,     step5: {}/{}={:.2f}%".format(
        np.sum(ratingNoDiff_1) + np.sum(ratingNoDiff_1_db), totalc + totalc_db, (np.sum(ratingNoDiff_1) + np.sum(ratingNoDiff_1_db))/(totalc + totalc_db) * 100,
        np.sum(ratingNoDiff_3) + np.sum(ratingNoDiff_3_db), totalc + totalc_db, (np.sum(ratingNoDiff_3) + np.sum(ratingNoDiff_3_db))/(totalc + totalc_db) * 100,
        np.sum(ratingNoDiff_5) + np.sum(ratingNoDiff_5_db), totalc + totalc_db, (np.sum(ratingNoDiff_5) + np.sum(ratingNoDiff_5_db))/(totalc + totalc_db) * 100,
    ))
    print("----------------------")
    if skipCrowd:
        for step in [1, 3, 5]:
            print("\nCrowdrating result: STEP = {}".format(step))
            crowd_korr_ell = 0; crowd_inkorr_ell = 0
            for f in ellRatingResult_crowd:
                for item in ellRatingResult_crowd[f]:
                    for acqui in ellRatingResult_crowd[f][item]:
                        if acqui["finalRating"]["steps"][step] == "OK":
                            crowd_korr_ell += 1
                        if acqui["finalRating"]["steps"][step] == "NOK":
                            crowd_inkorr_ell += 1
            print("Crowd ell rating --> korrekt={}, inkorrekt={}".format(crowd_korr_ell, crowd_inkorr_ell))

            crowd_korr_db = 0; crowd_inkorr_db = 0
            for f in dbRatingResult_crowd:
                for item in dbRatingResult_crowd[f]:
                    for acqui in dbRatingResult_crowd[f][item]:
                        if acqui["finalRating"]["steps"][step] == "OK":
                            crowd_korr_db += 1
                        if acqui["finalRating"]["steps"][step] == "NOK":
                            crowd_inkorr_db += 1
            print("Crowd db rating --> korrekt={}, inkorrekt={}".format(crowd_korr_db, crowd_inkorr_db))
            print("Gesamt: Crowdrating (STEP={}): korrekt = {}, inkorrekt = {} ---> korrekt = {:.2f}%".format(step, crowd_korr_db + crowd_korr_ell, crowd_inkorr_db + crowd_inkorr_ell, 100*(crowd_korr_db + crowd_korr_ell)/(crowd_korr_db + crowd_korr_ell + crowd_inkorr_db + crowd_inkorr_ell)))
   
    print("----------------------------------\n")

    ## Combine Ratings
    method = "admin"    # "crowd"
    worker_rating_admin = combine_ratings(cars_in_shds, ellRatingResult, dbRatingResult, method, config)

    if skipCrowd:
        method = "crowd"
        worker_rating_crowd = combine_ratings(cars_in_shds, ellRatingResult_crowd, dbRatingResult_crowd, method, config, step=config["rating"]["step"])

    # -----------------------------------------------------------------------------
    #
    # Check if all workers get a rating, Plot every acquisition + evaluated rating  
    #
    # -----------------------------------------------------------------------------
    
    # Crowd
    if skipCrowd:
        worker_rating = worker_rating_crowd
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            diff = []
            for idx, cur_worker in enumerate(cars_in_shds[cur_job]['workerId']):
                rating = worker_rating[cur_job][cur_worker]
                cur_acqui_count = len(rating["OK"]["idc"]) + len(rating["NOK"]["idc"])
                ori_acqui_count = len(cars_in_shds[cur_job]["job_idc_orig"][idx])
                
                # Compare gathered rating count with original acqui count
                if cur_acqui_count != ori_acqui_count: #cars_in_shds[cur_job]["acqui_count"]
                    print("Job {}, Worker {}, cur_acqui_count {}, ori_acqui_count {}".format(cur_job, cur_worker, cur_acqui_count, ori_acqui_count))
                    raise Exception("Acqui count does not match")
                # Check if every acquisition idx entry is unique
                all_ratings = rating["OK"]["idc"][:]
                all_ratings.extend(rating["NOK"]["idc"])
                all_ratings.sort()
                
                if all_ratings != [*range(ori_acqui_count)]:
                    diff.append(ori_acqui_count-cur_acqui_count)
                    raise Exception("Original input of acquisitions and resulting rating output do not match")
    # Admin
    worker_rating = worker_rating_admin
    
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        diff = []
        for idx, cur_worker in enumerate(cars_in_shds[cur_job]['workerId']):
            rating = worker_rating[cur_job][cur_worker]
            cur_acqui_count = len(rating["OK"]["idc"]) + len(rating["NOK"]["idc"])
            ori_acqui_count = len(cars_in_shds[cur_job]["job_idc_orig"][idx])
            
            # Compare gathered rating count with original acqui count
            if cur_acqui_count != ori_acqui_count: #cars_in_shds[cur_job]["acqui_count"]
                print("Job {}, Worker {}, cur_acqui_count {}, ori_acqui_count {}".format(cur_job, cur_worker, cur_acqui_count, ori_acqui_count))
                print("ERROR: Acqui count does not match")
                raise Exception("Acqui count does not match")
            # Check if every acquisition idx entry is unique
            all_ratings = rating["OK"]["idc"][:]
            all_ratings.extend(rating["NOK"]["idc"])
            all_ratings.sort()
            
            if all_ratings != [*range(ori_acqui_count)]:
                diff.append(ori_acqui_count-cur_acqui_count)
                print("ERROR: Original input of acquisitions and resulting rating output do not match")
                raise Exception("Original input of acquisitions and resulting rating output do not match")
    
    # Plot every acquisition + evaluated rating   
    worker_rating = worker_rating_admin
    # worker_rating = worker_rating_crowd
    plotAcquiRating = False
    savePlot = True    
    if plotAcquiRating:
        for cur_job in range(config["jobs"]["number_of_jobs"]):
            create_dir_if_not_exist("figures/{}/Verify_Rating/OK/".format(cur_job+1))
            create_dir_if_not_exist("figures/{}/Verify_Rating/NOK/".format(cur_job+1))
            
            for idx, workerId in enumerate(worker_rating[cur_job]):
                worker = worker_rating[cur_job][workerId]
                # Plot "OK" rated acquisitions
                acqui_idc = worker["OK"]["idc"]
                if acqui_idc:                  
                    for acqui_idx in acqui_idc:
                        idx_orig = cars_in_shds[cur_job]['job_idc_orig'][idx][acqui_idx]                    
                        
                        x = cars_in_shds[cur_job]["all"]["start"]["x"][idx_orig]
                        y = cars_in_shds[cur_job]["all"]["start"]["y"][idx_orig]
                        x2 = cars_in_shds[cur_job]["all"]["end"]["x"][idx_orig]
                        y2 = cars_in_shds[cur_job]["all"]["end"]["y"][idx_orig]
                        
                        fig2, ax = plt.subplots(1, 1)
                        fig2.suptitle("Job {}, Worker {}, Acquisition {}, Rating = OK, startX={:.2f}, startY={:.2f}, endX={:.2f}, endY={:.2f}".format(cur_job, workerId, idx_orig, x, y, x2, y2))
                        
                        # Load batch image
                        cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
                        height, width = cur_img.shape
                        extent = [-1, width-1, height-1, -1]                 
                        ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
                        
                        xmin = 50000
                        ymin = 50000
                        xmax = 0
                        ymax = 0
                        
                        if x < xmin:
                            xmin = x
                        if x2 < xmin:
                            xmin = x2
                        if x > xmax:
                            xmax = x
                        if x2 > xmax:
                            xmax = x2
                        if y < ymin:
                            ymin = y
                        if y2 < ymin:
                            ymin = y2
                        if y > ymax:
                            ymax = y
                        if y2 > ymax:
                            ymax = y2
                        ax.plot([x, x2],
                                [y, y2], color="red", linewidth=2)
                        ax.plot([], [], color="red", label=r'Acquisition')  # empty only legend label
                        ax.axis([xmin-20, xmax+20, ymin-20, ymax+20])
                        ax.set_title("Acquisition, Rating = OK, Reason = !!!!!!!!!!!!! TO DO !!")
                        ax.set_xlabel("x [px]")
                        ax.set_ylabel("y [px]")
                        ax.legend()
                        fig = plt.gcf()
                        manager = plt.get_current_fig_manager()
                        manager.resize(*manager.window.maxsize())
                        # figManager = plt.get_current_fig_manager() # for fullscreen
                        # figManager.window.state("zoomed")
                        # plt.show()
                        if savePlot:
                            fig2.savefig('figures/{}/Verify_Rating/OK/job_{}_worker_{}_acquisition_{}.png'.format(cur_job+1, cur_job+1, workerId, idx_orig))
                            plt.close("all")
                            
                # Plot "NOK" rated acquisitions
                acqui_idc = worker["NOK"]["idc"]
                reason = worker["NOK"]["reason"]
                if acqui_idc:                  
                    for idx_reason, acqui_idx in enumerate(acqui_idc):
                        idx_orig = cars_in_shds[cur_job]['job_idc_orig'][idx][acqui_idx]                    
                        
                        x = cars_in_shds[cur_job]["all"]["start"]["x"][idx_orig]
                        y = cars_in_shds[cur_job]["all"]["start"]["y"][idx_orig]
                        x2 = cars_in_shds[cur_job]["all"]["end"]["x"][idx_orig]
                        y2 = cars_in_shds[cur_job]["all"]["end"]["y"][idx_orig]
                        
                        fig2, ax = plt.subplots(1, 1)
                        fig2.suptitle("Job {}, Worker {}, Acquisition {}, Rating = NOK, reason = {}".format(cur_job, workerId, idx_orig, reason[idx_reason]))
                        
                        # Load batch image
                        cur_img = plt.imread(config["directories"]["Img_Folder"] + '/job' + str(cur_job + 1) + '/' + 'shd.png')
                        height, width = cur_img.shape
                        extent = [-1, width-1, height-1, -1]                    
                        ax.imshow(cur_img, cmap="gray", origin = "upper", extent=extent, interpolation='none')
                        
                        xmin = 50000
                        ymin = 50000
                        xmax = 0
                        ymax = 0
                        
                        if x<xmin:
                            xmin = x
                        if x2<xmin:
                            xmin = x2
                        if x>xmax:
                            xmax = x
                        if x2>xmax:
                            xmax = x2
                        if y<ymin:
                            ymin = y
                        if y2<ymin:
                            ymin = y2
                        if y>ymax:
                            ymax = y
                        if y2>ymax:
                            ymax = y2
                        ax.plot([x, x2],
                            [y, y2], color="red", linewidth= 2)
                        ax.plot([],[], color="red", label=r'Acquisition')  # empty only legend label
                        ax.axis([xmin-20,xmax+20,ymin-20,ymax+20])
                        ax.set_xlabel("x [px]")
                        ax.set_ylabel("y [px]")
                        ax.legend()
                        fig = plt.gcf()
                        manager = plt.get_current_fig_manager()
                        manager.resize(*manager.window.maxsize())
                        #figManager = plt.get_current_fig_manager() # for fullscreen
                        #figManager.window.state("zoomed")
                        
                        if savePlot:
                            fig2.savefig('figures/{}/Verify_Rating/NOK/job_{}_worker_{}_acquisition_{}.png'.format(cur_job+1, cur_job+1, workerId, idx_orig))
                            plt.close("all")  
    
    SAVE_OUTPUT = True
    if SAVE_OUTPUT is True:
        print("Saving Output in backup folder")        
        # Write cars_in_shds to file
        with open(config["backup"]["cars_in_shds"], 'wb') as file:  
            obj = (cars_in_shds)
            pickle.dump(obj, file)            
        # Write ellRatingResult to file
        with open(config["backup"]["ellRatingResult"], 'wb') as file:  
            obj = (ellRatingResult)
            pickle.dump(obj, file)
        # Write dbRatingResult to file
        with open(config["backup"]["dbRatingResult"], 'wb') as file:  
            obj = (dbRatingResult)
            pickle.dump(obj, file)
        # Write worker_rating to file
        with open(config["backup"]["worker_rating_admin"], 'wb') as file:  
            obj = (worker_rating_admin)
            pickle.dump(obj, file)
            
        if skipCrowd:
            # Write ellRatingResult to file
            with open(config["backup"]["ellRatingResult_crowd"], 'wb') as file:  
                obj = (ellRatingResult_crowd)
                pickle.dump(obj, file)
            # Write dbRatingResult to file
            with open(config["backup"]["dbRatingResult_crowd"], 'wb') as file:  
                obj = (dbRatingResult_crowd)
                pickle.dump(obj, file)                
            # Write worker_rating to file
            with open(config["backup"]["worker_rating_crowd"], 'wb') as file:  
                obj = (worker_rating_crowd)
                pickle.dump(obj, file)
    
    # -----------------------------------------------------------------------------
    #
    # Calculate final ratings for all workers by Admin- or Crowdinterface
    #
    # -----------------------------------------------------------------------------
    
    if skipCrowd:
        method = "crowd"
        if method == "crowd":        
            #step = 3
            [ OK_int_cluster_idc_crowd, approvedCluster_weak_idc_crowd, worker_rating_crowd] = calc_final_rating(cars_in_shds, worker_rating_crowd, dbRatingResult_crowd, ellRatingResult_crowd, method, config, step=config["rating"]["step"])
    
    method = "admin"
    if method == "admin":
        [ OK_int_cluster_idc_admin, approvedCluster_weak_idc_admin, worker_rating_admin] = calc_final_rating(cars_in_shds, worker_rating_admin, dbRatingResult, ellRatingResult, method, config)

    # -----------------------------------------------------------------------------
    #
    # Compute quality parameters -> FP, TP, FN, TN pre admin/crowd verification
    #
    # -----------------------------------------------------------------------------
    
    # Pre interface
    params_pre = calc_quality_params(cars_in_shds, time="pre", config=config)
    
    # Admininterface quality params
    params_post_admin = calc_quality_params(cars_in_shds, config=config, method="admin", time="post", ok_weak_cluster = approvedCluster_weak_idc_admin, ok_int_cluster = OK_int_cluster_idc_admin, db_rating_result= dbRatingResult)
    
    # Crowdinterface quality params
    if skipCrowd:
        params_post_crowd = calc_quality_params(cars_in_shds, config=config, method="crowd", time="post", ok_weak_cluster = approvedCluster_weak_idc_crowd, ok_int_cluster = OK_int_cluster_idc_crowd, db_rating_result= dbRatingResult_crowd, step=config["rating"]["step"])

    ## Plot overlap quality params
    # test save -> n=25 & n=50
    #with open("test/quality_params_pre.dat", 'wb') as file:  
    #    obj = (params_pre)
    #    pickle.dump(obj, file)
    #with open("test/quality_params_admin.dat", 'wb') as file:  
    #    obj = (params_post_admin)
    #    pickle.dump(obj, file)
    #with open("test/quality_params_crowd.dat", 'wb') as file:  
    #    obj = (params_post_crowd)
    #    pickle.dump(obj, file)
    #with open("test/n=25/quality_params_pre.dat", "rb") as file:
    #    params_pre = pickle.load(file) 
    #with open("test/n=25/quality_params_admin.dat", "rb") as file:
    #    params_post_admin = pickle.load(file) 
    #with open("test/n=25/quality_params_crowd.dat", "rb") as file:
    #    params_post_crowd = pickle.load(file)
    if skipCrowd:
        plot_overlap(params_pre, params_post_admin, params_post_crowd, config)

    # Save quality params to file for CD
    for __idx, __quality_params in enumerate ([ params_pre, params_post_admin, params_post_crowd ]):
        if __idx == 0:
            __time = "pre_ueberpruefung"
        if __idx == 1:
            __time = "post_adminueberpruefung"
        if __idx == 2:
            __time = "post_crowdueberpruefung"
        
        for __cur_job in range(config["jobs"]["number_of_jobs"]):
            __path = "plotted_data_textformat/allgemeine_daten/"
            create_dir_if_not_exist(__path)
            __fname = "zeitpunkt_{}_streifen_{}_TP_FP_FN_coordinates_etc.txt".format(__time, __cur_job + 1)

            with open(__path + __fname, "w") as f:
                #TP
                f.write("{} TP -> Koordinaten der richtig positiven Erfassungen = integriertes Endergebnis: (x_start, x_ende, y_start, y_ende)".format(np.count_nonzero(__quality_params[__cur_job]["TP"])))
                for __idx, c in enumerate(__quality_params[__cur_job]["TP"]):
                    if c:   #c==True
                        f.write("\n{},{},{},{}".format(__quality_params[__cur_job]["x"][__idx], __quality_params[__cur_job]["x2"][__idx], __quality_params[__cur_job]["y"][__idx], __quality_params[__cur_job]["y2"][__idx]))
                #FP
                f.write("\n\n{} FP -> Koordinaten der falsch positiven Erfassungen: (x_start, x_ende, y_start, y_ende)".format(np.count_nonzero(__quality_params[__cur_job]["FP"])))
                for __idx, c in enumerate(__quality_params[__cur_job]["FP"]):
                    if c:   #c==True
                        f.write("\n{},{},{},{}".format(__quality_params[__cur_job]["x"][__idx], __quality_params[__cur_job]["x2"][__idx], __quality_params[__cur_job]["y"][__idx], __quality_params[__cur_job]["y2"][__idx]))
                #FN
                f.write("\n\n{} FN -> Koordinaten der falsch negativen Erfassungen: (x_start, x_ende, y_start, y_ende)".format(np.count_nonzero(__quality_params[__cur_job]["FN"])))
                for __idx in range(len(cars_in_shds[__cur_job]["gt"]["start"]["x"])):
                    if __quality_params[__cur_job]["FN"][__idx]:  # == True
                        f.write("\n{},{},{},{}".format(cars_in_shds[__cur_job]["gt"]["start"]["x"][__idx], cars_in_shds[__cur_job]["gt"]["end"]["x"][__idx], cars_in_shds[__cur_job]["gt"]["start"]["y"][__idx], cars_in_shds[__cur_job]["gt"]["end"]["y"][__idx]))
    
    # Plot performance in- /decrease 
    # Pre <-> Admin
    plot_performance(method="pre<->post-admin", quality_params1=params_pre, quality_params2=params_post_admin, config=config)

    if skipCrowd:
        # Pre <-> Crowd
        plot_performance(method="pre<->post-crowd", quality_params1=params_pre, quality_params2=params_post_crowd, config=config)
        # Admin <-> Crowd
        plot_performance(method="post-admin<->post-crowd", quality_params1=params_post_admin, quality_params2=params_post_crowd, config=config)

    # Save final worker rating
    SAVE_OUTPUT = True
    if SAVE_OUTPUT is True:
        print("Saving result of rating data in backup folder")

        if skipCrowd:
            with open(config["backup"]["worker_rating_ready4submit_crowd"], 'wb') as file:  
                obj = (worker_rating_crowd)
                pickle.dump(obj, file)

        with open(config["backup"]["worker_rating_ready4submit_admin"], 'wb') as file:  
            obj = (worker_rating_admin)
            pickle.dump(obj, file)


def mw_submit(mw_api, slotId, rating, comment):
    """ 
    Submit rating to microworkers
    
    Parameters:
    ----------
        slot: string
            slotId
        rating: string
            "OK", "NOK"
        comment: string
            Required if rating=="NOK"            
    """
    try:
        action = "/slots/" + slotId + "/rate"
        
        if rating == "OK":
            params = {
                "rating": rating
            }
        elif rating =="NOK":
            params = {
                "rating": rating,
                "comment": comment,
            }
        response = mw_api.do_request("put", action=action, params=params)
        print("rating send\n")
    except:
        print("Error: API request to submit rating failed")


def submit_rating(config, rate_method):
    """ 
    Submit worker ratings to microworkers 
    """
    # Establish API connection
    mw_api = MW_API(config["microworkers"]["api_key"], config["microworkers"]["api_url"])   # call class MW_API
    
    # Get info for active campaign
    _, _, campaign = get_active_campaign(mw_api, method="acquisitions", config=config)
    
    # Load stored backup values
    USE_STORED_VARIABLES = True
    if USE_STORED_VARIABLES is True:                
        # Restore old values
        try:
            if rate_method == "admin":
                with open(config["backup"]["worker_rating_ready4submit_admin"], "rb") as file:
                    worker_rating = pickle.load(file) 
            if rate_method == "crowd":
                with open(config["backup"]["worker_rating_ready4submit_crowd"], "rb") as file:
                    worker_rating = pickle.load(file) 
        except:
            print("Error trying to open backup file")
    
    params = { 
            # "pageSize": 90,
            "pageSize": config["jobs"]["number_of_jobs"] * config["jobs"]["number_of_acquisitions"],
            "status": "NOTRATED"          
    }
    
    HR_camp_slots = mw_api.do_request("get", "/hire-group-campaigns/" + campaign["campaignId"] + "/slots", params=params)
    
    stopped = False
    for cur_job in range(config["jobs"]["number_of_jobs"]):
        for idx, workerId in enumerate(worker_rating[cur_job]):
            worker = worker_rating[cur_job][workerId]
            
            # Check if workerId and slotId match with the mw server
            skip = mw_match(mw_api, worker["slotId"], workerId)
            if skip:
                continue
            
            # Params to submit
            try:
                comment = worker["finalRating"][rate_method]["comment"]          # !!!!!!!!!!!!!!!!!!!!!!
                rating = worker["finalRating"][rate_method]["rating"]
            except:
                print("Error setting method [admin/crowd]")
            
            print("idx={}, rating={}".format(idx, rating))
            
            # Submit rating to microworkers
            mw_submit(mw_api, worker["slotId"], rating, comment)
            
            if rating == "NOK" and stopped == False:
                time.sleep(2)
                # Stop the campaign -> otherwise campaign status switches back to "RUNNING" when "NOK" rating is submitted and new crowdworker gets assigned to the job --------> setting: removePositionOnNokRating = True
                # SUPPORT message -> "removePositionOnNokRating":
                # If you have for example 100 positions in your campaign, with removePositionOnNokRating: false (which is default mode for MW/TTV) 
                # if you rate task as NOK, system will re-assign same task to other worker. 100 in this case means 100 OK positions.
                # With removePositionOnNokRating: true system works differently. If you rate task as NOK, system is going to reduce number of available positions by 1.
                # Suppose campaign has 100 positions, you rated 70 as OK and 30 as NOK. Campaign will be finished and number of positions will be 70.
                
                # stopCampaign = mw_api.do_request("put", "/hire-group-campaigns/" + campaign["campaignId"] + "/stop")
                # if stopCampaign["type"] == "error":
                #     raise Exception("Campaign couldnt be stopped. Campaign may be running again, error code -> {}".format(stopCampaign["value"]["args"][0]))
                # HR_camp_slots = mw_api.do_request("get", "/hire-group-campaigns/" + campaign["campaignId"])
                stopped = True


def calc_time():
    """ 
    Load edit times of crowdworker, plot time + clicks (questions data)
    """
    
    path_acqui = ["plotData/time/n25_acqui/", "plotData/time/n50_acqui/"]    
    path_quest = ["plotData/time/n25_quest/", "plotData/time/n50_quest/"]
    
    ## Questions
    timeList_quest = []; timePerAcqui = []; clickList=[]    
    
    # Load textfiles
    for path in path_quest:
        for f in glob.glob(path + "*.txt"):
            with open(f) as f:
                for line in f:
                    line = line.split("\t")
                    questCount = int(line[2])
                    time = float(line[0])   # [s]
                    timeList_quest.append(time / 60)   # [min]
                    timePerAcqui.append(time / questCount) # [s]
                    clickList.append(int(line[1]) / questCount)
    
    # Plot
    try:
        # Bearbeitungszeit für n=25 und n=50 kombiniert
        fig, ax1 = plt.subplots(1,1)
       
        min_ = np.min(timeList_quest)
        max_ = np.max(timeList_quest)
        mean_ = np.mean(timeList_quest) # [min]
        binCount = 10
        weights1 = np.ones_like(timeList_quest)/len(timeList_quest)
        x1, bins1, p1 = ax1.hist(timeList_quest, binCount, alpha=0.5, label="Crowdworker", range=[ min_, max_], weights=weights1)
        ax1.axvline(mean_, color="red", linestyle="dashed", alpha=0.65, linewidth=1, label="Mittelwert = {:.2f}min".format(mean_))
        ax1.set_title("Histogramm der Bearbeitungszeit\n(Kampagnen zur Überprüfung)")
        ax1.set_ylabel("Relative Häufigkeit")
        ax1.set_xlabel("Zeit [min]")
        ax1.legend()
        
        fig.tight_layout()            
        fig1 = plt.gcf()
            
        savePlot = True
        if savePlot:
            path = 'figures/overall_quality_parameters/'
            create_dir_if_not_exist(path) 
            
            fname = 'kombiniert_n25_n50_zeit_frage.png'.format()
            path += fname            
            fig1.savefig(path, format='png', dpi=300)
            plt.close("all")    
        
        # Bearbeitungszeit pro Frage für n=25 und n=50 kombiniert
        fig, ax2 = plt.subplots(1,1)
        min_ = np.min(timePerAcqui)
        max_ = np.max(timePerAcqui)
        mean_ = np.mean(timePerAcqui)
        weights1 = np.ones_like(timePerAcqui)/len(timePerAcqui)
        x1, bins1, p1 = ax2.hist(timePerAcqui, binCount, alpha=0.5, label="Crowdworker", range=[ min_, max_], weights=weights1)
        ax2.axvline(mean_, color="red", linestyle="dashed", alpha=0.65, linewidth=1, label="Mittelwert = {:.2f}s/Frage".format(mean_))
        ax2.set_title("Histogramm der Bearbeitungszeit pro bewerteter Erfassung\n(Kampagnen zur Überprüfung)")
        ax2.set_ylabel("Relative Häufigkeit")
        ax2.set_xlabel("Zeit [s/bewerteter Erfassung]")
        ax2.legend()
        
        fig.tight_layout()            
        fig1 = plt.gcf()
            
        savePlot = True
        if savePlot:
            path = 'figures/overall_quality_parameters/'
            create_dir_if_not_exist(path) 
            
            fname = 'kombiniert_n25_n50_zeit_pro_frage.png'.format()
            path += fname            
            fig1.savefig(path, format='png', dpi=300)
            plt.close("all")  
        
        # Klickzahl für n=25 und n=50 kombiniert
        fig, ax3 = plt.subplots(1,1)
        min_ = np.min(clickList)
        max_ = np.max(clickList)
        mean_ = np.mean(clickList)
        weights1 = np.ones_like(clickList)/len(clickList)
        x1, bins1, p1 = ax3.hist(clickList, binCount, alpha=0.5, label="Crowdworker", range=[ min_, max_], weights=weights1)     
        ax3.axvline(mean_, color="red", linestyle="dashed", alpha=0.65, linewidth=1, label="Mittelwert = {:.2f}Klicks/Frage".format(mean_))  
        ax3.set_title("Histogramm der Klicks pro bewerteter Erfassung\n(Kampagnen zur Überprüfung)")
        ax3.set_ylabel("Relative Häufigkeit")
        ax3.set_xlabel("Anzahl [Klicks/Bewertete Erfassung]")
        ax3.legend()
        
        fig.tight_layout()            
        fig1 = plt.gcf()
            
        savePlot = True
        if savePlot:
            path = 'figures/overall_quality_parameters/'
            create_dir_if_not_exist(path) 
            
            fname = 'kombiniert_n25_n50_clicks_pro_frage.png'.format()
            path += fname            
            fig1.savefig(path, format='png', dpi=300)
            plt.close("all")  
    except:
        print("Error plotting time/click distribution for question interface")
    
    
    ## Acquisitions
    timeList_acqui = []
    for path in path_acqui:
        for f in glob.glob(path + "*.txt"):
            with open(f) as f:
                for line in f:
                    time = float(line)   # [s]
                    timeList_acqui.append(time / 60)   # [min]
    
    # Plot
    try:
        fig, ax1 =  plt.subplots(1,1)
        
        min_ = np.min(timeList_acqui)
        max_ = np.max(timeList_acqui)
        mean_ = np.mean(timeList_acqui)
        weights1 = np.ones_like(timeList_acqui)/len(timeList_acqui)
        x1, bins1, p1 = ax1.hist(timeList_acqui, binCount, alpha=0.5, label="Crowdworker", range=[ min_, max_], weights=weights1)    
        ax1.axvline(mean_, color="red", linestyle="dashed", alpha=0.65, linewidth=1, label="Mittelwert = {:.2f}min".format(mean_))  
        ax1.set_title("Histogramm der Bearbeitungszeit\n(Kampagnen zur Fahrzeugerfassung)")
        ax1.set_ylabel("Relative Häufigkeit")
        ax1.set_xlabel("Zeit [min]")  
        ax1.legend()
        
        fig.tight_layout()        
            
        fig1 = plt.gcf()
            
        savePlot = True
        if savePlot:
            path = 'figures/overall_quality_parameters/'
            create_dir_if_not_exist(path) 
            
            fname = 'kombiniert_n25_n50_zeit_fahrzeugerfassung.png'.format()
            path += fname            
            fig1.savefig(path, format='png', dpi=300)
            plt.close("all")
        
    except:
        print("Error plotting time distribution for acquisition ")


def plot_error_distrib():
    # load params
    with open("plotData/Save/qualityParams_n=25_pre", "rb") as file:
        qualityParams_n25_pre = pickle.load(file)
    with open("plotData/Save/qualityParams_n=50_pre", "rb") as file:
        qualityParams_n50_pre = pickle.load(file)
    
    pos_dst_PRE_25 = qualityParams_n25_pre["pos_dst"]
    pos_dst_PRE_50 = qualityParams_n50_pre["pos_dst"]
    
    len_dst_PRE_25 = qualityParams_n25_pre["len_dst"]
    len_dst_PRE_50 = qualityParams_n50_pre["len_dst"]
    
    ori_dst_PRE_25 = qualityParams_n25_pre["ori_dst"]
    ori_dst_PRE_50 = qualityParams_n50_pre["ori_dst"]
    
    hausdorff_dst_PRE_25 = qualityParams_n25_pre["hausdorff_dst"]
    hausdorff_dst_PRE_50 = qualityParams_n50_pre["hausdorff_dst"]
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(10,14))
        
    plt.suptitle('Histogramme vor der Überprüfung für n = 25 und n = 50')
    
    binCount = 20
    ################### POS
    min_ = 0  # np.min([np.min(pos_dst_PRE), np.min(pos_dst_ADMIN), np.min(pos_dst_CROWD)])
    max_ = np.max([np.max(pos_dst_PRE_25), np.max(pos_dst_PRE_50)])
    weights1 = np.ones_like(pos_dst_PRE_25)/len(pos_dst_PRE_25)
    weights2 = np.ones_like(pos_dst_PRE_50)/len(pos_dst_PRE_50)
    x1, bins1, p1 = ax1.hist(pos_dst_PRE_25, binCount, alpha=0.5, label="n=25", range=[min_, max_], weights=weights1)  #, histtype='step', stacked=True, fill=False)
    x2, bins2, p2 = ax1.hist(pos_dst_PRE_50, binCount, alpha=0.5, label="n=50", range=[min_, max_], weights=weights2)  #, histtype='step', stacked=True, fill=False)
    ax1.set_xlabel("Positionsfehler [m]")
    ax1.set_ylabel("Relative Häufigkeit")
    ax1.legend(shadow=True, loc="upper right", handlelength=1.5)
    ################### LEN
    min_ = 0  # np.min([np.min(len_dst_PRE), np.min(len_dst_ADMIN), np.min(len_dst_CROWD)])
    max_ = np.max([np.max(len_dst_PRE_25), np.max(len_dst_PRE_50)])
    weights1 = np.ones_like(len_dst_PRE_25)/len(len_dst_PRE_25)
    weights2 = np.ones_like(len_dst_PRE_50)/len(len_dst_PRE_50)
    x1, bins1, p1 = ax2.hist(len_dst_PRE_25, binCount, alpha=0.5, label="n=25", range=[min_, max_], weights=weights1)  # , histtype='step', stacked=True, fill=False)
    x2, bins2, p2 = ax2.hist(len_dst_PRE_50, binCount, alpha=0.5, label="n=50", range=[min_, max_], weights=weights2)  # , histtype='step', stacked=True, fill=False)
    ax2.set_xlabel("Längenfehler [m]")
    ax2.set_ylabel("Relative Häufigkeit")
    ax2.legend(shadow=True, loc="upper right", handlelength=1.5)
    ################### ORI
    min_ = 0 #np.min([np.min(ori_dst_PRE), np.min(ori_dst_ADMIN), np.min(ori_dst_CROWD)])
    max_ = np.max([np.max(ori_dst_PRE_25), np.max(ori_dst_PRE_50)])
    weights1 = np.ones_like(ori_dst_PRE_25)/len(ori_dst_PRE_25)
    weights2 = np.ones_like(ori_dst_PRE_50)/len(ori_dst_PRE_50)
    x1, bins1, p1 = ax3.hist(ori_dst_PRE_25, binCount, alpha=0.5, label="n=25", range=[ min_, max_], weights=weights1)#, histtype='step', stacked=True, fill=False)
    x2, bins2, p2 = ax3.hist(ori_dst_PRE_50, binCount, alpha=0.5, label="n=50", range=[ min_, max_], weights=weights2)#, histtype='step', stacked=True, fill=False)
    ax3.set_xlabel("Orientierungsfehler [°]")
    ax3.set_ylabel("Relative Häufigkeit")
    ax3.legend(shadow=True, loc="upper right", handlelength=1.5)
    ################### Hausdorff
    min_ = 0 #np.min([np.min(hausdorff_dst_PRE), np.min(hausdorff_dst_ADMIN), np.min(hausdorff_dst_CROWD)])
    max_ = np.max([np.max(hausdorff_dst_PRE_25), np.max(hausdorff_dst_PRE_50)])
    weights1 = np.ones_like(hausdorff_dst_PRE_25)/len(hausdorff_dst_PRE_25)
    weights2 = np.ones_like(hausdorff_dst_PRE_50)/len(hausdorff_dst_PRE_50)
    x1, bins1, p1 = ax4.hist(hausdorff_dst_PRE_25, binCount, alpha=0.5, label="n=25", range=[ min_, max_], weights=weights1)#, histtype='step', stacked=True, fill=False)
    x2, bins2, p2 = ax4.hist(hausdorff_dst_PRE_50, binCount, alpha=0.5, label="n=50", range=[ min_, max_], weights=weights2)#, histtype='step', stacked=True, fill=False)
    ax4.set_xlabel("Hausdorff-Metrik [m]")
    ax4.set_ylabel("Relative Häufigkeit")
    ax4.legend(shadow=True, loc="upper right", handlelength=1.5)
    
    fig.tight_layout()
    
    plt.subplots_adjust(bottom=0.045, hspace=0.27, top=0.95, right=0.975, wspace=0.202, left=0.079)
    
    fig1 = plt.gcf()
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    
    savePlot = True
    if savePlot:
        path = 'figures/overall_quality_parameters/'
        create_dir_if_not_exist(path) 
        
        fname = 'vergleich_fehlerverteilung_vor_ueberpruefung.png'.format()
        path += fname            
        fig1.savefig(path, format='png', dpi=300)
        plt.close("all")
    
    # Write data of overlapping histograms for CD       
    __path =  "plotted_data_textformat/allgemeine_daten/"
    create_dir_if_not_exist(__path)
    __fname = "fehlerverteilungen_pos_len_ori_hausdorff.txt"
    with open(__path + __fname, "w") as f:
        # Positionsfehler
        f.write("Positionsfehler in [m]")
        f.write("\nfuer n = 25:\n")
        for ele in pos_dst_PRE_25:
            f.write("{},".format(ele))
        f.write("\nfuer n = 50:\n")
        for ele in pos_dst_PRE_50:
            f.write("{},".format(ele))
        
        f.write("\n\nLaengenfehler in [m]")
        f.write("\nfuer n = 25:\n")
        for ele in len_dst_PRE_25:
            f.write("{},".format(ele))
        f.write("\nfuer n = 50:\n")
        for ele in len_dst_PRE_50:
            f.write("{},".format(ele))
            
        f.write("\n\nOrientierungsfehler in [grad]")
        f.write("\nfuer n = 25:\n")
        for ele in ori_dst_PRE_25:
            f.write("{},".format(ele))
        f.write("\nfuer n = 50:\n")
        for ele in ori_dst_PRE_50:
            f.write("{},".format(ele))
            
        f.write("\n\nHaudorff-Metrik in [m] ")
        f.write("\nfuer n = 25:\n")
        for ele in hausdorff_dst_PRE_25:
            f.write("{},".format(ele))
        f.write("\nfuer n = 50:\n")
        for ele in hausdorff_dst_PRE_50:
            f.write("{},".format(ele))


def plot_feedback():
    """ 
    Plot pie chart for gathered feedback
    """
    
    path_acqui = "plotData/Feedback/feedback_fahrzeugerfassung.txt"
    path_quest = "plotData/Feedback/feedback_ueberpruefung.txt"
    
    feedback_categories_acqui = []
    with open(path_acqui) as f:
        next(f) # Skip header
        for line in f:
            feedback_categories_acqui.append(line.split("\t")[0])
    
    feedback_categories_quest = []
    with open(path_quest) as f:
        next(f) # Skip header
        for line in f:
            feedback_categories_quest.append(line.split("\t")[0])
    
    keys_acqui = [*Counter(feedback_categories_acqui).keys()]    
    vals_acqui = [*Counter(feedback_categories_acqui).values()]
    
    comb = list(zip(keys_acqui, vals_acqui))
    comb.sort()
    comb = list(zip(*comb)) # unzip
    keys_acqui = list(comb[0])
    vals_acqui = list(comb[1])
    
    for idx, key in enumerate(keys_acqui):
        if key == "1":
            keys_acqui[idx] = "Positives Feedback"
        if key == "2":
            keys_acqui[idx] = "Negatives Feedback"
        if key == "3":
            keys_acqui[idx] = "Verbesserungsvorschläge"
        if key == "4":
            keys_acqui[idx] = "Spam"    
    
    keys_quest = [*Counter(feedback_categories_quest).keys()]
    vals_quest = [*Counter(feedback_categories_quest).values()]
    
    comb = list(zip(keys_quest, vals_quest))
    comb.sort()
    comb = list(zip(*comb)) # unzip
    keys_quest = list(comb[0])
    vals_quest = list(comb[1])

    for idx, key in enumerate(keys_quest):
        if key == "1":
            keys_quest[idx] = "Positives Feedback"
        if key == "2":
            keys_quest[idx] = "Negatives Feedback"
        if key == "3":
            keys_quest[idx] = "Verbesserungsvorschläge"
        if key == "4":
            keys_quest[idx] = "Spam"

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    explode = (0.1, 0, 0)
    
    pie1 = ax1.pie(vals_acqui, autopct='%1.1f%%', explode=explode)
    pie2 = ax2.pie(vals_quest, autopct='%1.1f%%', explode=explode)
    
    ax1.set_title("Feedback für Kampagnen\nder Fahrzeugerfassung")
    ax2.set_title("Feedback für Kampagnen\nder Überprüfung unsicherer Cluster")
    
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    
    if keys_acqui == keys_quest:
        fig.legend([pie1, pie2], labels=keys_acqui, fancybox=True, shadow=True, loc='lower center', title="Kategorien")
    else:
        print("need to update feedback chart manually")
        
    fig.tight_layout()
    
    path = 'plotData/Feedback/'    
    fname = 'feedback.png'.format()
    path += fname            
    fig.savefig(path, format='png', dpi=300)

    plt.clf()
    plt.close()
    

if __name__ == "__main__":
    
    plt.rcParams.update({'font.size': 18})
    # Plot legend
    fig, (ax1) = plt.subplots(1,1)    
    ax1.plot([],[], color="red", label=r'Erfassung')  # empty only legend label ax1.plot([],[], c="blue", label=r'Mittelpunkt einer Erfassung')  # empty only legend label
        
    ax1.scatter([], [], c='blue', s=25, label=r'Mittelpunkt einer Erfassung')   # r'$\overline{X_i}, \overline{Y_i}$')
    # ax1.scatter([40], [40], c='red', s=25, marker="D", label=r'Mittelpunkt der Ellipse')
    # ax1.add_patch(Rectangle((0,0), 1,1, edgecolor="red", alpha=0.3, facecolor='pink', label="3"+r'$\sigma$'))
    # rect1 = Rectangle((0,0), 1,1, edgecolor="red", alpha=0.3, facecolor='pink')
    # ax1.add_patch(Rectangle([], [], [], fill=False, linestyle="-", linewidth=1, edgecolor="red", alpha=0.3, facecolor='pink'))
    
    ax1.legend(fancybox=True, shadow=True, handlelength=1)
    
    plt.savefig("legende1.png", format='png', dpi=500, bbox_inches='tight')
    plt.show()
    
    # Plot final integrated results   

    # path_admin_25 = "plotted_data_textformat/n = 25/allgemeine_daten/erfolgsquoten_admin.txt"
    path_crowd_25 = "plotted_data_textformat/n = 25/allgemeine_daten/erfolgsquoten_crowd.txt"
    
    # path_admin_50 = "plotted_data_textformat/n = 50/allgemeine_daten/erfolgsquoten_admin.txt"
    path_crowd_50 = "plotted_data_textformat/n = 50/allgemeine_daten/erfolgsquoten_crowd.txt"
    
    counter = 0
    with open(path_crowd_50) as f:
        next(f)
        for line in f:
            print(counter)
            if counter == 0:
                erfolgsrate_crowd_50 = line.split(",")     # basierend auf integrierter Fahrzeuganzahl
                erfolgsrate_crowd_50.pop()
                erfolgsrate_crowd_50 = [float(i) for i in erfolgsrate_crowd_50]
            # if counter == 3:
            #    erfolgsrate_crowd = line.split(",")     # basierend auf Fahrzeuganzahl der Referenz
            #    erfolgsrate_crowd.pop()
            #    erfolgsrate_crowd = [float(i) for i in erfolgsrate_crowd]
            counter += 1
    
    counter = 0
    with open(path_crowd_25) as f:
        next(f)
        for line in f:
            print(counter)
            if counter == 0:
                erfolgsrate_crowd_25 = line.split(",")     # basierend auf integrierter Fahrzeuganzahl
                erfolgsrate_crowd_25.pop()
                erfolgsrate_crowd_25 = [float(i) for i in erfolgsrate_crowd_25]
            # if counter == 3:
            #    erfolgsrate_crowd = line.split(",")     # basierend auf Fahrzeuganzahl der Referenz
            #    erfolgsrate_crowd.pop()
            #    erfolgsrate_crowd = [float(i) for i in erfolgsrate_crowd]
            counter += 1
    
    # Plot Erfolgsquote
    fig, (ax1) = plt.subplots(1, 1)
    ax1.hist(erfolgsrate_crowd_50, len(erfolgsrate_crowd_50), alpha=0.5, label=r'Crowdworker$_{n=50}$', color="#1f77b4", range=[0, 100])
    # ax1.hist(erfolgsrate_crowd_25, len(erfolgsrate_crowd_25), alpha=0.5, label=r'Crowdworker$_{n=25}$', color="#2ca02c", range=[0, 100])
    
    ax1.set_ylabel("Absolute Häufigkeit")
    ax1.set_xlabel("Erfolgsquote")
    ax1.legend(fancybox=True, shadow=True, handlelength=1.5)

    def x_fmt(x, y):
        return "{:.0f}%".format(x)
    ax1.xaxis.set_major_locator(ticker.AutoLocator())
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(x_fmt))

    fig.tight_layout()
    plt.savefig("vortrag_erfolgsquote_n50_crowd.png", format='png', dpi=500, bbox_inches='tight')
    
    plt.show()
    
    path_cost_n50 = "plotted_data_textformat/n = 50/allgemeine_daten/kampagnenkosten_crowd.txt"
    
    from ast import literal_eval as make_tuple
    counter = 0
    with open(path_cost_n50) as f:
        next(f)
        for line in f:
            if counter == 0:
                camp1_2_cost = line.split(",(")
                camp1_2_cost[-1] = re.sub(",\n", "", camp1_2_cost[-1])
                
                for idx, ele in enumerate(camp1_2_cost):
                    if ele[0] != "(":
                        camp1_2_cost[idx] = "(" + ele
                    camp1_2_cost[idx] = make_tuple(camp1_2_cost[idx])    
            if counter == 3:
                camp1_cost = line.split(",(")
                if camp1_cost[-1][-1] == ",":                    
                    camp1_cost[-1] = camp1_cost[-1][:-1]
                
                for idx, ele in enumerate(camp1_cost):
                    if ele[0] != "(":
                        camp1_cost[idx] = "(" + ele
                    camp1_cost[idx] = make_tuple(camp1_cost[idx]) 
            counter += 1
    
    x_1, y_1 = zip(*camp1_cost)
    x, y = zip(*camp1_2_cost)
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(x_1, y_1, alpha=0.5, label=r'Kosten der Fahrzeugerfassung (Kampagne 1)', color="darkorange")
    ax1.plot(x, y, alpha=0.5, label=r'Gesamtkosten (Erfassung + Überprüfung)', color="darkgreen")
    
    ax1.plot([0, 100], [33.0, 33.0], color="k", linestyle="-", markersize=1, label="Maximale Kosten der Fahrzeugerfassung")    
    ax1.plot(33.7, 33.0, "o", markerfacecolor="red", markeredgecolor="white", markersize="8", label='Schnittpunkt (33,7%, 33,0$)')

    ax1.annotate("74,7% werden bezahlt", xy=(33.7, 33.5), xycoords="data", xytext=(33.7, 38.0), textcoords="data", size=17, va="center", ha="center", arrowprops=dict(arrowstyle="-|>"))

    ax1.set_ylabel("Kampagnenkosten")
    ax1.set_xlabel("Erfolgsquote")
    
    # plt.subplots_adjust(bottom=0.045, hspace=0.27, top=0.95, right=0.975, wspace=0.202, left=0.079)
    
    ax1.legend(fancybox=True, shadow=True, handlelength=1.5, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def x_fmt(x, y):
        return "{:.0f}%".format(x)
    ax1.xaxis.set_major_locator(ticker.AutoLocator())
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(x_fmt))
    
    def y_fmt(x, y):
        return "{:.0f}$".format(x)
    ax1.yaxis.set_major_locator(ticker.AutoLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
    
    # fig.tight_layout()
    plt.savefig("vortrag_Kampagnenkosten_n50_crowd.png", format='png', dpi=500, bbox_inches='tight')
    
    plt.show()
    
    print("xsasda")

    # Plot error distrib -> n=25 vs n=50 pre verification
    plot_error_distrib()

    # Plot top countries of hired crowdworkers
    plot_pie_chart("plotData/Herkunft/countries_erfassung_n=25.txt", "plotData/Herkunft/countries_erfassung_n=50.txt", "plotData/Herkunft/countries_quest.txt")
    
    # Plot time distrib for acqui & quest
    calc_time()
