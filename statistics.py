import math
from collections import Counter

import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from tabulate import tabulate
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pickle


def occurrences(lis):
    counter = sorted(Counter(lis).items())
    x = [i for i, k in counter]
    prob = [k for i, k in counter]
    prob = [val / sum(Counter(lis).values()) for val in prob]
    prob = 1 - np.cumsum(prob)
    return x, prob


def ecdf(lis):
    """Calculate empiric cumulative density function of list"""
    x, prob = occurrences(lis)
    plt.plot(x, prob)
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('size')
    plt.savefig('GMH_ecdf_size_log.png')


def scatter(lis1, lis2, stars, loglog=False):
    """Create scatterplot of two lists"""
    x = lis1
    y = lis2

    # for i in range(len(stars)):
    #     if stars[i] < 70:
    #         x[i] = np.nan
    #         y[i] = np.nan

    x = np.array(x)
    x = x[~np.isnan(x)]
    y = np.array(y)
    y = y[~np.isnan(y)]
    x = nonnegative_test(x)
    y = nonnegative_test(y)

    plt.scatter(x, y)
    plt.title('Size vs Forks loglog')
    plt.xlabel('forks')
    plt.ylabel('size')
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    # plt.grid()
    # plt.savefig('D22-23_size_comparison.png')

    plt.show()


def hexbin(lis1, lis2, stars):
    """Create logarithmic heatmap plot of two lists"""
    x = lis1
    y = lis2

    for i in range(len(stars)):
        if stars[i] < 70:
            x[i] = np.nan
            y[i] = np.nan

    x = np.array(x)
    x = x[~np.isnan(x)]
    y = np.array(y)
    y = y[~np.isnan(y)]
    x = nonnegative_test(x)
    y = nonnegative_test(y)
    hb = plt.hexbin(x, y, gridsize=500, bins="log", cmap='inferno')
    #plt.ylim(0, 10000)
    #plt.xlim(0, 50000)
    # , xscale='log', yscale='log'
    plt.title("forks vs stars")
    plt.xlabel('stars')
    plt.ylabel('forks')
    cb = plt.colorbar(hb, label='log10(N)')
    # plt.axhline(y=1.5, color='r', linestyle='-')
    # plt.axhline(y=2600, color='r', linestyle='-')
    # plt.axvline(x=10, color='r', linestyle='-')
    # plt.axvline(x=2600, color='r', linestyle='-')

    # line of best fit
    xnew = np.log10(x)
    ynew = np.log10(y)
    # b, m = np.polyfit(xnew, ynew, 1)
    # plt.plot(x, np.power(10, b) * np.power(x, m))
    # plt.ylim(1, 10**9)

    plt.show()


def nonnegative_test(lis1):
    """
    Test for nonnegativity in list
    If all elements are nonnegative, return list
    Else, add 1 to all elements
    """
    dummy = lis1
    for i in lis1:
        if i <= 0:
            dummy = [j + 1 for j in lis1]
            return dummy
    return dummy


def pearson_corr(lis1, lis2, stars):
    x = lis1
    y = lis2
    for i in range(len(stars)):
        if stars[i] < 70:
            x[i] = np.nan
            y[i] = np.nan

    x = np.array(x)
    x = x[~np.isnan(x)]
    y = np.array(y)
    y = y[~np.isnan(y)]
    x = nonnegative_test(x)
    y = nonnegative_test(y)
    corr, _ = pearsonr(x, y)
    return corr


def spearman_corr(lis1, lis2, stars):
    x = lis1
    y = lis2
    for i in range(len(stars)):
        if stars[i] < 70:
            x[i] = np.nan
            y[i] = np.nan

    x = np.array(x)
    x = x[~np.isnan(x)]
    y = np.array(y)
    y = y[~np.isnan(y)]
    x = nonnegative_test(x)
    y = nonnegative_test(y)
    corr, _ = spearmanr(x, y)
    return corr


def get_lists(file):
    column_names = ["Full_Names", "Stars", "forks", "commits", "Topics", "Size"]
    df = pandas.read_csv(file, names=column_names)

    full_names = df.Full_Names.to_list()
    stars = df.Stars.to_list()
    forks = df.forks.to_list()
    commits = df.commits.to_list()
    topics = df.commits.to_list()
    size = df.Size.to_list()

    del full_names[0]
    del stars[0]
    del forks[0]
    del commits[0]
    del topics[0]
    del size[0]

    stars = [int(star) for star in stars]
    forks = [int(num_fork) for num_fork in forks]
    commits = [int(num_commit) for num_commit in commits]
    size = [int(siz) for siz in size]

    return full_names, stars, forks, commits, topics, size


def histogram1d(lis):
    plt.hist(lis, bins=3000)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("count")
    plt.xlabel("forks")
    plt.title("Forks 1-D Histogram")
    plt.savefig("SGMH_hist1d_forks_loglog.png")


def dbscan2d(full_names, x, y):
    """Perform dbscan on two lists and plot the results"""
    # Preprocess lists
    x = nonnegative_test(x)
    x = np.asarray(x)
    y = nonnegative_test(y)
    y = np.asarray(y)
    x_log = np.log10(x)
    y_log = np.log10(y)

    # Compute DBSCAN
    data = list(zip(x_log, y_log))
    data = np.asarray(data)
    db = DBSCAN(eps=0.15, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # write noisy repositories to a file
    noise = []
    for i in range(len(labels)):
        if labels[i] == -1:
            noise.append(full_names[i])

    with open('noisy_reps_forks_stars.txt', 'w') as f:
        for rep in noise:
            f.write(rep)
            f.write('\n')

    # Plot
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))

    # Plot results
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title("Forks vs Stars")
    plt.xlabel("stars")
    plt.ylabel("forks")
    plt.savefig("SGMH_DBSCAN_forks_stars.png")
    plt.show()


def dbscan_all(full_names, stars, forks, commits, size):
    """Perform dbscan on all four lists"""
    full_names = np.asarray(full_names)
    stars = nonnegative_test(stars)
    stars = np.asarray(stars)
    forks = nonnegative_test(forks)
    forks = np.asarray(forks)
    commits = nonnegative_test(commits)
    commits = np.asarray(commits)
    size = nonnegative_test(size)
    size = np.asarray(size)

    stars_log = np.log10(stars)
    forks_log = np.log10(forks)
    commits_log = np.log10(commits)
    size_log = np.log10(size)

    data = list(zip(stars_log, forks_log, commits_log, size_log))
    data = np.asarray(data)

    db = DBSCAN(eps=0.55, min_samples=10).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    outlier_reps = np.sort(full_names[labels == -1])

    with open('noisy_reps.txt', 'w') as f:
        for rep in outlier_reps:
            f.write(rep)
            f.write('\n')

    return data, labels, n_clusters_


def analyze_dbscan_all(data, labels, n_clusters):
    """Analyze results of DBSCAN on four lists"""
    data = np.power(10, data)  # undo log_10 in data
    undo_minus = np.array([0, 1, 0, 1])  # forks and size at index 1 and 3 of the array are log(x+1), undo the +1

    # separates each cluster in the data (including outliers) and subtracts the one from forks and size
    clusters = [data[labels == i] - undo_minus for i in range(n_clusters)]
    outliers = data[labels == -1] - undo_minus
    clusters.append(outliers)

    mean = [np.mean(cluster, axis=0) for cluster in clusters]
    med = [np.median(cluster, axis=0) for cluster in clusters]
    std = [np.std(cluster, axis=0) for cluster in clusters]
    min = [np.amin(cluster, axis=0) for cluster in clusters]
    max = [np.amax(cluster, axis=0) for cluster in clusters]

    print(tabulate(mean, headers=["Index", "Stars", "Forks", "Commits", "Size"], showindex='always'))


def isolation_forest_all(full_names, stars, forks, commits, size):
    """Perform isolation forest anomaly detection on all four lists"""
    full_names = np.asarray(full_names)
    stars = nonnegative_test(stars)
    stars = np.asarray(stars)
    forks = nonnegative_test(forks)
    forks = np.asarray(forks)
    commits = nonnegative_test(commits)
    commits = np.asarray(commits)
    size = nonnegative_test(size)
    size = np.asarray(size)

    stars_log = np.log10(stars)
    forks_log = np.log10(forks)
    commits_log = np.log10(commits)
    size_log = np.log10(size)

    data = list(zip(stars_log, forks_log, commits_log, size_log))
    data = np.asarray(data)

    clf = IsolationForest()


def best_fit(x, y):
    x = nonnegative_test(x)
    x = np.asarray(x)
    y = nonnegative_test(y)
    y = np.asarray(y)

    x_log = np.log10(x)
    y_log = np.log10(y)
    m, b = np.polyfit(x_log, y_log, 1)

    plt.scatter(x_log, y_log, c='black')
    plt.plot(x_log, m * x_log + b)
    plt.axvline(x=2, color='g', linestyle='-')
    plt.axvline(x=2.5, color='g', linestyle='-')
    plt.axvline(x=3, color='g', linestyle='-')
    plt.axvline(x=3.5, color='g', linestyle='-')
    plt.axvline(x=4, color='g', linestyle='-')
    plt.axvline(x=4.5, color='g', linestyle='-')
    plt.axvline(x=5, color='g', linestyle='-')

    plt.title("Forks vs Stars")
    plt.xlabel("stars (logx)")
    plt.ylabel("forks (log(y+1)")
    plt.show()


def get_outliers(lis1, cutoff11, cutoff12, lis2, cutoff21, cutoff22, fullnames):
    count = 0
    for i in range(len(fullnames)):
        if cutoff11 < lis1[i] < cutoff12 and cutoff21 < lis2[i] < cutoff22:
            print(fullnames[i], lis1[i], lis2[i])
            count += 1
    print(count)


def comparison(old_full_names, full_names, lis1, lis2):
    ordered_lis2 = []
    count = 0
    for name in old_full_names:
        count += 1
        if name in full_names:
            for j in range(len(full_names)):
                if name == full_names[j]:
                    ordered_lis2.append(lis2[j])
                    break
        else:
            ordered_lis2.append(None)

    return ordered_lis2

if __name__ == '__main__':
    full_names1, stars1, forks1, commits1, topics1, size1 = get_lists("1000_stars_2023_new.csv")
    full_names2, stars2, forks2, commits2, topics2, size2 = get_lists("1000_stars_2023_update.csv")
    full_names3, stars3, forks3, commits3, topics3, size3 = get_lists("300..1000_stars_2023_new.csv")
    full_names4, stars4, forks4, commits4, topics4, size4 = get_lists("300..1000_stars_2023_update.csv")
    full_names5, stars5, forks5, commits5, topics5, size5 = get_lists("200..299_stars_2023_new.csv")
    full_names6, stars6, forks6, commits6, topics6, size6 = get_lists("200..299_stars_2023_update.csv")
    full_names7, stars7, forks7, commits7, topics7, size7 = get_lists("150..199_stars_2023_new.csv")
    full_names8, stars8, forks8, commits8, topics8, size8 = get_lists("150..199_stars_2023_update.csv")
    full_names9, stars9, forks9, commits9, topics9, size9 = get_lists("100..149_stars_2023_new.csv")
    full_names10, stars10, forks10, commits10, topics10, size10 = get_lists("100..149_stars_2023_update.csv")
    full_names11, stars11, forks11, commits11, topics11, size11 = get_lists("90..100_stars_2023_new.csv")
    full_names12, stars12, forks12, commits12, topics12, size12 = get_lists("90..99_stars_2023_update.csv")
    full_names13, stars13, forks13, commits13, topics13, size13 = get_lists("80..89_stars_2023_new.csv")
    full_names14, stars14, forks14, commits14, topics14, size14 = get_lists("80..89_stars_2023_update.csv")
    full_names15, stars15, forks15, commits15, topics15, size15 = get_lists("70..79_stars_2023_new.csv")
    full_names16, stars16, forks16, commits16, topics16, size16 = get_lists("70..79_stars_2023_update.csv")

    old_full_names1, old_stars1, old_forks1, old_commits1, old_topics1, old_size1 = get_lists("1000_stars.csv")
    old_full_names2, old_stars2, old_forks2, old_commits2, old_topics2, old_size2 = get_lists("300..1000_stars.csv")
    old_full_names3, old_stars3, old_forks3, old_commits3, old_topics3, old_size3 = get_lists("200..299_stars.csv")
    old_full_names4, old_stars4, old_forks4, old_commits4, old_topics4, old_size4 = get_lists("150..199_stars.csv")
    old_full_names5, old_stars5, old_forks5, old_commits5, old_topics5, old_size5 = get_lists("100..149_stars.csv")
    old_full_names6, old_stars6, old_forks6, old_commits6, old_topics6, old_size6 = get_lists("90..100_stars.csv")
    old_full_names7, old_stars7, old_forks7, old_commits7, old_topics7, old_size7 = get_lists("80..89_stars.csv")
    old_full_names8, old_stars8, old_forks8, old_commits8, old_topics8, old_size8 = get_lists("70..79_stars.csv")

    count = 0
    for i in range(len(stars11)):
        if stars11[i] == 100:
            del full_names11[i - count]
            del forks11[i - count]
            del commits11[i - count]
            del topics11[i - count]
            del size11[i - count]
            count += 1

    count = 0
    for i in range(len(old_stars6)):
        if old_stars6[i] == 100:
            del old_full_names6[i - count]
            del old_forks6[i - count]
            del old_commits6[i - count]
            del old_topics6[i - count]
            del old_size6[i - count]
            count += 1

    stars11 = [value for value in stars11 if value != 100]
    old_stars6 = [value for value in old_stars6 if value != 100]

    full_names = full_names1 + full_names2 + full_names3 + full_names4 + full_names5 + full_names6 + full_names7 + \
                 full_names8 + full_names9 + full_names10 + full_names11 + full_names12 + full_names13 + full_names14 + \
                 full_names15 + full_names16
    stars = stars1 + stars2 + stars3 + stars4 + stars5 + stars6 + stars7 + stars8 + stars9 + stars10 + stars11 + \
            stars12 + stars13 + stars14 + stars15 + stars16
    forks = forks1 + forks2 + forks3 + forks4 + forks5 + forks6 + forks7 + forks8 + forks9 + forks10 + forks11 + \
            forks12 + forks13 + forks14 + forks15 + forks16
    commits = commits1 + commits2 + commits3 + commits4 + commits5 + commits6 + commits7 + commits8 + commits9 + \
              commits10 + commits11 + commits12 + commits13 + commits14 + commits15 + commits16
    topics = topics1 + topics2 + topics3 + topics4 + topics5 + topics6 + topics7 + topics8 + topics9 + topics10 + \
             topics11 + topics12 + topics13 + topics14 + topics15 + topics16
    size = size1 + size2 + size3 + size4 + size5 + size6 + size7 + size8 + size9 + size10 + size11 + size12 + size13 + \
           size14 + size15 + size16

    old_full_names = old_full_names1 + old_full_names2 + old_full_names3 + old_full_names4 + old_full_names5 + \
                     old_full_names6 + old_full_names7 + old_full_names8
    old_stars = old_stars1 + old_stars2 + old_stars3 + old_stars4 + old_stars5 + \
                old_stars6 + old_stars7 + old_stars8
    old_forks = old_forks1 + old_forks2 + old_forks3 + old_forks4 + old_forks5 + \
                old_forks6 + old_forks7 + old_forks8
    old_commits = old_commits1 + old_commits2 + old_commits3 + old_commits4 + old_commits5 + \
                  old_commits6 + old_commits7 + old_commits8
    old_topics = old_topics1 + old_topics2 + old_topics3 + old_topics4 + old_topics5 + \
                 old_topics6 + old_topics7 + old_topics8
    old_size = old_size1 + old_size2 + old_size3 + old_size4 + old_size5 + \
               old_size6 + old_size7 + old_size8