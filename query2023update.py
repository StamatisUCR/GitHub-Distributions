import os
import sys
import traceback

import github
import pandas
from github import Github
import wget
import config
import time
from pandas import DataFrame

full_names = []
stars = []
stargazers_users = []
stargazers_login_times = []
contributors_count = []
contributors_users = []
num_contributors = []
num_forks = []
num_commits = []
topics = []
size = []


def save_work():
    df = DataFrame(
        {'Full_names': full_names, 'Stars': stars,
         'Num_Forks': num_forks, 'Num_Commits': num_commits,
         'Topics': topics, 'Size': size})

    df.to_csv('1000_stars_2023_August.csv')


def query(auth_key, query_list, full_names, full_names_previous):
    """Loop through all queries and perform get_rep"""
    for query in query_list:
        print("Getting repos with query: " + query)

        repos = auth_key.search_repositories(query=query, sort="stars")

        get_rep_info(repos, full_names, full_names_previous)


def get_rep_info(repos, full_names, full_names_previous):  # 16 requests + 2 * num_stars + 1 * num_contributors
    """Given a set of repositories from Github API,
    puts their name in one text file and downloads
    zip files of their default branch to current directory"""

    count = 0
    try:
        for repo in repos:
            count += 1
            time.sleep(1)
            # info

            full_name = repo.full_name

            if full_name in full_names_previous or full_name in full_names:
                count -= 1
                continue

            time.sleep(1)
            print(count, full_name)
            num_commits.append(repo.get_commits().totalCount)
            stars.append(repo.stargazers_count)
            num_forks.append(repo.get_forks().totalCount)
            topics.append(repo.get_topics())
            size.append(repo.size)

            # stargazers_count, stargazers_user, stargazers_login_time = get_stargazers_list(repo)  # 2 requests + 2 per star
            # stars.append(stargazers_count)
            # stargazers_users.append(stargazers_user)
            # stargazers_login_times.append(stargazers_login_time)

            # num_contributor, contributors_user = get_contributors_list(repo)  # 2 requests + 1 per contributor
            # num_contributors.append(num_contributors)
            # contributors_count.append(num_contributors)
            # contributors_users.append(contributors_user)
            # download
            """
            default_branch = repo.default_branch
            
            try:
                download_url = "https://" + repo.url[12:22] + repo.url[
                                                              28:] + "/archive/refs/heads/" + default_branch + ".zip"
                path = "C:\\Users\\16507\\2021_Research\\Track2\\repositories1k"
                repo_full_name = repo.owner.login + "--" + repo.name
                os.mkdir(os.path.join(path, repo_full_name))
                dest = "C:\\Users\\16507\\2021_Research\\Track2\\repositories1k" + "\\" + repo_full_name
                print("Downloading: " + download_url)
                wget.download(download_url, out=dest)
            except FileExistsError:
                continue
            """
            full_names.append(full_name)
            print(len(full_names), len(stars), len(num_forks), len(num_commits), len(topics), len(size))
            save_work()
    except Exception as e:
        tb = traceback.format_exc()
        if "API rate limit exceeded for user ID 75396346" in tb:
            print(e)
            sys.exit(1)
        if "The listed users and repositories cannot be searched either because the resources do not exist or you do " \
           "not have permission to view them" or "Git Repository is empty" in tb:
            time.sleep(2)
            print("No permission or empty")
            print(e)

            if len(stars) > len(full_names):
                stars.pop()
            if len(num_forks) > len(full_names):
                num_forks.pop()
            if len(num_commits) > len(full_names):
                num_commits.pop()
            if len(topics) > len(full_names):
                num_forks.pop()
            if len(size) > len(full_names):
                size.pop()
            full_names.append(full_name)
            stars.append(0)
            num_forks.append(0)
            num_commits.append(0)
            topics.append([])
            size.append(0)
            print(len(full_names), len(stars), len(num_forks), len(num_commits), len(topics), len(size))
            save_work()
        else:
            print(e)


def get_stargazers_list(repo):  # 2 requests + 2 per star
    stars_list = repo.get_stargazers_with_dates()  # 1 request
    number_of_stargazers = stars_list.totalCount  # 1 request

    stargazers_user = []
    stargazers_login_time = []

    for i in range(0, number_of_stargazers):
        user = stars_list[i].user.login  # 1 request
        star_time = stars_list[i].starred_at  # 1 request
        stargazers_user.append(user) if user is not None else stargazers_user.append("None")
        stargazers_login_time.append(star_time.strftime("%m/%d/%Y, %H:%M:%S"))

    return number_of_stargazers, stargazers_user, stargazers_login_time


def get_contributors_list(repo):  # 2 requests + 1 per contributor
    contributors_list = repo.get_contributors()  # 1 request
    number_of_contributors = contributors_list.totalCount  # 1 request

    contributors_user = []

    for i in range(0, number_of_contributors):
        login = contributors_list[i].login  # 1 request
        contributors_user.append(login) if login is not None else contributors_user.append("None")

    return number_of_contributors, contributors_user


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


if __name__ == '__main__':
    # create config.py file with attribute api_key that is a string of Github PTA
    g = Github(config.api_key1)

    try:
        full_names, stars, num_forks, num_commits, topics, size = get_lists("1000_stars_2023_update.csv")
    except Exception:
        print(Exception)

    full_names_previous, stars_previous, forks_previous, commits_previous, topics_previous, size_previous = get_lists("1000_stars_2023_final.csv")

    query_list = [
        "language:python stars:>1000 created:2000-01-01..2012-01-01",
        "language:python stars:>1000 created:2012-01-02..2013-01-01",
        "language:python stars:>1000 created:2013-01-02..2014-01-01",
        "language:python stars:>1000 created:2014-01-02..2015-01-01",
        "language:python stars:>1000 created:2015-01-02..2015-11-01",
        "language:python stars:>1000 created:2015-11-02..2016-07-01",
        "language:python stars:>1000 created:2016-07-02..2017-01-01",
        "language:python stars:>1000 created:2017-01-02..2017-07-01",
        "language:python stars:>1000 created:2017-07-02..2018-01-01",
        "language:python stars:>1000 created:2018-01-02..2018-06-01",
        "language:python stars:>1000 created:2018-06-02..2019-01-01",
        "language:python stars:>1000 created:2019-01-02..2019-06-01",
        "language:python stars:>1000 created:2019-06-02..2020-01-01",
        "language:python stars:>1000 created:2020-01-02..2020-06-01",
        "language:python stars:>1000 created:2020-06-02..2021-01-01",
        "language:python stars:>1000 created:2021-01-02..2021-06-01",
        "language:python stars:>1000 created:2021-06-02..2022-01-01",
        "language:python stars:>1000 created:2022-01-02..2022-06-01"
        "language:python stars:>1000 created:2022-06-02..2023-01-01"
        "language:python stars:>1000 created:2023-01-02..2023-06-01"
        "language:python stars:>1000 created:>2023-06-02"
    ]

    query(g, query_list, full_names, full_names_previous)