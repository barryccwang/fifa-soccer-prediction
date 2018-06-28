#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


FIFA_2018_TEAMS = ['Russia', 'Saudi Arabia', 'Egypt', 'Uruguay', 'Portugal', 'Spain', 'Morocco', 'Iran', 'France', 'Australia', 'Peru', 'Denmark', 'Argentina', 'Iceland', 'Croatia', 'Nigeria', 'Brazil', 'Switzerland', 'Costa Rica', 'Serbia', 'Germany', 'Mexico', 'Sweden', 'Korea Republic', 'Belgium', 'Panama', 'Tunisia', 'England', 'Poland', 'Senegal', 'Colombia', 'Japan']


PARSER = argparse.ArgumentParser(
    description='FIFA 2018 Soccer Team Prediction',
    usage='python %(prog)s "Russia" "Saudi Arabia"')

PARSER.add_argument(
    'teams',
    nargs=2,
    choices=FIFA_2018_TEAMS,
    metavar='soccer teams',
    help='FIFA 2018 participating teams')


def urlretrieve(url, path):
    import ssl
    if hasattr(ssl, 'create_default_context'):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx = None

    import urllib2
    import inspect
    args = inspect.getargspec(urllib2.urlopen).args
    if 'context' in args:
        res = urllib2.urlopen(url, timeout=10, context=ctx)
    else:
        res = urllib2.urlopen(url, timeout=10)
    if res.getcode() != 200:
        raise Exception('failed to urlretrieve, url:{url}, returncode:{rt}, info:{info}'.format(
            url=url, rt=res.getcode(), info=res.info()))
    with open(path, 'wb+') as f:
        f.write(res.read())
    res.close()


def generate_team_ranking():
    # Load soccer team ranking
    # For more detailed, refer to
    # https://www.kaggle.com/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now
    team_ranking = pd.read_csv('datasets/fifa_rankings.csv')
    team_ranking = team_ranking[team_ranking['Team'].isin(FIFA_2018_TEAMS)]

    # Load soccer player stats.
    # For more detailed, refer to
    # https://www.kaggle.com/sawya34/football-world-cup-2018-dataset/data
    team_players = pd.read_csv('datasets/players.csv')

    # Load soccer player ranking.
    player_scores = pd.read_csv('datasets/player_scores.csv')

    team_players.insert(0, 'rating', team_players['player'].map(player_scores.set_index('player')['Rating']))

    team_player_scores = []
    for team in team_ranking['Team']:
        players = team_players[team_players.nationality == team]
        scores = players['rating'].dropna().get_values()
        if len(scores):
            team_player_scores.append(round(sum(scores) / len(scores), 2))
        else:
            team_player_scores.append(0)
    team_ranking.insert(3, 'Player Avg Points', team_player_scores)
    team_ranking['Points'] = team_ranking['Points'].apply(lambda point: round(point / 100))
    team_ranking = team_ranking.sort_values(
        by=['Points', 'Player Avg Points'], ascending=False).reset_index(drop=True)
    team_ranking['Position'] = team_ranking.index
    return team_ranking


def preprocess_group_32():
    # Load international football result from 1872 to 2018
    # For more detailed, refer to
    # https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017
    results = pd.read_csv('datasets/results.csv')

    # Mark winning team in international football from 1872 to 2018
    winner = []
    for i in range(len(results['home_team'])):
        if results['home_score'][i] > results['away_score'][i]:
            winner.append(results['home_team'][i])
        elif results['home_score'][i] < results['away_score'][i]:
            winner.append(results['away_team'][i])
        else:
            winner.append('Draw')
    results['winning_team'] = winner

    # Mark goal difference in international football from 1872 to 2018
    results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])

    # Extract teams who take part in FIFA 2018.
    df_teams_home = results[results['home_team'].isin(FIFA_2018_TEAMS)]
    df_teams_away = results[results['away_team'].isin(FIFA_2018_TEAMS)]
    df_teams = pd.concat((df_teams_home, df_teams_away))
    df_teams.drop_duplicates()
    df_teams = results.drop_duplicates()

    # Mark years in international football
    year = []
    for row in df_teams['date']:
        year.append(int(row[:4]))
    df_teams['match_year'] = year

    # Drop results before 1930s
    df_teams_1930 = df_teams[df_teams.match_year >= 1930]
    df_teams_1930 = df_teams_1930.drop(labels=[
        'date', 'home_score', 'away_score', 'tournament', 'city',
        'country', 'goal_difference', 'match_year'], axis=1)

    df_teams_1930 = df_teams_1930.reset_index(drop=True)
    df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.home_team, 'winning_team'] = 2
    df_teams_1930.loc[df_teams_1930.winning_team == 'Draw', 'winning_team'] = 1
    df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.away_team, 'winning_team'] = 0

    # Load the latest FIFA 2018 results
    # For more detailed, refer to
    # https://fixturedownload.com/results/fifa-world-cup-2018
    new_result_filename = 'fifa-world-cup-2018-RussianStandardTime.csv'
    new_result_path = os.path.join(os.getcwd(), new_result_filename)
    urlretrieve(
        'https://fixturedownload.com/download/fifa-world-cup-2018-RussianStandardTime.csv',
        new_result_path)
    new_results = pd.read_csv(new_result_path)
    new_results = new_results.dropna(axis=0)

    # Mark winning team in FIFA 2018
    new_winners = []
    for i in range(len(new_results['Home Team'])):
        scores = new_results['Result'][i].split(' - ')
        if int(scores[0]) > int(scores[1]):
            new_winners.append(new_results['Home Team'][i])
        elif int(scores[0]) < int(scores[1]):
            new_winners.append(new_results['Away Team'][i])
        else:
            new_winners.append('Draw')
    new_results['home_team'] = new_results['Home Team']
    new_results['away_team'] = new_results['Away Team']
    new_results['winning_team'] = new_winners

    new_results = new_results.reset_index(drop=True)
    new_results.loc[new_results.winning_team == new_results.home_team, 'winning_team'] = 2
    new_results.loc[new_results.winning_team == 'Draw', 'winning_team'] = 1
    new_results.loc[new_results.winning_team == new_results.away_team, 'winning_team'] = 0

    new_results = new_results.drop(labels=[
        'Round Number', 'Date', 'Location', 'Home Team', 'Away Team', 'Group', 'Result'], axis=1)

    new_results = pd.concat((df_teams_1930, new_results))

    columns = ['home_team', 'away_team']
    final = pd.get_dummies(new_results, prefix=columns, columns=columns)

    x = final.drop(['winning_team'], axis=1)
    y = final['winning_team']
    y = y.astype('int')
    return train_test_split(x, y, test_size=0.20, random_state=49)


def fit_logreg_on_group_32():
    X_train, X_test, y_train, y_test = preprocess_group_32()

    logreg = LogisticRegression(solver='newton-cg', max_iter=10000, C=100.0)
    logreg.fit(X_train, y_train)
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)

    return (X_train, X_test, y_train, y_test, logreg)


def fit_mlp_on_group_32():
    seed = os.environ.get('BATCH_TASK_INSTANCE_INDEX', 0)
    import random
    random.seed(seed)
    rand = random.randint(2000, 2000000)
    X_train, X_test, y_train, y_test = preprocess_group_32()
    clf = MLPClassifier(
        solver='adam', activation='logistic',
        hidden_layer_sizes=(
            random.randint(50, 100),
            random.randint(20, 50),
            random.randint(10, 20)),
        learning_rate_init=0.00001,
        alpha=100.0, max_iter=1000, random_state=49)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    return (X_train, X_test, y_train, y_test, clf)


def clean_and_predict(matches, ranking, final, model):
    positions = []
    for match in matches:
        positions.append(ranking.loc[ranking['Team'] == match[0], 'Position'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == match[1], 'Position'].iloc[0])

    pred_set = []
    i = 0
    j = 0

    while i < len(positions):
        team_pair = {}

        if positions[i] < positions[i + 1]:
            team_pair.update({'home_team': matches[j][0], 'away_team': matches[j][1]})
        else:
            team_pair.update({'home_team': matches[j][1], 'away_team': matches[j][0]})

        pred_set.append(team_pair)
        i += 2
        j += 1

    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set
    pred_set = pd.get_dummies(
        pred_set,
        prefix=['home_team', 'away_team'],
        columns=['home_team', 'away_team'])

    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    predictions = model.predict(pred_set)
    for i in range(len(pred_set)):
        streams = []
        streams.append(backup_pred_set.iloc[i, 1])  # team1
        streams.append(backup_pred_set.iloc[i, 0])  # team2
        if predictions[i] == 2:
            streams.append(backup_pred_set.iloc[i, 1])
        elif predictions[i] == 1:
            streams.append('Draw')
        elif predictions[i] == 0:
            streams.append(backup_pred_set.iloc[i, 0])
        streams.append('%.3f' % (model.predict_proba(pred_set)[i][2]))  # team1 winning
        streams.append('%.3f' % (model.predict_proba(pred_set)[i][1]))  # Draw
        streams.append('%.3f' % (model.predict_proba(pred_set)[i][0]))  # team2 winning
        print ','.join(streams)


def predict(team1, team2, X_train, y_train, ranking, model):
    clean_and_predict([(team1, team2)], ranking, X_train, model)


def main():
    teams = PARSER.parse_args().teams
    ranking = generate_team_ranking()
    X_train, X_test, y_train, y_test, model = fit_mlp_on_group_32()
    predict(teams[0], teams[1], X_train, y_train, ranking, model)


if __name__ == "__main__":
    main()
