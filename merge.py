#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def main():
    result_path = os.environ.get('OUTPUT_PATH')
    if not result_path or not os.path.exists(result_path):
        raise Exception('prediction result path not found')

    result = []
    # Load multi results
    for item in os.listdir(result_path):
        file_path = os.path.join(result_path, item)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                result.append([s.strip() for s in f.read().split(',')])

    final_result = result[0]
    final_result[3] = float(final_result[3])
    final_result[4] = float(final_result[4])
    final_result[5] = float(final_result[5])

    for item in result[1:]:
        final_result[3] += float(item[3])  # team1 win
        final_result[4] += float(item[4])  # draw
        final_result[5] += float(item[5])  # team2 win

    final_result.append(sum(final_result[3:6]))

    if final_result[3] > final_result[5]:
        final_result[2] = final_result[0]
    elif final_result[3] < final_result[5]:
        final_result[2] = final_result[1]
    else:
        final_result[2] = 'Draw'

    print 'The winner of %s and %s is %s' % (final_result[0], final_result[1], final_result[2])
    print 'Probability of %s winning is %.3f' % (final_result[0], final_result[3] / final_result[6])
    print 'Probability of draw is %.3f' % (final_result[4] / final_result[6])
    print 'Probability of %s winning is %.3f' % (final_result[1], final_result[5] / final_result[6])


if __name__ == "__main__":
    main()
