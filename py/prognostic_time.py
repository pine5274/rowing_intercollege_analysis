import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize
from statistics import median
import functions

def read_cev():
    usecols = ['year', 'boat_type', 'section_code', '2000m', 'qualify']
    df = pd.read_csv(
        './../src/csv/intercollege_results.csv',
        usecols=usecols,
        dtype={
            'year': int,
            'boat_type': str,
            'section_code': str,
            '2000m': float,
            'qualify': str
        }
    )

    return df[df['2000m'] != 0]

def calc_trends(df, boat_type):
    # 外れ値の除去
    q = df['2000m'].quantile(0.954)
    df = df[(df['2000m'] < q)]

    x = df.loc[:, 'year']
    y = df.loc[:, '2000m']      
    y = y.map(functions.speed)
    #近似式の係数
    res1=np.polyfit(x, y, 1)
    res2=np.polyfit(x, y, 2)
    res3=np.polyfit(x, y, 3)
    #近似式の計算
    y1 = np.poly1d(res1)(x) #1次
    y2 = np.poly1d(res2)(x) #2次
    y3 = np.poly1d(res3)(x) #3次

    # 2021年の推定値
    PT_1 = np.poly1d(res1)(2021)
    PT_2 = np.poly1d(res2)(2021)
    PT_3 = np.poly1d(res3)(2021)
    #グラフ表示
    plt.scatter(x, y, label='time')
    plt.plot(x, y1, label='1d')
    plt.plot(x, y2, label='2d')
    plt.plot(x, y3, label='3d')
    plt.xticks(np.arange(2000, 2022, 2))
    plt.title(boat_type + " trends")
    plt.xlabel('year', fontsize=12)  # x軸ラベル
    plt.ylabel('speed[m/s]', fontsize=12)  # y軸ラベル
    plt.grid()
    plt.legend()
    plt.figure()
    plt.savefig('./../dst/prognostic_time/' + boat_type + '.jpg')

    return  median([PT_1, PT_2, PT_3])

df = read_cev()
df = df[df['section_code'].str.contains("決勝")]
winner_df = df.groupby(['boat_type', 'year'], as_index=False)['2000m'].min()
boat_types = (winner_df['boat_type'].unique())
prognostic_time = {}
for boat_type in boat_types:
    if (boat_type == 'w4x') | (boat_type == 'w4+'):
        continue
    prognostic_time[boat_type] = calc_trends(winner_df[winner_df['boat_type'] == boat_type], boat_type)

pt_df = pd.DataFrame.from_dict(prognostic_time, orient="index", columns=["PT"])
pt_df.to_csv('./../dst/prognostic_time/PT_time.csv')