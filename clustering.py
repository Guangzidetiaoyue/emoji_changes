import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import os
import numpy as np
import matplotlib.ticker as mt
from pyecharts.charts import ThemeRiver
import pyecharts.options as opts
import seaborn as sns
import statsmodels.api as sm
from scipy.stats.mstats import zscore

color_b = ['#fee8df','#e0f3ed','#fff7d5','#edf7dd','#fae7f3','#f4eef4','#f5dcf4']
color_l = ['#fb8d61','#66c2a4','#8d9fca','#a6d753','#e789c3','#c6abc6','#cc4ec6']
file_emotions = 'data_si/VAD_with_Face2010.txt'
emojis = []
with open(file_emotions,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')[0]
        if len(emoji)!= 0:
            emojis.append(emoji)
emoji_temp = emojis.copy()
for i in range(1,12):
    emoji_temp.extend(emojis)
data = pd.read_csv('emoji_all_score.csv')
data_cluster_y = {}
data_cluster_y['emoji'] = emoji_temp
y = [x for x in range(2010,2022)]
for j in y:
    for item in ['v-','a-','d-']:
        temp = list(data[item+str(j)+'-01'])
        for i in range(2,13):
            if i > 9:
                temp.extend(list(data[item+str(j)+'-'+str(i)]))
            else:
                temp.extend(list(data[item+str(j)+'-0'+str(i)]))
        data_cluster_y[item+str(j)] = temp

emoji_cluster_y = pd.DataFrame(data_cluster_y)
emoji_cluster_y.to_csv('emoji_cluster_y.csv')

emojis_file = 'emojis_group'
emojis_affection = []
emojis_concerned = []
emojis_negative = []
emojis_neutral = []
emojis_sleepy = []
emojis_smiling = []
emojis_tongue = []
emojis_unwell = []
files = os.listdir(emojis_file)
for file in files:
    emoji_temp = []
    with open(emojis_file+'/'+file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            emoji = line.strip().split(' ')[0]
            if len(emoji)!= 0:
                emoji_temp.append(emoji)
    emo_f = file.split('.')[0].split('_')[1]
    if emo_f == 'affection':
        emojis_affection = emoji_temp
    if emo_f == 'concerned':
        emojis_concerned = emoji_temp
    if emo_f == 'negative':
        emojis_negative = emoji_temp
    if emo_f == 'neutral':
        emojis_neutral = emoji_temp
    if emo_f == 'sleepy':
        emojis_sleepy = emoji_temp
    if emo_f == 'smiling':
        emojis_smiling = emoji_temp
    if emo_f == 'tongue':
        emojis_tongue = emoji_temp
    if emo_f == 'unwell':
        emojis_unwell = emoji_temp
emoji_class = ['affection','concerned','negative','neutral',
                'sleepy','smiling','tongue','unwell']

emoji_df = {}
emoji_cluster_idx = []
emoji_cluster_name = []
for emoji in emojis:
    if emoji in emojis_affection:
        emoji_cluster_name.append('affection')
        emoji_cluster_idx.append(emoji_class.index('affection'))
    elif emoji in emojis_concerned:
        emoji_cluster_name.append('concerned')
        emoji_cluster_idx.append(emoji_class.index('concerned'))
    elif emoji in emojis_negative:
        emoji_cluster_name.append('negative')
        emoji_cluster_idx.append(emoji_class.index('negative'))
    elif emoji in emojis_neutral:
        emoji_cluster_name.append('neutral')
        emoji_cluster_idx.append(emoji_class.index('neutral'))
    elif emoji in emojis_sleepy:
        emoji_cluster_name.append('sleepy')
        emoji_cluster_idx.append(emoji_class.index('sleepy'))
    elif emoji in emojis_smiling:
        emoji_cluster_name.append('smiling')
        emoji_cluster_idx.append(emoji_class.index('smiling'))
    elif emoji in emojis_tongue:
        emoji_cluster_name.append('tongue')
        emoji_cluster_idx.append(emoji_class.index('tongue'))
    elif emoji in emojis_unwell:
        emoji_cluster_name.append('unwell')
        emoji_cluster_idx.append(emoji_class.index('unwell'))
    else:
        print(emoji)
emoji_df['emoji'] = emojis
emoji_df['cluster_id'] = emoji_cluster_idx
emoji_df['cluster_name'] = emoji_cluster_name
emoji_cluster_data_df = pd.DataFrame(emoji_df)
emoji_cluster_data_df.to_csv('emoji_cluster_data_df.csv')


