from matplotlib import pyplot as plt
import os
import matplotlib.ticker as mtick
from pyecharts.charts import ThemeRiver
import pyecharts.options as opts
from pyecharts.charts import Sankey
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats.mstats import zscore
from plot_func import many_densities_plot
from tsmoothie.smoother import LowessSmoother
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.signal import savgol_filter
from scipy.spatial import distance
from scipy.spatial.distance import pdist
import collections
from matplotlib.patches import PathPatch
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
from pyecharts.globals import CurrentConfig

# CurrentConfig.ONLINE_HOST = "http://127.0.0.0:8000/assets/"#如果是内网必须要更改为自己服务器的ip

data_scores = pd.read_csv('emoji_emotion_score_only_1_time.csv')
data_scores['emoji_e'] = [x+'*' for x in data_scores['emoji']]
nodes = []
index = ['emoji'] + [str(i) for i in range(2010,2022,2)] + ["2021"] + ["emoji_e"]
for idx in index:
    if idx == 'emoji' or idx == 'emoji_e':
        values = data_scores[idx].unique()
    else:
        values = data_scores[idx+'-12'].unique()
    for value in values:
        # dic = {}
        # dic['name'] = value
        # nodes.append(dic)
        nodes.append(str(value))

type_list = ["emoji"] + [str(i)+'-12' for i in range(2010,2022,2)]+ ["2021-12"]+ ["emoji_e"]#, "2016-01",'2021-12' emoji
from_to_list = []
for idx in range(len(type_list)-1):
    from_type = type_list[idx]
    to_type = type_list[idx+1]
    print(from_type, to_type)

    df_agg = data_scores.groupby([from_type, to_type]).size().reset_index()
    df_agg.columns = ["from", "to", "value"]

    for _, (from_key, to_key, value) in df_agg.iterrows():
        from_to_list.append([str(from_key), str(to_key), value])

# df_agg = data_scores.groupby(["total", "xyz_campaign_id"]).size().reset_index()
# df_agg.columns = ["from", "to", "value"]

# 1. 转换节点列表为桑基图形式
pyecharts_nodes = [{"name": node} for node in nodes]

# 2. 转换跳转列表

pyecharts_links = [
            {"source": source, "target": target, "value": value}
            for source, target, value in from_to_list
        ]

sankey = (
    Sankey(init_opts=opts.InitOpts(width='2000px',height='900px')) #,renderer='svg'
    .add(
        "",
        pyecharts_nodes,
        pyecharts_links,
        node_width = 20,
        node_gap = 5,
        node_align="right",
        # orient='horizontal', #vertical horizontal
        linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
        label_opts=opts.LabelOpts(position="left"),
        # levels=[opts.SankeyLevelsOpts(depth=0,itemstyle_opts=opts.LabelOpts(position="left"))]
    )
    # .set_global_opts(title_opts=opts.TitleOpts(title="Test"))
    .set_series_opts(label_opts=opts.LabelOpts(font_size=15,font_family='Arial',font_weight='bold',position='right'))
)

sankey.render(path='figure_res_0330/emotion_emoji_time.html')
make_snapshot(snapshot, sankey.render(),'figure_res_0330/emotion_emoji_time.svg') #pdf

# links = []
# for i in data_scores.values:
#     dic = {}
#     dic['source'] = i[1]
    
# node_emoji = data_scores['emoji']
# node_emotion = data_scores['2010-01']


