from matplotlib import pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import os
import numpy as np
import matplotlib.ticker as mt
from pyecharts.charts import ThemeRiver
import pyecharts.options as opts
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats.mstats import zscore
from tsmoothie.smoother import LowessSmoother

smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)
color_b = ['#fee8df','#e0f3ed','#fff7d5','#edf7dd','#fae7f3','#f4eef4','#f5dcf4']
color_l = ['#fb8d61','#66c2a4','#8d9fca','#a6d753','#e789c3','#c6abc6','#cc4ec6']
file_emotions = 'data_si/VAD_with_Face2010.txt'

res_vad = 'figure_res_0330/'
emojis_file = 'emojis_group'
data_scores = pd.read_csv('emoji_all_score.csv')

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
emojis = []
with open(file_emotions,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')[0]
        if len(emoji)!= 0:
            emojis.append(emoji)

emojis_affection_res = []
emojis_concerned_res = []
emojis_negative_res = []
emojis_neutral_res = []
emojis_sleepy_res = []
emojis_smiling_res = []
emojis_tongue_res = []
emojis_unwell_res = []
for emoji in emojis:
    if emoji in emojis_affection:
        emojis_affection_res.append((emojis.index(emoji),emoji))
    if emoji in emojis_concerned:
        emojis_concerned_res.append((emojis.index(emoji),emoji))
    if emoji in emojis_negative:
        emojis_negative_res.append((emojis.index(emoji),emoji))
    if emoji in emojis_neutral:
        emojis_neutral_res.append((emojis.index(emoji),emoji))
    if emoji in emojis_sleepy:
        emojis_sleepy_res.append((emojis.index(emoji),emoji))
    if emoji in emojis_smiling:
        emojis_smiling_res.append((emojis.index(emoji),emoji))
    if emoji in emojis_tongue:
        emojis_tongue_res.append((emojis.index(emoji),emoji))
    if emoji in emojis_unwell:
        emojis_unwell_res.append((emojis.index(emoji),emoji))
emojis_classification_res = [emojis_affection_res,emojis_concerned_res,emojis_negative_res,emojis_neutral_res,emojis_sleepy_res,emojis_smiling_res,emojis_tongue_res,emojis_unwell_res] #

emoji_png = {}
for b in emojis:
    if len(b) == 1:
        emoji_png[b]='{:x}.png'.format(ord(b))
    if len(b) > 1:
        out = ""
        for c in b:
            out += '{:x}_'.format(ord(c))
        emoji_png[b]='{}.png'.format(out[:-1])


every_emoji_vad = []
time_y_m = []
year = [x for x in range(2010,2022)]
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
for dimen_i, item in enumerate(['v-','a-','d-']) :
    emoji_pro_every_dim = []
    for j in year:
            for i in month:
                emoji_pro_every_dim.append(data_scores[item+str(j)+'-'+i].values)
    every_emoji_vad.append(emoji_pro_every_dim)

v_array = np.array(every_emoji_vad[0])
a_array = np.array(every_emoji_vad[1])
d_array = np.array(every_emoji_vad[2])

i_num = a_array.shape[1]
t_num = a_array.shape[0]

# fig = plt.figure(dpi=300,figsize=(10,8))
# ax = fig.add_subplot()
coef_df = pd.DataFrame()
fig, axes = plt.subplots(10,6,sharex='col',sharey='row',figsize=(8,12)) #sharey='row'
# plt.tick_params(bottom=False,top=False,left=False,right=False)
# plt.tight_layout()

# fig = plt.figure()
data_fr_all = []
I = 0
Num = 0
flag = 0
for i,emo_cls in enumerate(emojis_classification_res):
    if flag > 9:
        break
    else:
        I = i
        data = []
        emo_every_cls = []
        data_fr = []
        T = [j for j in range(0,t_num)]
        for j,item in enumerate(emo_cls):
            data_fr_est = []
            emo_every_cls.append(item[1])

            emo_v_all = v_array[:,item[0]]
            emo_a_all = a_array[:,item[0]]
            emo_d_all = d_array[:,item[0]]

            smoother.smooth(emo_v_all)
            low, up = smoother.get_intervals('confidence_interval') # prediction_interval
            emo_v_all_sm = smoother.smooth_data.squeeze(0)

            smoother.smooth(emo_a_all)
            low, up = smoother.get_intervals('confidence_interval') # prediction_interval
            emo_a_all_sm = smoother.smooth_data.squeeze(0)

            smoother.smooth(emo_d_all)
            low, up = smoother.get_intervals('confidence_interval') # prediction_interval
            emo_d_all_sm = smoother.smooth_data.squeeze(0)

            axes[flag+j//6][(j)%6].tick_params(bottom=True,top=False,left=True,right=False)
            axes[flag+j//6][(j)%6].set_xticks([])
            # axes[flag+j//6][(j)%6].set_xticks(list(range(0,145,36)))
            # axes[flag+j//6][(j)%6].set_xticklabels([str(x)+'-01' for x in range(2010,2021,3)]+['2021-12'],rotation=36,fontsize=5)
            axes[flag+j//6][(j)%6].set_ylim(-0.3,0.3)
            axes[flag+j//6][(j)%6].spines['top'].set_linewidth(1.2)
            axes[flag+j//6][(j)%6].spines['bottom'].set_linewidth(1.2)
            axes[flag+j//6][(j)%6].spines['left'].set_linewidth(1.2)
            axes[flag+j//6][(j)%6].spines['right'].set_linewidth(1.2)
            # axes = plt.subplot(8,i+1,j+1)

            axes[flag+j//6][(j)%6].plot(T,emo_d_all_sm,c=color_l[I])
            axes[flag+j//6][(j)%6].scatter(T,emo_d_all,c=color_l[I],s=1,alpha=0.3)

            # sns.regplot(x=list(data_fr_est_df['Time']),y=list(data_fr_est_df['Value']),order=1,scatter_kws={"s": 5},line_kws= {"linestyle":'-','linewidth':3},ax=axes[flag+j//6][(j)%6],color=color_l[I])

            emoj = plt.imread(f"emoji_image/{emoji_png[item[1]]}")
            imagebox = OffsetImage(emoj, zoom=0.2)
            imagebox.image.axes = axes[flag+j//6][(j)%6]
            ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0.75, 0.02), #(0.3, -0.33)
                        xycoords='data',
                        boxcoords='axes fraction',
                        box_alignment=(0, 0),
                        pad=0,
                        bboxprops=dict(facecolor='w'),frameon=False)
            axes[flag+j//6][(j)%6].add_artist(ab)

        for j_j in range((j)%6+1,6):
            plt.delaxes(axes[flag+j//6][j_j])
            # sns.despine(top=False,right=False,bottom=False,left=False)
            # plt.xticks(T[0:len(T):12],time_y_m[0:len(time_y_m):12])
            # fig.set_size_inches(10,6)
        flag += (j//6+1)
        Num += len(emo_cls)
print(('Done'))
plt.xlabel('Time',loc='right',fontsize=12)
axes[0,0].set_ylabel('Projection score',fontsize=6)
axes[1,0].set_ylabel('Projection score',fontsize=6)
axes[2,0].set_ylabel('Projection score',fontsize=6)
axes[3,0].set_ylabel('Projection score',fontsize=6)
axes[4,0].set_ylabel('Projection score',fontsize=6)
axes[5,0].set_ylabel('Projection score',fontsize=6)
axes[6,0].set_ylabel('Projection score',fontsize=6)
axes[7,0].set_ylabel('Projection score',fontsize=6)
axes[8,0].set_ylabel('Projection score',fontsize=6)
axes[9,0].set_ylabel('Projection score',fontsize=6)

plt.subplots_adjust(left=None,bottom=None,top=None,right=None,wspace=0.2,hspace=0.2)
# plt.suptitle('v',x=0.5,y=0.92,fontsize=15)
plt.savefig('figure_res_0330/res_d_every_emoji.svg',dpi=300)
