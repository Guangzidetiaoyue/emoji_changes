from matplotlib import pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import os
import numpy as np
import matplotlib.ticker as mtick
from pyecharts.charts import ThemeRiver
import pyecharts.options as opts
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats.mstats import zscore
from tsmoothie.smoother import LowessSmoother
import pyecharts.options as opts
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.signal import savgol_filter
from scipy.spatial import distance
from matplotlib.patches import PathPatch

smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)

color_bb = ['#fee8df','#e0f3ed','#fff7d5','#edf7dd','#fae7f3','#f4eef4','#f5dcf4'] ##'#8d9fca'
color_ll = ['#fb8d61','#66c2a4','#ffd92e','#a6d753','#e789c3','#c6abc6','#cc4ec6'] ##'#8d9fca'
color_b = ['#fae7f3','#f5dcf4','#e0f3ed'] #,'#f5dcf4' '#f4eef4'
color_l = ['#e789c3','#cc4ec6','#66c2a4'] # '#cc4ec6' '#c6abc6'

file_emotions = 'data_si/VAD_with_Face2010.txt'
file_path = 'emoji_adj_res_0328/'
emoji_meaning = 'data_si/Face2010_meaning.txt'
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

# cls_s = ['Affection','Concerned','Negative','Neutral','Sleepy','Smiling','Tongue','Unwell']
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
emojis_all = []
for emoji in emojis:
    emojis_all.append((emojis.index(emoji),emoji))
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
# emojis_classification_res = [emojis_affection_res,emojis_concerned_res,emojis_negative_res,emojis_neutral_res,emojis_sleepy_res,emojis_smiling_res,emojis_tongue_res,emojis_unwell_res] #,emojis_unwell_res  #[emojis_smiling_res]
emojis_classification_res = [emojis_concerned_res,emojis_smiling_res,emojis_negative_res,emojis_neutral_res,emojis_tongue_res,emojis_sleepy_res,emojis_affection_res,emojis_unwell_res,emojis_all]

cls_s = ['Concerned','Smiling','Negative','Neutral','Tongue','Sleepy','Affection','Unwell','All']
fig, axes = plt.subplots(2,4,sharex='col',figsize=(16,6)) #sharey='row'
# plt.xlabel('Time',fontsize=15)
# plt.ylabel('Intraclass distance',fontsize=15)
fig1, axes1 = plt.subplots(1,1,sharex='col',figsize=(8,5)) #sharey='row'
emoji_cluster_score = {}

year = [x for x in range(2010,2022)]
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
labels_vad = ['Valence','Arousal','Dominance']
for dimen_i, item in enumerate(['v-','a-','d-']) : #
# item = 'v-'

    color_dim_line = color_l[dimen_i]
    color_dim_fill = color_b[dimen_i]
    emoji_pro_dis_all = []
    for idx,emo_cls in enumerate(emojis_classification_res):
        emoji_pro_dis = []
        for j in year:
                for i in month:
                    emoji_pro_temp = []
                    for emo_temp in emo_cls:
                        emoji_pro_temp.append(data_scores[item+str(j)+'-'+i].iloc[emo_temp[0]])

                    emoji_pro_temp = np.cov(np.array(emoji_pro_temp)).tolist()
                    emoji_pro_dis.append(emoji_pro_temp*2)
        emoji_pro_dis_all.append(emoji_pro_dis)
        emoji_cluster_score[item+cls_s[idx]] = emoji_pro_dis

    emojis_dis_y_np = np.array(emoji_pro_dis_all)
    emojis_dis_y_np_zs = zscore(emojis_dis_y_np)
    # emojis_dis_y_np_01 = preprocessing.MinMaxScaler().fit_transform(emojis_dis_y_np)
    smoother.smooth(emojis_dis_y_np)
    low, up = smoother.get_intervals('confidence_interval') # prediction_interval
    emojis_dis_y_np_sm = smoother.smooth_data
    
    y = savgol_filter(emojis_dis_y_np,21,1,mode='nearest') #emojis_dis_y_np_zs
    
    time = [t for t in range(0,emojis_dis_y_np_zs.shape[1])]
    for i in range(0,y.shape[0]):
        # plt.scatter(time,emojis_dis_y_np_zs[i],s=5,alpha=0.3,linewidths=2)
        # plt.plot(time,y[i])
        if i < y.shape[0]-1:
            row = i // 4
            col = i % 4
            axes[row,col].plot(time,emojis_dis_y_np_sm[i],c=color_dim_line,linewidth=1.2,label=labels_vad[dimen_i]) #y[i]
            # axes[row,col].scatter(time,emojis_dis_y_np[i],s=1,alpha=0.2,linewidths=1)
            axes[row,col].plot(time,low[i],c=color_dim_line,linewidth=0.2)
            axes[row,col].plot(time,up[i],c=color_dim_line,linewidth=0.2)
            axes[row,col].fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.5,facecolor=color_dim_fill)

            axes[row,col].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
            axes[row,col].tick_params(bottom=True,top=False,left=True,right=False)
            axes[row,col].set_xticks(list(range(0,145,36)))
            axes[row,col].set_xticklabels([str(x)+'-01' for x in range(2010,2021,3)]+['2021-12'],rotation=36)
            # axes[row,col].set_ylim(-0.3,0.25)
            axes[row,col].spines['top'].set_linewidth(1.2)
            axes[row,col].spines['bottom'].set_linewidth(1.2)
            axes[row,col].spines['left'].set_linewidth(1.2)
            axes[row,col].spines['right'].set_linewidth(1.2)
            
            axes[row,col].set_title(cls_s[i]+'('+str(len(emojis_classification_res[i]))+')',fontsize=13)
            axes[row,col].legend(loc='upper center',fontsize=8,framealpha=0.5)
        else:
            row = 0
            col = 0
            axes1.plot(time,emojis_dis_y_np_sm[i],c=color_dim_line,linewidth=1.2,label=labels_vad[dimen_i]) #y[i]
            # axes[row,col].scatter(time,emojis_dis_y_np[i],s=1,alpha=0.2,linewidths=1)
            axes1.plot(time,low[i],c=color_dim_line,linewidth=0.2)
            axes1.plot(time,up[i],c=color_dim_line,linewidth=0.2)
            axes1.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.5,facecolor=color_dim_fill)

            axes1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
            axes1.tick_params(bottom=True,top=False,left=True,right=False)
            axes1.set_xticks(list(range(0,145,36)))
            axes1.set_xticklabels([str(x)+'-01' for x in range(2010,2021,3)]+['2021-12'],rotation=36)
            # axes[row,col].set_ylim(-0.3,0.25)
            axes1.spines['top'].set_linewidth(1.2)
            axes1.spines['bottom'].set_linewidth(1.2)
            axes1.spines['left'].set_linewidth(1.2)
            axes1.spines['right'].set_linewidth(1.2)
            
            axes1.set_title(cls_s[i]+'('+str(len(emojis_classification_res[i]))+')',fontsize=13)
            axes1.legend(loc='upper center',fontsize=12,framealpha=0.5)
axes[0,0].set_ylabel('Intraclass distance',fontsize=15,loc='bottom')
axes[1,0].set_ylabel('Intraclass distance',fontsize=15)
axes1.set_ylabel('Intraclass distance',fontsize=15)
# y_lines,labels = fig.axes[-1].get_legend_handles_labels()
# # fig.legend(y_lines,labels,loc='center left',framealpha=1)
# fig.legend(y_lines,labels,loc='center right',framealpha=1) #,ncol=1,borderaxespad=0,mode='expand' bbox_to_anchor=(1.01,0.5),
plt.subplots_adjust(left=None,bottom=None,top=None,right=None,wspace=0.3,hspace=0.2)
# plt.xlabel('Time',fontsize=15)
# plt.ylabel('Intraclass distance',fontsize=15)
e_d_res_df = pd.DataFrame(emoji_cluster_score)
e_d_res_df.to_csv('emoji_cluster_score_0725.csv')
print('done')


