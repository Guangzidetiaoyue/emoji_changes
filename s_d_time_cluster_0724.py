from math import sqrt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats.mstats import zscore
from scipy.stats import pearsonr, beta
import random
from tsmoothie.smoother import LowessSmoother
from scipy.signal import savgol_filter
import matplotlib.ticker as mtick

smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)

color_b = ['#fae7f3','#f5dcf4','#e0f3ed'] #,'#f5dcf4' '#f4eef4'
color_l = ['#e789c3','#cc4ec6','#66c2a4'] # '#cc4ec6' '#c6abc6'

s_d_meaning_file = 'data_si/observe_d_0723.txt'
social_dimen = {}

with open(s_d_meaning_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        s_d = line.strip()
        if len(s_d)!= 0:
            social_dimen[s_d] = i

social_dimen_list = [['white', 'black', 'asian', 'irish', 'hispanic','native American'],['men','he','his','women','she','her'],['old','aged','elderly','young','youngster','teenager'],['fat','obesity','puffy','thin','lean','skinny'],['rich','affluent','wealthy','poor','needy','indigent'],['health','wellness','fitness','illness','sickness','disease'],['republic','republican','democratic','democracy','populist','populism','hedonic','anarchism','anarchistic'],['environmental','environment-friendly','eco-friendly','polluted','contaminable','contaminate'],['scientific','science','technology','antiscientific','antiscience','antitechnology'],['hungry','hunger','starvation','satiety','satiation','repletion'],['vagrant','wandering','nomadic','homely','home','household'],['unemployed','jobless','unemployment','employment','hire','working']]

x_s_d_data = pd.read_csv('social_dimension_all_score_0723.csv').drop(columns=['Unnamed: 0'])
cls_s = ['Race','Gender','Age','Stature','Wealth ','Health','Ideology','Environment','Science','Food','Housing','Working'] #'Partisan',

year = [x for x in range(2010,2022)]
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
labels_vad = ['Valence','Arousal','Dominance']
fig, axes = plt.subplots(3,4,sharex='col',figsize=(16,6)) #sharey='row'

social_dimension_cluster = {}
for dimen_i, item in enumerate(['v-','a-','d-']) : #
# item = 'v-'
    color_dim_line = color_l[dimen_i]
    color_dim_fill = color_b[dimen_i]
    emoji_pro_dis_all = []
    for idx,social_cls in enumerate(social_dimen_list):
        emoji_pro_dis = []
        for j in year:
                for i in month:
                    emoji_pro_temp = []
                    for emo_temp in social_cls:
                        emoji_pro_temp.append(x_s_d_data[item+str(j)+'-'+i].iloc[social_dimen[emo_temp]])

                    emoji_pro_temp = np.cov(np.array(emoji_pro_temp)).tolist()
                    emoji_pro_dis.append(emoji_pro_temp*2)
        emoji_pro_dis_all.append(emoji_pro_dis)
        social_dimension_cluster[item+cls_s[idx]] = emoji_pro_dis

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
        
        axes[row,col].set_title(cls_s[i]+'('+str(len(social_dimen_list[i]))+')',fontsize=13)
        axes[row,col].legend(loc='upper center',fontsize=8,framealpha=0.5)

axes[0,0].set_ylabel('Intraclass distance',fontsize=15,loc='bottom')
axes[1,0].set_ylabel('Intraclass distance',fontsize=15)
axes[2,0].set_ylabel('Intraclass distance',fontsize=15)
# y_lines,labels = fig.axes[-1].get_legend_handles_labels()
# # fig.legend(y_lines,labels,loc='center left',framealpha=1)
# fig.legend(y_lines,labels,loc='center right',framealpha=1) #,ncol=1,borderaxespad=0,mode='expand' bbox_to_anchor=(1.01,0.5),
plt.subplots_adjust(left=None,bottom=None,top=None,right=None,wspace=0.3,hspace=0.2)

s_d_res_df = pd.DataFrame(social_dimension_cluster)
s_d_res_df.to_csv('social_dimension_cluster_score_0723.csv')
print('done')