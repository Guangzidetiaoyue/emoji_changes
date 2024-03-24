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

smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)

color_l = ['#0ee800','#66c2a4','#a00311','#cc4ec6','#0007d5','#fb7f1f','#5f07dd','#f017f3','#7789f5','#521fff','#f30b6a','#0f5cf4','#fb8d61','#8d9fca','#a6d753','#e789c3','#c6abc6','#fbb000']

s_d_meaning_file = 'data_si/observe_d.txt'
social_dimen = []
with open(s_d_meaning_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        s_d = line.strip()
        if len(s_d)!= 0:
            social_dimen.append(s_d)

x_s_d_data = pd.read_csv('social_dimension_all_score.csv').drop(columns=['Unnamed: 0'])

x_data_temp = []
y_data_temp = []
year = [x for x in range(2010,2022)]
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
for item in ['v-']: #,'a-','d-'
    for j in year:
        for i in month:
            x_data_temp.append(x_s_d_data[item+str(j)+'-'+i].values)


res_idx = []
data_res_np = np.array(x_data_temp, dtype = 'float64')
x_d = data_res_np.shape[0]

time = list(range(0,144))

x_labels = ['2010-01','2012-01','2014-01','2016-01','2018-01','2020-01']
fig, axes = plt.subplots(5,7,sharex='col',figsize=(24,8),sharey='row') #sharey='row'
for i in range(0,len(social_dimen)):
    c_i = random.randint(0,17)
    coef_df = pd.DataFrame()
    row = i // 7
    col = i % 7
    data_dimen_x = np.array(time)
    data_dimen_y = data_res_np[0:,i]
    data_dimen_y = zscore(np.array(data_dimen_y))

    pear_r = pearsonr(data_dimen_x,data_dimen_y)
    X = sm.add_constant(data_dimen_x) #words_fast_vad_mean_np_res
    etoh_ploy_2 = sm.OLS(data_dimen_y ,X).fit()
    # print(summary_table(etoh_ploy_2,alpha=0.05))
    coef_df_temp = pd.DataFrame({"params": etoh_ploy_2.params,  # 回归系数
                    "std err": etoh_ploy_2.bse,    # 回归系数标准差
                    "t": np.round(etoh_ploy_2.tvalues,3),   # 回归系数T值
                    "p-values": np.round(etoh_ploy_2.pvalues,3), # 回归系数P值
                    "pearson_r":pear_r[0],
                    "pearson_p":pear_r[1]
                        })
    coef_df_temp[['coef_0.025','coef_0.975']] = etoh_ploy_2.conf_int()
    coef_df = coef_df.append(coef_df_temp,ignore_index=False)

    smoother.smooth(data_dimen_y)
    low, up = smoother.get_intervals('confidence_interval') # prediction_interval
    data_dimen_y_sm = smoother.smooth_data.squeeze(0)
    axes[row,col].tick_params(bottom=True,top=False,left=True,right=False)
    axes[row,col].set_xticks([x for x in range(0,132,24)])
    axes[row,col].set_ylim(-2,2)
    axes[row,col].set_xticklabels(x_labels,rotation=36,fontsize=9)
    axes[row,col].spines['top'].set_linewidth(1.2)
    axes[row,col].spines['bottom'].set_linewidth(1.2)
    axes[row,col].spines['left'].set_linewidth(1.2)
    axes[row,col].spines['right'].set_linewidth(1.2)
    axes[row,col].set_title(social_dimen[i],fontsize=13)
    
    # sns.regplot(x=data_dimen_x,y=data_dimen_y,ci=95,order=1,scatter_kws={"s":0.5,"alpha":0.8,"color":color_l[c_i]},line_kws= {"linestyle":'-','linewidth':1},ax=axes[row,col],color=color_l[c_i])
    axes[row,col].plot(time,data_dimen_y_sm,c=color_l[c_i])
    axes[row,col].scatter(time,data_dimen_y_sm,c=color_l[c_i],s=1,alpha=0.3)

plt.delaxes(axes[4][6])

plt.subplots_adjust(left=None,bottom=None,top=None,right=None,wspace=0.2,hspace=0.3)
axes[0,0].set_ylabel('Valence',fontsize=15)
axes[1,0].set_ylabel('Valence',fontsize=15)
axes[2,0].set_ylabel('Valence',fontsize=15)
axes[3,0].set_ylabel('Valence',fontsize=15)
axes[4,0].set_ylabel('Valence',fontsize=15)
print('done')
print('done')