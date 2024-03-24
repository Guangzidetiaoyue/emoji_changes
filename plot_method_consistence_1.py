from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.stats import pearsonr, beta
import seaborn as sns
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import zscore
words_vad_total = []
vad_words_all = []
# color_b = ['#fae7f3','#f4eef4','#f5dcf4']
# color_l = ['#e789c3','#c6abc6','#cc4ec6']
# color_b = ['#fee8df','#e0f3ed','#fff7d5','#edf7dd','#fae7f3','#f4eef4','#f5dcf4']
color_l = ['#fb8d61','#66c2a4','#a6d753','#e789c3','#c6abc6','#cc4ec6']
with open('data_si/NRC-VAD-Lexicon.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        vad_words_all.append(line.strip().split('\t')[0])
        words_vad_total.append([float(x) for x in line.strip().split('\t')[1:]])
words_vad_total_np = np.array(words_vad_total)
words_vad_total_np_zs = zscore(words_vad_total_np)

words_fast_vad_mean_np = 0
year = '2018'
files = sorted(os.listdir('vad_w_p_res_0328'))
for f in files:
    file_path = os.path.join('vad_w_p_res_0328',f)
    time = f.split('_')[1]
    if year in time:
        with open(file_path,'r',encoding='utf-8') as f:
            words_fast_vad_total = []
            lines = f.readlines()
            for line in lines:
                words_fast_vad_total.append([float(x) for x in line.strip().split('\t')[1:]])
            words_fast_vad_total_np = np.array(words_fast_vad_total)
            words_fast_vad_mean_np += words_fast_vad_total_np

words_fast_vad_mean_np_res = words_fast_vad_mean_np / 10 
words_fast_vad_mean_np_res_zs = zscore(words_fast_vad_mean_np_res)

scaler = MinMaxScaler( )
scaler.fit(words_fast_vad_mean_np_res)
scaler.data_max_
words_fast_vad_mean_np_res_normalized=scaler.transform(words_fast_vad_mean_np_res)


fig, axes = plt.subplots(1,3,sharex='col',figsize=(12,4)) #sharey='row'

for i in range(0,3):
    coef_df = pd.DataFrame()
    data_dimen_x_ori = words_fast_vad_mean_np_res_normalized[:,i]
    data_dimen_y_ori = words_vad_total_np[:,i]

    # idx_0_1_np = np.linspace(0,1,len(data_dimen_x_ori))[1:-1]
    # idx_0_1_np = np.arange(len(data_dimen_x_ori))[1:-1]

    dist_idx = beta.rvs(0.9,0.9,scale=len(data_dimen_x_ori),size=int(len(data_dimen_x_ori)/3)).astype("int")
    data_dimen_x_ori_sort_idx = np.argsort(data_dimen_x_ori)

    data_dimen_x = []
    data_dimen_y = []
    for item in dist_idx:
        idx_real = data_dimen_x_ori_sort_idx[item]
        data_dimen_x.append(data_dimen_x_ori[idx_real])
        data_dimen_y.append(data_dimen_y_ori[idx_real])
    # idx_np_up = np.where(data_dimen_x_ori > 0.7)[0]
    # idx_np_lower = np.where(data_dimen_x_ori < 0.3)[0]
    # idx_np_all = np.concatenate((idx_np_up,idx_np_lower))
    # data_dimen_x = []
    # data_dimen_y = []
    # for item in idx_np_all:
    #     data_dimen_x.append(data_dimen_x_ori[item])
    #     data_dimen_y.append(data_dimen_y_ori[item])
    
    # scaler = MinMaxScaler()
    # scaler.fit(np.array(data_dimen_x).reshape(-1,1))
    # scaler.data_max_
    # data_dimen_x_n=scaler.transform(np.array(data_dimen_x).reshape(-1,1))
    
    data_dimen_x = zscore(np.array(data_dimen_x))
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

    axes[i].tick_params(bottom=True,top=False,left=True,right=False)
    # axes[i].set_xticks([])
    # axes[i].set_ylim(-0.3,0.25)
    axes[i].spines['top'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)
    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['right'].set_linewidth(1.2)
    sns.regplot(x=data_dimen_x,y=data_dimen_y,ci=95,order=1,scatter_kws={"s":0.5,"alpha":0.3,"color":color_l[i]},line_kws= {"linestyle":'-','linewidth':1},ax=axes[i],color=color_l[i]) #words_fast_vad_mean_np_res ,x_estimator=np.mean color_b[i]
    axes[i].annotate('r = '+str('%.3f'%pear_r[0]),xy=(0.7,0.15),fontsize=10,weight='bold')
    # axes[i].annotate('p = '+str('%.3f'%pear_r[0]),xy=(0.8,0.2),fontsize=10)
    coef_df.to_csv('figure_res_0330/'+str(i)+'vad_coef.csv',encoding='utf-8')
    # plt.scatter(words_fast_vad_total_np[i:],words_vad_total_np[i:],s=0.1)