from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.stats import pearsonr, beta, spearmanr
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

file_emojis = 'data_si/VAD_with_Face2010.txt'
emojis = []
with open(file_emojis,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')[0]
        if len(emoji)!= 0:
            emojis.append(emoji)
emotion_words = []
# emotion_word_file = 'emoji_adj_res_0328/RC_2010-01_e_w_emb.txt'
emotion_word_file = 'data_si/emotion_concepts_mine.txt'
with open(emotion_word_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        item = line.strip().split(' ')[0]
        if item not in emojis:
            emotion_words.append(item)
emotion_words_set_my_emoji = list(sorted(set(emotion_words)))
# with open(emotion_word_file,'w',encoding='utf-8') as f:
#     for item in emotion_words_set_my_emoji:
#         f.write(item+'\n')
emotional_word_set_hed = (pd.read_csv('data_si/Hedonometer.csv')['Word'].values).tolist()

emotion_word_adj_mine = []
with open('data_si/mine_total_emotion_list.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emotion_word_adj_mine.append(line.strip())

index_candidates = []
vad_w_candidates = []
for w in emotion_words_set_my_emoji:
    if w in vad_words_all:
        index = vad_words_all.index(w)
        index_candidates.append(index)
        vad_w_candidates.append(words_vad_total[index])
vad_w_candidates_np = np.array(vad_w_candidates)
vad_w_candidates_np_zs = zscore(vad_w_candidates_np)


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
            for idx in index_candidates:
                words_fast_vad_total.append([float(x) for x in lines[idx].strip().split('\t')[1:]])
            # print(words_fast_vad_total[50][2])
            words_fast_vad_total_np = np.array(words_fast_vad_total)
            words_fast_vad_mean_np += words_fast_vad_total_np

words_fast_vad_mean_np_res = words_fast_vad_mean_np / 12
words_fast_vad_mean_np_res_zs = zscore(words_fast_vad_mean_np_res)

scaler = MinMaxScaler( )
scaler.fit(words_fast_vad_mean_np_res)
scaler.data_max_
words_fast_vad_mean_np_res_normalized=scaler.transform(words_fast_vad_mean_np_res)


fig, axes = plt.subplots(1,3,sharex='col',figsize=(13,5)) #sharey='row'

dimension = ['Valence', 'Arousal', 'Dominance']
for i in range(0,3):
    coef_df = pd.DataFrame()
    data_dimen_x_ori = words_fast_vad_mean_np_res_normalized[:,i]
    data_dimen_y_ori = vad_w_candidates_np[:,i]

    # idx_0_1_np = np.linspace(0,1,len(data_dimen_x_ori))[1:-1]
    # # idx_0_1_np = np.arange(len(data_dimen_x_ori))[1:-1]
    # # dist = beta(0.99,0.99)
    # idx_np_all_prob = beta.pdf(idx_0_1_np,a=0.99,b=0.99,scale=0.1)
    # plt.plot(idx_0_1_np,idx_np_all_prob)
    
    # idx_np = np.arange(len(data_dimen_x_ori))[1:-1]
    # idx_np_all = np.random.choice(idx_np,p=idx_np_all_prob)

    # plt.plot(idx_np,idx_np_all)
    # plt.show()
    idx_np_up = np.where(data_dimen_x_ori > -0.1)[0]
    idx_np_lower = np.where(data_dimen_x_ori < 0.0)[0]
    idx_np_all = np.concatenate((idx_np_up,idx_np_lower))

    data_dimen_x = []
    data_dimen_y = []
    for item in idx_np_all:
        data_dimen_x.append(data_dimen_x_ori[item])
        data_dimen_y.append(data_dimen_y_ori[item])
    data_dimen_x = np.array(data_dimen_x)
    data_dimen_y = np.array(data_dimen_y)

    data_dimen_x = data_dimen_x_ori
    data_dimen_y = data_dimen_y_ori
    pear_r = pearsonr(data_dimen_x,data_dimen_y)
    spearman_r = pearsonr(data_dimen_x,data_dimen_y)
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
    axes[i].set_ylim(0.0,1.0)
    axes[i].spines['top'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)
    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['right'].set_linewidth(1.2)
    axes[i].set_title(dimension[i],fontsize=13)
    axes[i].set_xlabel('Semantic projection score',fontsize=13)
    axes[0].set_ylabel('Manual rating score',fontsize=13)
    sns.regplot(x=data_dimen_x,y=data_dimen_y,ci=95,order=1,scatter_kws={"s":2,"alpha":0.5,"color":color_l[i]},line_kws= {"linestyle":'-','linewidth':1},ax=axes[i],color=color_l[i]) #words_fast_vad_mean_np_res ,x_estimator=np.mean color_b[i]
    axes[i].annotate('r = '+str('%.3f'%pear_r[0]),xy=(0.7,0.20),fontsize=10,weight='bold')
    if pear_r[1] < 0.001:
        axes[i].annotate('p < 0.001',xy=(0.695,0.12),fontsize=10,weight='bold')
    coef_df.to_csv('figure_res_0330/'+str(i)+'vad_coef.csv',encoding='utf-8')
    # plt.scatter(words_fast_vad_total_np[i:],words_vad_total_np[i:],s=0.1)