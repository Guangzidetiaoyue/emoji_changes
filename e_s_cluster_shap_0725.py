from math import sqrt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from snapshot_selenium import snapshot
import shap
import xgboost as xgb
from scipy.stats import zscore
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from shap.plots.colors  import _colorconv
def lch2rgb(x):
    return _colorconv.lab2rgb(_colorconv.lch2lab([[x]]))[0][0]
blue_lch = [54., 70., 4.6588]
l_mid = 40.
red_lch = [54., 90., 0.35470565 + 2* np.pi]
gray_lch = [55., 0., 0.]
blue_rgb = lch2rgb(blue_lch)
red_rgb = lch2rgb(red_lch)
gray_rgb = lch2rgb(gray_lch)
white_rgb = np.array([1.,1.,1.])


# blue_rgb = np.array([0.56078,0.73725,0.56078])
# red_rgb = np.array([0.39216,0.59431,0.92941])
# white_rgb = np.array([1.,1.,1.,])
colors = []
for alpha in np.linspace(1, 0, 100):
    c = blue_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)
for alpha in np.linspace(0, 1, 100):
    c = red_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)


shap.initjs()  # notebook环境下，加载用于可视化的JS代码
x_s_d_data = pd.read_csv('social_dimension_cluster_score_0723.csv').drop(columns=['Unnamed: 0'])
y_emo_data = pd.read_csv('emoji_cluster_score_0725.csv').drop(columns=['Unnamed: 0'])

x_social_dimension = ['Race','Gender','Age','Stature','Wealth ','Health','Ideology','Environment','Science','Food','Housing','Working']
y_emoji_cluster = ['All','Concerned','Smiling','Negative','Neutral','Tongue','Sleepy','Affection','Unwell']
for item in ['v-','a-','d-']: #
    if item == 'v-':
        x_data_temp = x_s_d_data.iloc[:,:len(x_social_dimension)]
        y_data_temp = y_emo_data.iloc[:,:len(y_emoji_cluster)]
    if item == 'a-':
        x_data_temp = x_s_d_data.iloc[:,len(x_social_dimension):2*len(x_social_dimension)]
        y_data_temp = y_emo_data.iloc[:,len(y_emoji_cluster):2*len(y_emoji_cluster)]
    if item == 'd-':
        x_data_temp = x_s_d_data.iloc[:,2*len(x_social_dimension):]
        y_data_temp = y_emo_data.iloc[:,2*len(y_emoji_cluster):]

    x_data_temp_t = pd.DataFrame(zscore(x_data_temp,axis=1))
    y_data_temp_t = pd.DataFrame(zscore(y_data_temp,axis=1))

    for emo_i in y_emoji_cluster:
        y_data_temp_i = y_data_temp_t[item+emo_i].values

        X,y = x_data_temp_t, y_data_temp_i
        model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
        model.fit(X, y)
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X) 
        shap_values_obj = explainer(X)
        # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],plot_cmap='BrBG',matplotlib=True)
        # shap.plots.heatmap(shap_values_obj)

        feature_order, values_T_array = shap.plots.heatmap(shap_values_obj,max_display= shap_values.shape[1],show=False)

        # values_T_array = np.expand_dims(values_T_array,axis=1)
        # clist = ["#F7A065", "#fffea9", "#b7e3a0", "#78c49d", "#44a298","#237f8b","#1D717C"]
        # clist = ["#44a298","#fffea9","#F7A065","#E1874A","#D7722E","#A44301"]
        newcmp = LinearSegmentedColormap.from_list('chaos', colors)

        plt.figure(figsize=(16, 4))
        plt.rcParams.update({'font.family': 'Times New Roman'})
        vmin, vmax = np.nanpercentile(shap_values_obj.values.flatten(), [1, 99])
        ax_heatmap = sns.heatmap(values_T_array, cmap=newcmp, linewidths=0, linecolor='gray',vmin=min(vmin,-vmax),vmax=max(-vmin,vmax))

        # 设置色条的位置
        cbar = ax_heatmap.collections[0].colorbar
        cbar.ax.set_position([0.755, 0.11, 0.02, 0.77])  # 根据需要调整数值以确定位置
        if abs(vmin) > abs(vmax):
            vmax = abs(vmin)
        # 设置色条的刻度标签
        # cbar.set_ticks([min(vmin,-vmax), max(-vmin,vmax)])  # 根据数据调整数值
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['Low', 'High'])  # 根据需要调整标签
        cbar.set_label("SHAP value", size=11, labelpad=-15)

        # 设置纵坐标 y轴 刻度标签["Greek labels"] + 
        yticklabels = [x_social_dimension[i] for i in feature_order]
        ax_heatmap.set_yticklabels(yticklabels, rotation=0, size=11)
        # 获取第一个标签的文本对象
        label_text = ax_heatmap.get_yticklabels()[0:3]
        # 设置第一个标签的字体样式为粗体
        for lt in label_text:
            lt.set_weight('bold')
        # 刷新图像以应用更改
        plt.draw()

        # 设置横坐标 x轴 的刻度位置
        ax_heatmap.set_xticks(np.arange(0, values_T_array.shape[1]+1, 24))
        ax_heatmap.set_xticklabels(['01/2010', '01/2012', '01/2014', '01/2016', '01/2018', '01/2020','12/2021'], rotation=0, size=11)

        # 绘制横坐标x轴的黑色坐标轴线
        ax_heatmap.axhline(y=values_T_array.shape[0], color='black', linewidth=2)
        ax_heatmap.axhline(y=0, color='#363636', linewidth=0.18)

        ax_heatmap.set_xlabel('Year', size=13)

        # 加格子外框
        def highlight_cell(x, y, ax=None, **kwargs):
            rect = plt.Rectangle((x, y), 1, 1, fill=False, **kwargs)
            ax = ax or plt.gca()
            ax.add_patch(rect)
            return rect
        for i in range(1, values_T_array.shape[1]):
            for j in range(0, values_T_array.shape[0]+1):
                highlight_cell(i, j, color="#363636", linewidth=0.05)

        # 刻度点
        ax_heatmap.tick_params(axis="x", bottom=True, length=2)
        ax_heatmap.tick_params(axis="y", left=True, length=2)

        # plt.savefig('./heatmap_plot_300dpi.jpg', bbox_inches='tight', dpi=300)
        # plt.savefig('./fig3_300dpi.png', bbox_inches='tight', dpi=300)
        # plt.savefig('./fig3_300dpi.tiff', bbox_inches='tight', dpi=300)
        plt.savefig('figure_res_0330/emoji_social_cluster_0727/{}_emoji_social_{}0727.svg'.format(emo_i,item), bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()
print('done')