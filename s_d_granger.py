from math import sqrt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

def plot_mapping_matrix(cm, savename,classes_y,classes_x,emojis,emoji_png,social_dimen):
    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax1 = fig.add_subplot(111)
    np.set_printoptions(precision=2)
    title = 'The C-S-M of emojis on ' + ' over Time'
    ind_array_x = np.arange(classes_x.shape[1]) #emotion_class
    ind_array_y = np.arange(classes_y.shape[0])
    x, y = np.meshgrid(ind_array_x, ind_array_y)
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary) #cmap=plt.cm.binary
    plt.title(title)
    xlocations = np.array(range(classes_x.shape[1])) #emotion_class
    ylocations = np.array(range(classes_y.shape[0]))
    #plt.xticks(xlocations, ind_array_x, rotation=90) #,fontsize=7
    plt.yticks(ind_array_y,social_dimen,fontsize=5) #,fontsize=7
    plt.xticks([])
    #plt.ylabel('Actual label')
    #plt.xlabel('Predict label')
    # cax = plt.axes([0.65, 0.1, 0.04, 0.8])
    # plt.colorbar(cax=cax)
    
    plt.colorbar(fraction=0.015)
    # offset the tick
    x_tick_marks = np.array(range(classes_x.shape[1])) + 0.5
    y_tick_marks = np.array(range(classes_y.shape[0])) + 0.5
    plt.gca().set_xticks(x_tick_marks, minor=True)
    plt.gca().set_yticks(y_tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')

    for i,emoji in enumerate(emojis):
        emoj = plt.imread(f"emoji_image/{emoji_png[emoji]}")
        imagebox = OffsetImage(emoj, zoom=0.06)
        imagebox.image.axes = ax1
        ab = AnnotationBbox(imagebox, (0,0),
                    xybox=(0.75, -0.05), #(0.3, -0.33)
                    xycoords='data',
                    boxcoords='axes fraction',
                    box_alignment=(0, 0),
                    pad=0,
                    bboxprops=dict(facecolor='w'),frameon=False)
        ax1.add_artist(ab)
    plt.gcf().subplots_adjust(bottom=0.1) 
    plt.savefig(savename, format='svg',dpi=300)


class CCA:
    '''
    # 说明
    该类用于典型相关分析。
    # 参数
    x_dataset 自变量数据，以[样本1, 样本2, ..., 样本t] 给出。

    y_dataset 因变量数据，以[样本1, 样本2, ..., 样本t] 给出。

    x_dataset 和 y_dataset 的样本应该一一对应。(第i个自变量决定第i个因变量)
    '''
    def __init__(self, x_dataset, y_dataset):
    	# 需要对数据转置一下，才能跟上文对上
        self.x_dataset = np.array(x_dataset, dtype = 'float64').T
        self.y_dataset = np.array(y_dataset, dtype = 'float64').T

    '''
    结果以三元组(rho, alpha, beta)形式给出:

        - rho: 典型变量的相关系数
        - alpha: 自变量系数
        - beta: 因变量系数
    '''
    def fit(self):
        A = []
        for sample in self.x_dataset:
            A.append(list(sample))
        for sample in self.y_dataset:
            A.append(list(sample))
        
        # 构造上面提到的A矩阵
        A = np.array(A, dtype = 'float64')
        
        # 标准化: 减去每行均值再除以标准差 
        for i in range(A.shape[0]):
            avg = np.mean(A[i])
            std = np.std(A[i])
            A[i] = (A[i] - avg) / std

		# bias = True 即计算时不采用对方差的无偏修正（除以n-1,样本方差）
		# 这里只是为了跟ppt里的数据对上,实际可以取消这个可选参数
        Cov = np.cov(A, bias = True)
        n = self.x_dataset.shape[0]

        R_11 = np.matrix(Cov[:n, :n])
        R_12 = np.matrix(Cov[:n, n:])
        R_21 = np.matrix(Cov[n:, :n])
        R_22 = np.matrix(Cov[n:, n:])

        M = np.linalg.inv(R_11) * R_12 * np.linalg.inv(R_22) * R_21
        N = np.linalg.inv(R_22) * R_21 * np.linalg.inv(R_11) * R_12

        eig1, vector1 = np.linalg.eig(M)

        data = []

        for i in range(len(eig1)):
        	# 若为0（精度误差，改为"绝对值小于一个很小的值"）
            if abs(eig1[i]) < 1e-10:
                continue
            # 下面变量与上面步骤中的意义相同
            rho = np.round(sqrt(eig1[i]), decimals = 5)
            alpha = np.round(vector1[:, i], decimals = 5)
            k = 1 / (alpha.T * R_11 * alpha)
            alpha *= sqrt(k)
            beta = np.round(np.linalg.inv(R_22) * R_21 * alpha / rho, decimals = 5)

            # 三元组分别为相关系数, 自变量系数, 因变量系数
            data.append((rho, alpha, beta))

        data.sort(key = lambda x: x[0], reverse = True)

        return data

file_emotions = 'data_si/VAD_with_Face2010.txt'
emojis = []
with open(file_emotions,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')[0]
        if len(emoji)!= 0:
            emojis.append(emoji)

emoji_png = {}
for b in emojis:
    if len(b) == 1:
        emoji_png[b]='{:x}.png'.format(ord(b))
    if len(b) > 1:
        out = ""
        for c in b:
            out += '{:x}_'.format(ord(c))
        emoji_png[b]='{}.png'.format(out[:-1])

s_d_meaning_file = 'data_si/observe_d.txt'
social_dimen = []
with open(s_d_meaning_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        s_d = line.strip()
        if len(s_d)!= 0:
            social_dimen.append(s_d)

x_s_d_data = pd.read_csv('social_dimension_all_score.csv').drop(columns=['Unnamed: 0'])
y_emo_data = pd.read_csv('emoji_all_score.csv').drop(columns=['emoji','Unnamed: 0'])
x_data_temp = []
y_data_temp = []
year = [x for x in range(2010,2022)]
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
for item in ['v-']: #,'a-','d-'
    for j in year:
        for i in month:
            x_data_temp.append(x_s_d_data[item+str(j)+'-'+i].values)
            y_data_temp.append(y_emo_data[item+str(j)+'-'+i].values)


# CCA
# cca_method = CCA(x_data_temp,y_data_temp)
# results = cca_method.fit()
# sum_c = 0
# for data in results:
#     sum_c += data[0]

# c_explain = 0
# for i,data in enumerate(results):
#     c_explain += data[0] / sum_c
#     if c_explain > 0.9:
#         print(i)


# df = pd.DataFrame(np.random.randint(0, 100, size=(10, 2)), columns=['a', 'b'])
# res = grangercausalitytests(df[['a', 'b']], maxlag=2)

res_idx = []
x_data_temp_t = np.array(x_data_temp, dtype = 'float64').T
y_data_temp_t = np.array(y_data_temp, dtype = 'float64').T
x_d = x_data_temp_t.shape[0]
y_d = y_data_temp_t.shape[0]

res_html = []
res_data_granger = np.zeros([y_d,x_d])
Max_L = 2
threshold = 0.05
for j in range(y_d):
    for i in range(x_d):
        test_data = pd.DataFrame({'p':y_data_temp_t[j],'r':x_data_temp_t[i]})
        # test_data = pd.DataFrame({'p':x_data_temp_t[i],'r':y_data_temp_t[j]})
        res = grangercausalitytests(test_data,maxlag=Max_L)
        for log in range(1,Max_L+1):
            res_temp = res[log][0]
            p_ftest = res_temp['ssr_ftest'][1]
            p_chi2test = res_temp['ssr_chi2test'][1]
            p_lrtest = res_temp['lrtest'][1]
            p_param = res_temp['params_ftest'][1]
            if p_ftest < threshold and p_chi2test < threshold and p_lrtest < threshold and p_param < threshold:
                print(i,'\t',j)
                res_idx.append([j,i,log])
                if log == 1:
                    res_html.append([j,i,1])
                    res_data_granger[j,i] += 1
# new_cm = np.array(res_data_granger)
# new_cm_t = new_cm.T
# plot_mapping_matrix(new_cm_t,'figure_res_0330/granger.svg',new_cm_t,new_cm_t,emojis,emoji_png,social_dimen)

value = res_html #res_idx #res_html
granger_every_dimension_sum = res_data_granger.sum(axis=0)
granger_every_dimension_sum_str = [str(int(item))+'test' for item in granger_every_dimension_sum ]

y_label =[social_dimen[i]+' ('+ str(int(item))+')' for i,item in enumerate(granger_every_dimension_sum)]
c = (
    HeatMap(init_opts=opts.InitOpts(width='2000px',height='1000px'))
    .add_xaxis(emojis)
    .add_yaxis(
        "series1",
        y_label, # social_dimen
        value,
        label_opts=opts.LabelOpts(is_show=False, position="inside"),
        itemstyle_opts=opts.ItemStyleOpts(opacity=0.5)
    )
    # .add_yaxis(
    #     "series0",
    #    granger_every_dimension_sum_str,
    #    value,
    #    yaxis_index=1,
    #    label_opts=opts.LabelOpts(is_show=False, position="inside"),
    #    itemstyle_opts=opts.ItemStyleOpts(opacity=0)
    # )
    # .extend_axis(
    #     yaxis=opts.AxisOpts(
    #         # axislabel_opts=opts.LabelOpts(font_size=15,font_family='Arial',font_weight='bold'),
    #         position='right',
    #     )
    # )
    .set_series_opts()
    .set_global_opts(
        # title_opts=opts.TitleOpts(title="HeatMap-Label 显示"),
        # visualmap_opts=opts.VisualMapOpts(min_=0, max_=10, is_calculable=False, orient="horizontal", pos_left="center"),
        legend_opts=opts.LegendOpts(is_show=False),

        xaxis_opts=opts.AxisOpts(
        type_="category",
        splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.5,color='rgba(255, 255, 255, 0.5)')),
        axislabel_opts=opts.LabelOpts(font_size=20,font_family='Arial',font_weight='bold'),
        splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(width=2))
        ),

        yaxis_opts=opts.AxisOpts(
            type_="category",
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.5,color='rgba(255, 255, 255, 0.5)')),
            axislabel_opts=opts.LabelOpts(font_size=15,font_family='Arial',font_weight='bold'),
            splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(width=2))
            ),
    )
)
c.render(path='figure_res_0330/heatmap_with_granger_0.html')
make_snapshot(snapshot, c.render(),'figure_res_0330/heatmap_with_granger.html_0.pdf')
print('done')