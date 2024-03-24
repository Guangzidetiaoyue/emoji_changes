import skccm as ccm
import numpy as np
import pandas as pd
from scipy.stats import zscore
from skccm.utilities import train_test_split
import matplotlib.pyplot as plt
import pyEDM

lag = 1
embed = 2

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

    for soc_i in x_social_dimension: # emo_i y_emoji_cluster[:1]
        x_data_temp_i = x_data_temp_t[item+'Race'].values #soc_i
        y_data_temp_i = x_data_temp_t[item+'Ideology'].values#y_data_temp_t[item+'All'].values

        # my_data = pyEDM.CCM(dataFrame=x_data_temp_i)
        x,y = x_data_temp_i, y_data_temp_i
        x_e = ccm.Embed(x)
        y_e = ccm.Embed(y)

        x1 = x_e.embed_vectors_1d(lag,embed)
        y1 = y_e.embed_vectors_1d(lag,embed)

        x1tr, x1te, y1tr, y1te = train_test_split(x1,y1, percent=.78)

        CCM = ccm.CCM() #initiate the class

        #library lengths to test
        len_tr = len(x1tr)
        lib_lens = np.arange(8, len_tr, len_tr/len_tr, dtype='int')

        #test causation
        CCM.fit(x1tr,y1tr)
        x1p, x2p = CCM.predict(x1te, y1te,lib_lengths=lib_lens)

        sc1,sc2 = CCM.score()
        plt.plot(lib_lens,sc1)
        plt.plot(lib_lens,sc2)
        plt.show()