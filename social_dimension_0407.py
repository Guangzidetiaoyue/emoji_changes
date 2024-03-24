import enum
import numpy as np
import os
from statistics import mean
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

s_d_meaning_file = 'data_si/observe_d_0723.txt'
V_L = ['bad','awful','sad','unhappy','annoyed','unsatisfied','melancholic','despaired','negative']
V_H = ['good','wonderful', 'great','happy','pleased','satisfied','contented','hopeful','positive']
A_L = ['boring','unexciting','dull','relaxed','calm','sluggish','sleepy','unaroused','bored','unarousable']
A_H = ['arousal','active','frenzy','interesting','exciting','fun','stimulated','excited','frenzied','jittery']
D_L = ['influential','important','dominant','controlling','powerful','autonomous']
D_H = ['controlled','influenced','cared-for','awed','submissive','guided','weak']
files = sorted(os.listdir('/data1/reddit_comments/month_word_embedding'))
social_dimen = []
with open(s_d_meaning_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        s_d = line.strip()
        if len(s_d)!= 0:
            social_dimen.append(s_d)

s_d_scores = {}
for file in files:
    if 'bin' in file:
        file_name = file.split('.')[0]
        y_m = file.split('.')[0].split('_')[1]
        model_path = os.path.join('/data1/reddit_comments/month_word_embedding',file)
        model = fasttext.load_model(model_path)
        #all vad words and s_d similarity

        S_D_EMB = []

        V_L_EMB = []
        V_H_EMB = []
        A_L_EMB = []
        A_H_EMB = []
        D_L_EMB = []
        D_H_EMB = []
        V = []
        A = []
        D = []
        s_d_p_v = []
        s_d_p_a = []
        s_d_p_d = []

        for v_l,v_h in zip(V_L,V_H):
            V_L_EMB.append(model.get_word_vector(v_l))
            V_H_EMB.append(model.get_word_vector(v_h))
        for a_l,a_h in zip(A_L,A_H):
            A_L_EMB.append(model.get_word_vector(a_l))
            A_H_EMB.append(model.get_word_vector(a_h))
        for d_l,d_h in zip(D_L,D_H):
            D_L_EMB.append(model.get_word_vector(d_l))
            D_H_EMB.append(model.get_word_vector(d_h))
        for v_l_item in V_L_EMB:
            for v_h_item in V_H_EMB:
                V.append(v_h_item-v_l_item)
        for d_l_item in D_L_EMB:
            for d_h_item in D_H_EMB:
                D.append(d_h_item-d_l_item)
        for a_l_item in A_L_EMB:
            for a_h_item in A_H_EMB:
                A.append(a_h_item-a_l_item)
        V_MEAN = np.mean(np.array(V),axis=0)
        D_MEAN = np.mean(np.array(D),axis=0)
        A_MEAN = np.mean(np.array(A),axis=0)
        for item in social_dimen:
            s_d_w = item
            try:
                S_D_EMB.append(model.get_word_vector(s_d_w))
            except:
                print(s_d_w+'Not found!')

        for s_b_emb in S_D_EMB:
            s_d_v_vector = s_b_emb.dot(V_MEAN)/(np.linalg.norm(s_b_emb)*np.linalg.norm(V_MEAN))
            s_d_p_v.append(s_d_v_vector)

            s_d_a_vector = s_b_emb.dot(A_MEAN)/(np.linalg.norm(s_b_emb)*np.linalg.norm(A_MEAN))
            s_d_p_a.append(s_d_a_vector)

            s_d_d_vector = s_b_emb.dot(D_MEAN)/(np.linalg.norm(s_b_emb)*np.linalg.norm(D_MEAN))
            s_d_p_d.append(s_d_d_vector)

        s_d_scores['v-'+y_m] = s_d_p_v
        s_d_scores['d-'+y_m] = s_d_p_d
        s_d_scores['a-'+y_m] = s_d_p_a

        print(file+' done')

s_d_res_df = pd.DataFrame(s_d_scores)
s_d_res_df.to_csv('social_dimension_all_score_0723.csv')