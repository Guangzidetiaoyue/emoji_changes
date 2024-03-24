import enum
import numpy as np
import os
from statistics import mean
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
emoji_res = []
emoji_meaning_file = 'data_si/Face2010_meaning.txt'
V_L = ['bad','awful','sad','unhappy','annoyed','unsatisfied','melancholic','despaired','negative']
V_H = ['good','wonderful', 'great','happy','pleased','satisfied','contented','hopeful','positive']
A_L = ['boring','unexciting','dull','relaxed','calm','sluggish','sleepy','unaroused','bored','unarousal']
A_H = ['arousal','active','frenzy','interesting','exciting','fun','stimulated','excited','frenzied','jittery']
D_L = ['controlled','influenced','cared-for','awed','submissive','guided','weak']
D_H = ['influential','important','controlling','dominant','powerful','autonomous']
files = sorted(os.listdir('/data1/reddit_comments/month_word_embedding'))
emojis = []
with open(emoji_meaning_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')
        if len(emoji)!= 0:
            emojis.append(emoji)

words_vad_total = []
with open('data_si/NRC-VAD-Lexicon.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        words_vad_total.append(line.strip().split('\t')[0])

for file in files:
    if 'bin' in file:
        file_name = file.split('.')[0]
        y_m = file.split('.')[0].split('_')[1]
        model_path = os.path.join('/data1/reddit_comments/month_word_embedding',file)
        model = fasttext.load_model(model_path)
        #all vad words and emoji similarity
        words_length = len(model.words)
        e_w = {}
        #all emotional words and emoji similarity
        #all emoji projections on VAD
        emoji_matrix = []
        word_emb = []
        V_L_EMB = []
        V_H_EMB = []
        A_L_EMB = []
        A_H_EMB = []
        D_L_EMB = []
        D_H_EMB = []
        V = []
        A = []
        D = []
        emoji_p_v_a_d = []
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
        for item in words_vad_total:
            word = item
            try:
                word_emb = model.get_word_vector(word)
                emoji_v_vector = word_emb.dot(V_MEAN)/(np.linalg.norm(word_emb)*np.linalg.norm(V_MEAN))
                emoji_a_vector = word_emb.dot(A_MEAN)/(np.linalg.norm(word_emb)*np.linalg.norm(A_MEAN))
                emoji_d_vector = word_emb.dot(D_MEAN)/(np.linalg.norm(word_emb)*np.linalg.norm(D_MEAN))
                emoji_p_v_a_d.append([word,emoji_v_vector,emoji_a_vector,emoji_d_vector])
            except:
                print(word+' Not found!')
        with open(f'vad_w_p_res_0328/{file_name}_w_v_a_d_{words_length}.txt','w',encoding='utf-8') as f1:
            for i,c_vad in enumerate(emoji_p_v_a_d):
                f1.writelines('\t'.join([str(c)for c in c_vad])+'\n')

        print('all semantic projection done\n')
        print(file+' done')

# emoji_res_df = pd.DataFrame(emoji_res,columns=['emoji','emotion description','time','cos_sim'])
# emoji_res_df.to_csv('emoji_all_res.csv')