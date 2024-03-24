import enum
import numpy as np
import os
from statistics import mean
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

emoji_meaning_file = 'data_si/Face2010_meaning.txt'
V_L = ['bad','awful','sad','unhappy','annoyed','unsatisfied','melancholic','despaired','negative']
V_H = ['good','wonderful', 'great','happy','pleased','satisfied','contented','hopeful','positive']
A_L = ['boring','unexciting','dull','relaxed','calm','sluggish','sleepy','unaroused','bored','unarousal']
A_H = ['arousal','active','frenzy','interesting','exciting','fun','stimulated','excited','frenzied','jittery']
D_L = ['influential','important','dominant','controlling','powerful','autonomous']
D_H = ['controlled','influenced','cared-for','awed','submissive','guided','weak']
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

emoji_scores = {}
for file in files:
    if 'bin' in file:
        file_name = file.split('.')[0]
        y_m = file.split('.')[0].split('_')[1]
        model_path = os.path.join('/data1/reddit_comments/month_word_embedding',file)
        model = fasttext.load_model(model_path)
        #all vad words and emoji similarity
        words_length = len(model.words)
        e_w = {}
        for emo_t in emojis:
            emo = emo_t[0]
            e_w_tmp = {}
            emo_emb = model.get_word_vector(emo)
            for w in words_vad_total:
                try:
                    w_emb = model.get_word_vector(w)
                except:
                    print(w+' Not found!')
                e_w_tmp[w] = (emo_emb.dot(w_emb)/(np.linalg.norm(emo_emb)*np.linalg.norm(w_emb)))
            e_w_tmp_sort = sorted(e_w_tmp.items(),key=lambda d:d[1],reverse=True)
            e_w_tmp_sort_s = [(item[0],str(item[1]))for item in e_w_tmp_sort]
            e_w[emo] = e_w_tmp_sort_s
        with open('emoji_vad_res_0328/'+y_m+':'+str(words_length)+':emoji_w_res.json', 'w',encoding='utf-8') as f1:
            json.dump(e_w,f1)
        print('all vad words done\n')
        #all emotional words and emoji similarity
        emoji_meaning = []
        for item in emojis:
            words_emb = []
            words_emo_si = []
            emoji_temp = item[0]
            words = item[1:]
            try:
                emoji_embedding = model.get_word_vector(emoji_temp)
            except:
                print(emoji+'Not found!')
            emoji_meaning.append([emoji_temp]+list(map(lambda x:str(x),emoji_embedding)))
            # one method
            for word in words:
                # words_emb.append(model.get_word_vector(word))
                word_emb = model.get_word_vector(word)
                emoji_meaning.append([word]+list(map(lambda x:str(x),word_emb)))
        with open(f'res_emoji_adj/{file_name}_e_w_emb.txt','w',encoding='utf-8') as f1:
            for i,c in enumerate(emoji_meaning):
                f1.writelines(' '.join(c)+'\n')
        print('all emotional words done\n')
        #all emoji projections on VAD
        emoji_matrix = []
        EMOJI_EMB = []
        V_L_EMB = []
        V_H_EMB = []
        A_L_EMB = []
        A_H_EMB = []
        D_L_EMB = []
        D_H_EMB = []
        V = []
        A = []
        D = []
        emoji_p_v = []
        emoji_p_a = []
        emoji_p_d = []

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
        for item in emojis:
            emoji = item[0]
            try:
                EMOJI_EMB.append(model.get_word_vector(emoji))
            except:
                print(emoji+'Not found!')
        for emoji_emb in EMOJI_EMB:
            # emoji_v_vector = cosine_similarity(emoji_emb.reshape(1,-1),V_MEAN.reshape(1,-1))[0][0]
            emoji_v_vector = emoji_emb.dot(V_MEAN)/(np.linalg.norm(emoji_emb)*np.linalg.norm(V_MEAN))
            emoji_p_v.append(emoji_v_vector)

            # emoji_a_vector = cosine_similarity(emoji_emb.reshape(1,-1),A_MEAN.reshape(1,-1))[0][0]
            emoji_a_vector = emoji_emb.dot(A_MEAN)/(np.linalg.norm(emoji_emb)*np.linalg.norm(A_MEAN))
            emoji_p_a.append(emoji_a_vector)

            # emoji_d_vector = cosine_similarity(emoji_emb.reshape(1,-1),D_MEAN.reshape(1,-1))[0][0]
            emoji_d_vector = emoji_emb.dot(D_MEAN)/(np.linalg.norm(emoji_emb)*np.linalg.norm(D_MEAN))
            emoji_p_d.append(emoji_d_vector)

        emoji_scores['v-'+y_m] = emoji_p_v
        emoji_scores['d-'+y_m] = emoji_p_d
        emoji_scores['a-'+y_m] = emoji_p_a


        # with open(f'res_1025/{file_name}_e_v_matrix.txt','w',encoding='utf-8') as f1:
        #     for i,c in enumerate(emoji_p_v):
        #         f1.writelines(str(c)+'\n')
        # with open(f'res_1025/{file_name}_e_a_matrix.txt','w',encoding='utf-8') as f2:
        #     for i,c in enumerate(emoji_p_a):
        #         f2.writelines(str(c)+'\n')
        # with open(f'res_1025/{file_name}_e_d_matrix.txt','w',encoding='utf-8') as f3:
        #     for i,c in enumerate(emoji_p_d):
        #         f3.writelines(str(c)+'\n')
        # print('all semantic projection done\n')
        print(file+' done')

emoji_res_df = pd.DataFrame(emoji_scores)
emoji_res_df.to_csv('emoji_all_score.csv')