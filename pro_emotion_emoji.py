import os
import numpy as np
import pandas as pd
import collections
file_emojis = 'data_si/VAD_with_Face2010.txt'
file_nrc_emotion_lexicon = 'data_si/NRC-Emotion-Lexicon.txt'
file_emojis_adj = 'emoji_adj_res_0328'
emotion_class = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", 'trust']

emojis = []
with open(file_emojis,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')[0]
        if len(emoji)!= 0:
            emojis.append(emoji)

emotion_words = []
emotion_word_file = 'emoji_adj_res_0328/RC_2010-01_e_w_emb.txt'
with open(emotion_word_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        item = line.split(' ')[0]
        if item not in emojis:
            emotion_words.append(item)
emotion_words_set = list(set(emotion_words))

nrc_emotion_lexicons_emoji = collections.defaultdict(list)
with open(file_nrc_emotion_lexicon,'r',encoding='utf') as f1:
    lines = f1.readlines()
    for line in lines:
        temp = line.strip().split('\t')
        if temp[0] in emotion_words_set and temp[1] in emotion_class and int(temp[2])==1:
            nrc_emotion_lexicons_emoji[temp[1]].append(temp[0])

all_emojis_adj_scores = {}

emotions_col = []
emojis_col = []
for emotion in emotion_class:
    for emoji in emojis:
        emotions_col.append(emotion)
        emojis_col.append(emoji)

all_emojis_adj_scores['emotion'] = emotions_col
all_emojis_adj_scores['emoji'] = emojis_col

all_file_emojis_adj = sorted(os.listdir(file_emojis_adj))
for file in all_file_emojis_adj:
    time = file.split('_')[1]
    emoji_emotion_col = []
    emojis_adj_dic = collections.defaultdict(list)
    file_path = os.path.join(file_emojis_adj,file)
    with open(file_path,'r',encoding='utf-8') as fi:
        lines = fi.readlines()
        for line in lines:
            temp = line.strip().split(' ')
            emojis_adj_dic[temp[0]] = [float(item) for item in temp[1:]]

    for emotion in emotion_class:
        w_emb = []
        emojis_emb_score = []
        for w in nrc_emotion_lexicons_emoji[emotion]:
            w_emb.append(emojis_adj_dic[w])
        emotion_emb_mean = np.mean(np.array(w_emb),axis=0)

        for e in emojis:
            emojis_emb = np.array(emojis_adj_dic[e])
            cos_emoji_emotion = (emojis_emb.dot(emotion_emb_mean)/(np.linalg.norm(emojis_emb)*np.linalg.norm(emotion_emb_mean)))
            emojis_emb_score.append(cos_emoji_emotion)
        emoji_emotion_col.extend(emojis_emb_score)

    all_emojis_adj_scores[time] = emoji_emotion_col

emoji_emotion_df = pd.DataFrame(all_emojis_adj_scores)
emoji_emotion_df.to_csv('emoji_emotion_score.csv')
print('done')

