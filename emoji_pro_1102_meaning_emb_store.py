import enum
import numpy as np
import os
from statistics import mean
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

emoji_meaning_file = 'Face2010_meaning.txt'
files = sorted(os.listdir('/media/zoujj/ZOUJIAJUNU/My_Data/reddit_comments_1015/month_word_embedding'))
emojis = []
with open(emoji_meaning_file,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')
        if len(emoji)!= 0:
            emojis.append(emoji)

for file in files:
    if file == 'RC_2016-09.bin':
        break
    else:
        file_name = file.split('.')[0]
        y_m = file.split('.')[0].split('_')[1]
        model_path = os.path.join('/media/zoujj/ZOUJIAJUNU/My_Data/reddit_comments_1015/month_word_embedding',file)
        model = fasttext.load_model(model_path)

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

        with open(f'res_1103/{file_name}_e_w_emb.txt','w',encoding='utf-8') as f1:
            for i,c in enumerate(emoji_meaning):
                f1.writelines(' '.join(c)+'\n')

    print(file+' done')

# emoji_res_df = pd.DataFrame(emoji_res,columns=['emoji','emotion description','time','cos_sim'])
# emoji_res_df.to_csv('emoji_all_res.csv')