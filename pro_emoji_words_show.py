import json
import os
import collections
import pandas as pd
filepath = 'emoji_vad_res_0328'
file_emojis = 'data_si/VAD_with_Face2010.txt'

TOP = 3
emojis = []
with open(file_emojis,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        emoji = line.strip().split(' ')[0]
        if len(emoji)!= 0:
            emojis.append(emoji)

emoji_words = collections.defaultdict(list)
files = sorted(os.listdir(filepath))
for file in files:
    time = file.split(':')[0]
    path = os.path.join(filepath, file)
    data = json.load(open(path))
    for top in range(0,TOP):
        emoji_top_word = []
        for emoji in emojis:
            emoji_data_word_top = data[emoji][top]
            emoji_top_word.append(emoji_data_word_top[0])
        emoji_words[time+'-'+str(top)] = emoji_top_word

emoji_word_df = pd.DataFrame(emoji_words)
emoji_word_df.to_csv('emoji_word_df.csv')
print('done')

