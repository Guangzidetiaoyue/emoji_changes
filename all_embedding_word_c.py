import os
path = 'vad_w_p_res_0328/'
files = list(sorted(os.listdir(path)))
words_c = []
for f in files:
    c = f.split('_')[-1].split('.')[0]
    words_c.append(c)
with open('data_si/words_c.txt', 'w',encoding='utf-8') as f:
    for item in words_c:
        f.write(item+'\n')
