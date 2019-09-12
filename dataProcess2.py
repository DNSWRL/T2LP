import os

data_type = 'wiki_data'
version = 'large'

triple2id = 'data/'+ data_type +'/'+ version+'/triple2id.txt'

f = open(triple2id)
line = f.readline()

# print(line.split('\t')[0])
i = 1

while line:
    if line == '\n':
        line
    line = f.readline()
    i += 1
