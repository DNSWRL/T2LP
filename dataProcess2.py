# import json
import os

data_type = 'wiki_t2lp'
version = ''

entity2id = data_type + version + '/entity2id.txt'
relation2id = data_type + version + '/relation2id.txt'
train_data = data_type + version + '/train.txt'
valid_data  = data_type + version + '/valid.txt'
test_data  =  data_type + version + '/test.txt'
triple2id  =   data_type + version + '/triple'

save_dir = 'wiki_t2lp/'
if not os.path.exists(save_dir): os.makedirs(save_dir)
new_data = open(save_dir + '/triple2id.txt','w')

f = open(triple2id)
line = f.readline()

# print(line.split('\t')[0])
i = 1

while line:
    # print(i)
    if line == '\n':
        i += 1
        line = f.readline()
        continue
    else:
        new_data.write(line)
    line = f.readline()

# i = 0
#
# while line:
#     entity = line.split('\t')[0]
#     id = line.split('\t')[1]
#     start_time = line.split('\t')[2].split('-')[0]
#     end_time = line.split('\t')[3].split('-')[0]
#     if '####' in start_time and '####' not in end_time:
#         start_time = end_time
#     if '####' in start_time and '####' in end_time:
#         print(i)
#     new_data.write(entity + '\t' + id + '\t' + start_time + '\n')
#     line = f.readline()
#     i += 1
print(i)
f.close()
new_data.close()