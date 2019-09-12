# import json
import os

data_type = 'yago'
version = 'large'

entity2id = data_type + '/entity2id.txt'
relation2id = data_type + '/relation2id.txt'
train_data = data_type + '/train.txt'
valid_data  = data_type + '/valid.txt'
test_data  =  data_type + '/test.txt'
triple2id  =   data_type + '/triple2id_cp.txt'

save_dir = 'wiki/'
if not os.path.exists(save_dir): os.makedirs(save_dir)
new_data = open(save_dir + '/valid_data.txt','w')

f = open(valid_data)
line = f.readline()

# print(line.split('\t')[0])
i = 1

while line:
    head = line.split('\t')[0]
    rel = line.split('\t')[1]
    tail = line.split('\t')[2]
    start_time = line.split('\t')[3]
    if len(start_time) < 5:
        # i += 1
        print(i)
        start_time = '-50\n'
        # end_time = line.split('\t')[4].split('-')[0]
    # if '####' in start_time and end_time.find('#') != -1:
    #     new_data.write(head + '\t' + rel + '\t' + tail + '\t' + end_time)
    # if '####' in start_time:
    #     new_data.write(head + '\t' + rel + '\t' + tail + '\t' + str(-50) + '\n')
    # elif start_time.find('#') != -1 or len(start_time) != 4:
    #     new_data.write(head + '\t' + rel + '\t' + tail + '\t' + str(-50) + '\n')
    # else:
    new_data.write(head + '\t' + rel + '\t' + tail + '\t' + start_time)
    line = f.readline()
    i += 1

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