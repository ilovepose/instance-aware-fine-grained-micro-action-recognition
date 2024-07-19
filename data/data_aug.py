import os
import pandas as pd
import copy
import math

train_txt = "/home/wangchen/projects/datasets/Microaction-52/annotations/train_trace2part_valpart.txt"
train_data = pd.read_csv(train_txt, sep=' ', header=None, names=['veideo_name', 'class_id'])
counts = train_data['class_id'].value_counts()
print(counts)
counts_df = counts.to_frame()
counts_df.columns = ["counts"]
counts_df_select = counts_df[counts_df.counts < 100]
aug_repeat_dict = {}
for index, value in counts_df_select.iterrows():
    # select_id.append(index)
    # print(type(index), int(value))
    aug_repeat_dict[index] = int(math.log2(100 // int(value))) + 1
# exit()
# print(aug_repeat_dict)
# exit()
# train_data_copy = copy.deepcopy(train_data)


for id, repeat_times in aug_repeat_dict.items():
    for i in range(repeat_times):
        train_data = train_data.append(train_data[train_data.class_id == id], ignore_index=True)
# counts_last = train_data['class_id'].value_counts()
# counts_sort = counts_last.sort_index(ascending=True)
# print(counts_sort)
train_data.to_csv("/home/wangchen/projects/datasets/Microaction-52/annotations/train_trace2part_valpart_aug.txt", sep=' ', index=False, header=False)