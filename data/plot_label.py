import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

def fine2coarse(x):
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6


def draw_plot(draw_data, save_path, status):
    plt.figure(figsize=(16, 12))
    sns.countplot(x='class_id', data=draw_data, saturation=0.75)
    counts = draw_data['class_id'].value_counts()
    counts_sort = counts.sort_index(ascending=True)
    plt.title('Distribution of veideos ' + status)
    for index, value in counts_sort.iteritems():  # 在Pandas版本0.24及以前使用
        plt.text(index, value, value, ha="center", va="bottom")
    plt.savefig(os.path.join(save_path, status + ".png"))

train_txt = "annotations/train_trace2part_valpart_aug.txt"
# aug_train_txt = "annotations/train_list_videos_aug.txt"
# val_txt = "annotations/val_list_videos.txt"

train_data = pd.read_csv(train_txt, sep=' ', header=None, names=['veideo_name', 'class_id'])
# aug_train_data = pd.read_csv(aug_train_txt, sep=' ', header=None, names=['veideo_name', 'class_id'])
# val_data = pd.read_csv(val_txt, sep=' ', header=None, names=['veideo_name', 'class_id'])

coarse_train_data = copy.deepcopy(train_data)
# coarse_aug_train_data = copy.deepcopy(aug_train_data)
# coarse_val_data = copy.deepcopy(val_data)

for i in range(52): 
    coarse_train_data['class_id'].replace(i, fine2coarse(i), inplace=True)
    # coarse_aug_train_data['class_id'].replace(i, fine2coarse(i), inplace=True)
    # coarse_val_data['class_id'].replace(i, fine2coarse(i), inplace=True)

save_path = "data_distribution"

draw_plot(train_data, save_path, "fine_train")
# draw_plot(aug_train_data, save_path, "fine_aug_train")
# draw_plot(val_data, save_path, "fine_val")

draw_plot(coarse_train_data, save_path, "coarse_train")
# draw_plot(coarse_aug_train_data, save_path, "coarse_aug_train")
# draw_plot(coarse_val_data, save_path, "coarse_val")