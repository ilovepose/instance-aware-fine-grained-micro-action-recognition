import os


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

def fine2coarse_txt(path_in, path_out):
    with open(path_in, 'r') as fr:
        data=fr.readlines()

    temp=[]
    for aline in data:
        video_name, label = aline.split()
        newlabel = fine2coarse(int(label))
        newline = '{} {}\n'.format(video_name, newlabel)
        temp.append(newline)

    with open(path_out, 'w') as fr:
        fr.writelines(temp)


if __name__=="__main__":
    fine2coarse_txt('annotations/train_list_videos_aug2.txt', 'annotations/train_list_videos_aug2_coarse.txt')

