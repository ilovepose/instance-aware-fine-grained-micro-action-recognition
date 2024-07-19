import random

random.seed(0)

def fine2coarse_txt(path_in, path_out):
    with open(path_in, 'r') as fr:
        data=fr.readlines()

    temp=[]
    for aline in data:
        video_name, label = aline.split()
        if (label=='8' or label=='9' or label=='10') and (random.random()<0.5):
            continue
        # if (label=='8' or label=='9' or label=='19' or label=='21' or label=='24' or label=='49') and (random.random()<0.5):
        #     continue
        # label_int = int(label)
        # if label_int==0 or 5<=label_int<=11 or label_int==17 or \
        #     label_int==19 or 21<=label_int<=24 or 29<=label_int<=31 or 48<=label_int<=49:
        #     continue
        # newlabel = fine2coarse(int(label))
        newline = '{} {}\n'.format(video_name, label)
        temp.append(newline)

    with open(path_out, 'w') as fr:
        fr.writelines(temp)


if __name__=="__main__":
    fine2coarse_txt('annotations/val_list_videos.txt', 'annotations/val_list_videos_part.txt')

