import pickle, tqdm
import glob


# temp=[]
temp={}
pkl_list = glob.glob('test_instances/*.pickle')
for path_pkl in tqdm.tqdm(pkl_list):
    with open(path_pkl, 'rb') as fr:
        data=pickle.load(fr)
    temp.update(data)
    # data['original_shape'] = (1080, 900)
    # data['img_shape'] = (1080, 900)
    # temp.append(data)

with open('annotations/trace1_fix_trace2_fix.pickle', 'wb') as fr:
    pickle.dump(temp, fr)

exit()

with open('annotations/pose_train.pkl', 'rb') as fr:
    data=pickle.load(fr)

with open('annotations/train_list_videos.txt', 'r') as fr:
    all_lines = fr.readlines()

labels={}
for aline in all_lines:
    video_name, label  = aline.split()
    video_name = video_name[:-4]
    labels[video_name]=int(label)

for pose_dict in data:
    video_name = pose_dict['frame_dir']
    pose_dict['label']=labels[video_name]

with open('annotations/pose_train_fine.pkl', 'wb') as fr:
    pickle.dump(data, fr)
