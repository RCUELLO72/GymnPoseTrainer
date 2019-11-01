"""
Utilities to query the dataset
"""
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from itertools import combinations

def load_dataset(pkl_file_name):
    # Load a saved dataset into memory
    with open(pkl_file_name,'rb') as f:
        my_dataset = pickle.load(f)
    return(my_dataset)
    
def get_joint_pairs():
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]] 
    return(joint_pairs)    
    
def draw_skeleton(W,H,coords):
    img = np.zeros((W,H,3), np.uint8)
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]] 
    colormap_index = np.linspace(0, 1, len(joint_pairs))
    pts = coords
    for cm_ind, jp in zip(colormap_index, joint_pairs):
        cm_color = tuple([int(x * 255) for x in plt.cm.cool(cm_ind)[:3]]) 
        pt1 = (int(pts[jp, 0][0]), int(pts[jp, 1][0]))
        pt2 = (int(pts[jp, 0][1]), int(pts[jp, 1][1]))
        cv2.line(img, pt1, pt2, cm_color, 3)
    return(img)
    
    
    
def get_skeleton_frame(gymn_dataset,clip_ID,frame_number):
    coords = gymn_dataset["Skeletons"][clip_ID][frame_number]
    img = draw_skeleton(800,600,coords)
    return(img)
    
def show_skeleton_movie(gymn_dataset,clip_ID):
    #not working
    sk_list = gymn_dataset["Skeletons"][clip_ID]
    fig = plt.figure()
    ims = []
    for sk_frame in sk_list:
        if type(sk_frame)==np.ndarray:
            skeleton_img = draw_skeleton(800,600,sk_frame)
            cv2.imshow('La',skeleton_img)
            im = plt.imshow(skeleton_img,animated=True)
            ims.append([im])
    ani = animation.ArtistAnimation(fig,ims,interval=500,blit=True,repeat_delay=1000)            
    plt.show()
    return(ani)

def get_RP(skeleton_coords):
    jp = get_joint_pairs()
    jn = np.arange(len(jp))
    comb = combinations(jn,2)
    
            
class GymnDataSet:
    
    def __init__(self,pickle_file_name,load_now=True):
        self.pkl_file_name = pickle_file_name
        self.threshold = 0.2
        if load_now:
            self.loadData()
    
    def loadData(self):
        with open(self.pkl_file_name,'rb') as f:
            data = pickle.load(f)  
            clip_list = data["ClipList"]
            val_idx = clip_list['CroppedPerson'] /clip_list['NumberOfFrames'] < self.threshold
            self.ClipList = clip_list[val_idx]
            skeletons = {}
            for idx, clip in self.ClipList.iterrows():
                clip_id = clip['PoseClipId']
                sk_list = data['Skeletons'][clip_id]
                cp = data['CroppedPerson'][clip_id]
                ep = data['ExtraPerson'][clip_id]
                frames_with_issues = np.logical_or(cp,ep) 
                vf = []
                frm_idx = 0                
                frames_with_issues = np.logical_or(cp,ep)
                for frame in frames_with_issues:
                    if not frame:
                        vf.append(sk_list[frm_idx])
                    frm_idx += 1  
                skeletons[clip_id] = vf
            self.Skeletons = skeletons
            print('Data loaded and CLEANED!!.')
            
    def SetThreshold(self,new_threshold):
        self.threshold = new_threshold
    
    
    def getClipHeader(self,clip_id):
        c_l = self.ClipList
        idx_clip = c_l['PoseClipId'] == clip_id
        return(c_l[idx_clip])
        
    def getSkeletons(self,clip_id):
        clp_hdr = self.getClipHeader(clip_id)
        if len(clp_hdr)>0:
            return(self.Skeletons[clip_id])
        else:
            print('Clip not found or invalid!')
            return(None)
    

        