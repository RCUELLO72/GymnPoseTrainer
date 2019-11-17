"""
Utilities to query the dataset
"""
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import itertools
import pandas as pd

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
    
def get_joint_point_list(keypoint_list,joint_list):
    prod_list = list(itertools.product(keypoint_list,joint_list))
    curated_list = []
    for key_point,joint in prod_list:
        sw = key_point == joint[0] or key_point == joint[1]
        if not sw :
            curated_list.append([key_point,joint])
    return(curated_list)

def scale_linear_bycolumn(rawpoints, high=255.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)    

def convert_to_colorrange(rawpoints):
    d = scale_linear_bycolumn(rawpoints)
    return d.flatten().astype(np.uint8)
    
def length_square(p1,p2):
    # returns square of distance b/w two points 
    x1,y1 = p1
    x2,y2 = p2
    dy = y2-y1  
    dx = x2-x1  
    return dx*dx + dy*dy

def get_angles(p1,p2,p3):
    # Square of lengths be a2, b2, c2 
    a2 = length_square(p2,p3)
    b2 = length_square(p1,p3)
    c2 = length_square(p1,p2)
    # length of sides be a, b, c 
    a = math.sqrt(a2)
    b = math.sqrt(b2)    
    c = math.sqrt(c2)
    # From Cosine law 
    alpha = math.acos((b2 + c2 - a2)/(2*b*c))
    betta = math.acos((a2 + c2 - b2)/(2*a*c))
    gamma = math.acos((a2 + b2 - c2)/(2*a*b))
    return alpha*180/math.pi, betta*180/math.pi, gamma*180/math.pi
    
def calc_jointpoint_features(keypoint,joint_a,joint_b):
    # Calculates the distance("Magnitude") between two points
    # and relative position descriptors
    x1,y1 = joint_a
    x2,y2 = joint_b
    x3,y3 = keypoint
    dy = y2-y1  
    dx = x2-x1  
    distance_to_joint = 0
    if abs(dx)>0:
        m = dy/dx
        # generates an line equation of the form ax + by + c = 0 (p1,p2)
        a = m             
        b = -1
        c =  y1 - (m*x1)  
        nm = abs((a*x3)+(b*y3)+c)
        dm = math.sqrt(a**2+b**2)
        if dm>0:
            distance_to_joint = abs(nm/dm)
    if distance_to_joint>5:
        t,b,g = get_angles(keypoint,joint_a,joint_b)
        feat_list = [distance_to_joint, t, b]
    else:
        feat_list = [0, 0, 0]
    return feat_list
    
def calc_descriptor_feats(p1,p2):
    # Calculates the distance("Magnitude") between two points
    # and relative position descriptors
    x1,y1 = p1
    x2,y2 = p2
    dy = y2-y1  # Relative Position Descriptor (x)
    dx = x2-x1  # Relative Poisition Descriptor (y)
    magnitude = math.sqrt(dx*dx + dy*dy) 
    feat_list = [abs(dx), abs(dy), magnitude]
    return feat_list   

def normalize_blk(blk) :
    maxv = np.max(blk)
    if not maxv == 0:
        blk = scale_linear_bycolumn(blk)
    return blk

def normalize_data(blk_a,blk_b):
    blk_a = normalize_blk(blk_a)
    blk_b = normalize_blk(blk_b)
    blk = np.append(blk_a,blk_b,axis=1)
    return blk.flatten().astype(np.uint8)
   
            
class GymnDataSet:
    '''
    Class designed to read skeleton dataset from a picklefile
    and transform it to RGB images
    '''
    def __init__(self,pickle_file_name,load_now=True):
        self.pkl_file_name = pickle_file_name
        self.threshold = 0.2
        self.keypoints = 17   # 17 Keypoints COCO
        self.scene_size = 20 # 15 frames per scene (
        self.scene_skip_frames = 10  # 3 frames = 1s
        self.joints = [[0, 1], [1, 3], [0, 2], [2, 4],
                       [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                       [5, 11], [6, 12], [11, 12],
                       [11, 13], [12, 14], [13, 15], [14, 16]] 
        if load_now:
            self.loadData()
    
    def loadData(self):
        with open(self.pkl_file_name,'rb') as f:
            data = pickle.load(f)  
            clip_list = data["ClipList"]
            val_idx = clip_list['CroppedPerson'] /clip_list['NumberOfFrames'] < self.threshold
            self.ClipList = clip_list[val_idx]
            self.pruneSkeletons(data)
            self.calcClassList()
            self.setListsForFeatureExtraction()
            self.setClass()
            print('Data loaded and CLEANED!!.')
    
    def pruneSkeletons(self,data):
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
            for bad_frame in frames_with_issues:
                if not bad_frame:
                    vf.append(sk_list[frm_idx])
                frm_idx += 1  
            skeletons[clip_id] = vf
        self.Skeletons = skeletons
        
            
    def calcClassList(self):
        # Obtain summary of clips (counts)
        cl = self.ClipList[['ExerciseType','SampleType','PoseClipId']].groupby(['ExerciseType','SampleType']).count()
        self.Summary = cl.rename(columns={"PoseClipId":"Count"})
        # Establish Class List for classification (NN-LSTM)
        cl = self.Summary.reset_index()
        self.ClassList = cl.drop(columns=['Count'])
        
    def setListsForFeatureExtraction(self):
        self.RelativeDescriptorList = list(itertools.combinations(list(range(self.keypoints)),2))
        self.JointPointList = get_joint_point_list(list(range(self.keypoints)),self.joints)
        
            
    def setThreshold(self,new_threshold):
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
            
    def getDescriptorData(self,sk_list,sk_id,last_data=None):
        skeleton = sk_list[sk_id]
        data = []
        for kp_a,kp_b in self.RelativeDescriptorList:
            p1 = skeleton[kp_a]
            p2 = skeleton[kp_b]
            sk_feat = calc_descriptor_feats(p1,p2)
            data.append(sk_feat)
        data = np.array(data)
        if last_data is None:
            diff = np.zeros((len(self.RelativeDescriptorList),3))
        else:
            diff = data - last_data
        return data, np.abs(diff)
    
    def getJointPointData(self,sk_list,sk_id,last_data=None):
        data = []
        skeleton = sk_list[sk_id]
        for keypoint,joint in self.JointPointList:
            kp = skeleton[keypoint]
            joint_a = skeleton[joint[0]]
            joint_b = skeleton[joint[1]]
            sk_feat = calc_jointpoint_features(kp,joint_a,joint_b)
            data.append(sk_feat)  
        data = np.array(data)
        if last_data is None:
            diff = np.zeros((len(self.JointPointList),3))
        else:
            diff = data - last_data
        return data, np.abs(diff)
    
    def getSkeletonFeatures(self,clip_id,sk_id,last_desc_data=None,last_jointpoint_data=None):
        sk_list = self.getSkeletons(clip_id)
        if not sk_list is None:
            if sk_id > len(sk_list)-1:
                print('Error, beyond skeleton range ')
            else:
                descriptor_data, temporal_data_descriptor = self.getDescriptorData(sk_list,sk_id,last_desc_data)
                jointpoint_data, temporal_data_jointpoint = self.getJointPointData(sk_list,sk_id,last_jointpoint_data)
        return descriptor_data, temporal_data_descriptor,jointpoint_data, temporal_data_jointpoint
                
    def getFramePic(self,clip_id,sk_id,last_desc_data=None,last_jointpoint_data=None):
        d1,t1,d2,t2 = self.getSkeletonFeatures(clip_id,sk_id,last_desc_data,last_jointpoint_data)
        pt1 = normalize_data(d1,t1)
        pt2 = normalize_data(d2,t2)
        img = np.append(pt1,pt2)
        return img.reshape(94,8,3),d1,d2
    
    def GetScene(self,clip_id,sc_number=0):
        sk_list = self.getSkeletons(clip_id)
        first_frame = sc_number * self.scene_skip_frames
        descriptor_data = None
        jointpoint_data = None
        if (first_frame+self.scene_size>len(sk_list)):
            print('Not enough frames')
            return None
        else:
            for i in range(self.scene_size):
                a_frame,descriptor_data,jointpoint_data = self.getFramePic(clip_id,first_frame+i,descriptor_data,jointpoint_data)
                if i==0:
                    final_img = a_frame
                else:
                    final_img = np.append(final_img,a_frame,axis=1)
        return final_img
    
    def setClass(self):
        cl = self.ClassList.reset_index().rename(columns={'index':'ClassId'})
        ndf = pd.merge(self.ClipList,cl,on=['ExerciseType','SampleType'])
        self.ClipList = ndf
        
    def createDataSet(self, nr_scenes=3):
        labels = []
        data = []
        for index, clip in self.ClipList.iterrows():
            clip_id = clip['PoseClipId']
            for i in range(nr_scenes):
                print(clip_id,' ',i)
                data_point = self.GetScene(clip_id,i)
                class_id = clip['ClassId']
                if not data_point is None:
                    labels.append(class_id)
                    data.append(data_point)
        return labels, data
                    
                
        
                
                
        
        
            