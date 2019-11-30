import os
import numpy as np
import pandas as pd
import cv2 as cv
import mxnet as mx
import pickle
import math

from gluoncv import model_zoo
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.data.transforms.presets.ssd import transform_test

# Initial steps
path_to_videos = "VideoData"
# Loading pre-trained models
detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

detector.reset_class(["person"], reuse_weights=['person'])


def process_directory_videos(source_path):
    file_list = os.listdir(source_path)
    dict_holder = {
            "PoseClipId":0,
            "FileName": "",
            "VideoSourceId":0,
            "ExerciseType" : "",
            "ClipNumber":0,
            "SampleType":""
    }
    df = pd.DataFrame(columns=['PoseClipId','FileName','VideoSourceId',
                               'ExerciseType','ClipNumber','SampleType',
                               'CroppedPerson','ExtraPerson','NumberOfFrames'])
    
    for video_file in file_list:
        spl_fname = video_file.split(sep='_')
        dict_holder["PoseClipId"] = abs(hash(video_file)) % (10 ** 8)
        dict_holder["FileName"] = video_file
        dict_holder["VideoSourceId"] = spl_fname[0]
        dict_holder["ExerciseType"] = spl_fname[1]
        dict_holder["ClipNumber"] = spl_fname[2]
        dict_holder["SampleType"] = spl_fname[3].split(sep='.')[0]
        dict_holder["CroppedPerson"] = 0
        dict_holder["ExtraPerson"] = 0
        dict_holder["NumberOfFrames"] = 0
        df = df.append(dict_holder,ignore_index=True)
        
    return(df)            


def get_crop_dims(a_frame):
    """
    As the video contains a black box around it, this
    function calculates the coordiantes to get rid of it
    """
    gray = cv.cvtColor(a_frame,cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(gray,1,255,cv.THRESH_BINARY)
    contours = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cntB = contours[0]
    return cv.boundingRect(cntB)  # x,y,w,h = cv.boundingRect(cntB)

    
def get_video_frames(source_path,video_file,freq):
    """
    Process a single video file. The function selects the second frame for
    every second of the video, crop it if it is needed, and create an array
    of frames for future processing, e.g keypoint determination. 
    """
    frame_list = []
    full_path = os.path.join(source_path,video_file)
    video_cap = cv.VideoCapture(full_path)
    n_fps = int(video_cap.get(cv.CAP_PROP_FPS)) 
    checkpoint = math.floor(n_fps / freq)
    cnt =1
    sw = True
    frm_count = 0
    if video_cap.isOpened():
        ret, frame = video_cap.read()
        
        while (video_cap.isOpened()):
            ret, frame = video_cap.read()
                
            if ret == True:
                if sw:
                    # Use first frame to get black box 
                    x,y,w,h = get_crop_dims(frame) 
                    print('Cropped Size:',w-x+1,h-y+1)
                    sw = False
                if cnt==checkpoint:
                    cropped_frame = frame[y:y+h,x:x+w]
                    # Converting now the the fram to RGB
                    rgb_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)
                    frame_list.append(mx.nd.array(rgb_frame).astype('uint8'))
                    frm_count = frm_count +1
                    cnt=1
                else:
                    cnt+=1

            else:
                break
            if frm_count>100:
                print('Too much')
                break            
    else:
        print('Error opening the file')
    video_cap.release()
    return frame_list

def get_full_frame_info(a_frame):
    x, frame = transform_test(a_frame, short=512)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs)
    if len(upscale_bbox)>0:
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        pred_coords = pred_coords.asnumpy()
    return  class_IDs, scores, upscale_bbox, pred_coords, confidence, bounding_boxs

def get_skeleton_from_frame(a_frame):
    ok_flag = False
    extra_person_flag = False
    x, frame = transform_test(a_frame, short=512)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs)
    b_coords = 0
    if len(upscale_bbox)>0:
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        pred_coords = pred_coords.asnumpy()
        b_coords = pred_coords[0]
        if pred_coords.shape[0]>2:
            extra_person_flag = True
        if pred_coords.shape[0]==2:
            # doing best guest when two boxes ( subject and background are similar)
            if upscale_bbox[0][3]==512:
                b_coords = pred_coords[1]
        ok_flag= True
    return ok_flag, extra_person_flag, b_coords

    
def get_skeletons(frame_list):
    """
    Process all frames and returns skeleton list
    """
    skeleton_coords = []
    nr_frames = len(frame_list)
    frame_box_ok = np.zeros(nr_frames, dtype='uint8')
    extra_person = np.zeros(nr_frames, dtype='uint8')
    print('# frames :',nr_frames)
    fs =0
    err = 0
    for a_frame in frame_list:
        ok_flag, extra_flag, frame_skeleton = get_skeleton_from_frame(a_frame)
        msg='  '
        if not ok_flag:
            msg = msg + ':CROPPPED PERSON'
            err = err + 1
        if extra_flag:
            msg = msg + ':MULTIPLE PERSONS'
            err = err + 1
        print(fs+1,msg,end=' ')
        if not ok_flag:
            frame_box_ok[fs] = 1
        if extra_flag:
            extra_person[fs] = 1
        fs = fs + 1        
        skeleton_coords.append(frame_skeleton)
        if err>30:
            break
    print('*')        
    print('Finished file.')
    return frame_box_ok, extra_person, skeleton_coords


def scrap_videos(source_path,save_file_name="Skeletons.pickle",freq=30):
    print('Reading directory...')
    clip_list = process_directory_videos(source_path)
    print('Finished')
    print()
    skeletons = {}
    cropped_person = {}
    extra_person = {}
    for index, a_file in clip_list.iterrows():
        fname = a_file['FileName']
        clip_id = a_file['PoseClipId']
        print(' Processing ',fname,' -> ClipID=',clip_id)
        video_frames=get_video_frames(source_path,fname,freq)
        cropped_person[clip_id], extra_person[clip_id],skeletons[clip_id] = get_skeletons(video_frames) 
        a_file['CroppedPerson'] = np.sum(cropped_person[clip_id])
        a_file['ExtraPerson'] = np.sum(extra_person[clip_id])
        a_file['NumberOfFrames'] = len(extra_person[clip_id])
    dataset = {}
    dataset["ClipList"] = clip_list
    dataset["Skeletons"] = skeletons
    dataset["CroppedPerson"] = cropped_person
    dataset["ExtraPerson"] = extra_person
    with open(save_file_name,'wb') as f:
        pickle.dump(dataset,f)
    return(dataset)
    
#my_dataset = scrap_videos(path_to_videos)
