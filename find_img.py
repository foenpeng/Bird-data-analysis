import os
import csv
import numpy as np
import pandas as pd
import cv2



morph = "a0A60c-1C0"
#morph = str(raw_input("Which morphology is it?\n")) + "l070r1.5R025v020p000"
morph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),morph)
trial_dir = os.listdir(morph_path)

count = 0
all_frame = np.zeros((400, 350), dtype=np.float64)
trial_count = 0
for item in trial_dir:
    trial_path = os.path.join(morph_path, item)
    trial_folder = os.path.basename(trial_path)
    
    if os.path.isdir(trial_path) and trial_folder[0] == "2":
        trial_count += 1
        print(trial_folder)
        os.chdir(trial_path)    
        
        with open("video_analysis.csv",'r') as video_file:
            video_reader = csv.reader(video_file)
            #next(video_reader) 
            bird_list = list(video_reader)
            bird_info = pd.DataFrame(bird_list[1:],columns=bird_list[0])
            cols = ["perch","contour_area","frame_count", "intersection_area"]
            bird_info[cols] = bird_info[cols].apply(pd.to_numeric, errors='coerce')
        
        useful_frame = list(bird_info.loc[(bird_info["perch"] == 0) & (bird_info["contour_area"] > 2000) & (bird_info["contour_area"] < 15000) & (bird_info["intersection_area"] > 0), "frame_count"])
        print(len(useful_frame))
        cwd = os.getcwd()
        img_folder = os.path.join(cwd,"BirdImgs")
        img_dir = os.listdir(img_folder)
        if len(img_dir)==0:
            continue
        else:
            print(img_dir[0])
            
            for i in img_dir:
                splited = i.split('~')
                #print(splited)
                frame_id = int(splited[-2])
                
                if frame_id in useful_frame:
                    frame = cv2.imread(img_folder + "/" + i,0)
                    #print(frame[200,:])
                    if frame.shape == (380,350):
                        frame = np.concatenate((frame,np.zeros((20,350),dtype=np.uint8)),axis = 0)
                    
                    all_frame += np.round(frame/255)
                    #print(all_frame[200,:])
                    count+=1
                
print(count,np.amax(all_frame))
all_frame_percent = 1 - all_frame/count
#print(all_frame_percent[200,:])
#cv2.line(all_frame_percent, (325,71), (267,170), 0, 3)
#cv2.circle(all_frame_percent,(291,129),1,255,-1)
cv2.imshow("all_frame",all_frame_percent)
k = cv2.waitKey(5000) & 0xff

os.chdir(morph_path) 
cv2.imwrite(morph+"_{0}_summed.jpg".format(count),all_frame_percent*255)    
