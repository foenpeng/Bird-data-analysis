import os
import csv
import numpy as np
import shutil
import peakutils
import time
import matplotlib.pyplot as plt
from peakutils.plot import plot as pplot
from scipy.signal import butter, lfilter, freqz
from scipy import stats, integrate
from scipy.signal import butter, lfilter
from BirdVideoAnalysis import Bird_Video_Reanalysis


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
    
def smooth(x,window_len=3,window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

accel_vol = 9.8/12 #9.8 kg/s^2 per 12 voltage value changes
sample_freq = 1000 # Hz
PeakThreshold = 9.8*3 # the threshold for peak detection
NectarThreshold = 5
Dist = 10 #the minnimum distance between two peaks for accelerometer
NectarDist = 50
MinTimeEngagement = 500
# the followings are for signal filter
lowcut = 1
highcut = 50
fs = 1000
morph_list = ["a0A60c-1C3",
              "a0A60c-1C4",
              "a0A60c-2C0",
              "a0A60c-2C3",
              "a0A60c-2C4",
              "a0A60c-3C0",
              "a0A60c-3C3",
              "a0A60c-3C4"]
morph_list = ["a0A60c-1C0"]
morph_list = ["test"]
program_start_time = time.time()
#morph = str(raw_input("Which morphology is it?\n")) + "l070r1.5R025v020p000"
for morph in morph_list:
    morph_start_time = time.time()
    print("\n"+ morph + "\n")
    morph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),morph)
    trial_dir = os.listdir(morph_path)

    visit_info = []
    bird_full_info = []
    trial_count = 0
    for item in trial_dir:
        trial_path = os.path.join(morph_path, item)
        trial_folder = os.path.basename(trial_path)

        if os.path.isdir(trial_path) and trial_folder[0] == "2":
            print(trial_folder)
            trial_count += 1
            os.chdir(trial_path)
            e_file = open('e_data.csv', 'r')
            e_read =  csv.reader(e_file)
            v_file = open('v_data.csv', 'r')
            v_read = csv.reader(v_file)
            n_file = open('n_data.csv', 'r')
            n_read = csv.reader(n_file)
            x_file = open('x_data.csv', 'r')
            x_read = csv.reader(x_file)
            y_file = open('y_data.csv', 'r')
            y_read = csv.reader(y_file)
            z_file = open('z_data.csv', 'r')
            z_read = csv.reader(z_file)
            visit_file = open('visit_data_BasedOnVideo.csv', 'w')
            visit_write = csv.writer(visit_file, delimiter=',', lineterminator='\n')
            
            # find all the nectar empty events
            
            e_data=[]
            for row in e_read:
                e_data.append([int(row[0]),float(row[1])])
            
            # check if  e_data is correct. 
            empty_time = []
            for i in range(0,len(e_data)-1):
                if e_data[i][0] == 0: 
                    empty_time.append(round(float(e_data[i][1]),2))
            
            # read animal present/absent data, and x, y, z data
            animal = []
            for row in v_read:
                animal.append([float(row[0]), float(row[2])])
            animal = np.array(animal)
            animal = np.round(animal,2)
            
            n = []
            for row in n_read:
                n.append([float(row[0]), float(row[1])])
            n = np.array(n)[:,0]
            #n = n - stats.mode(n)[0]
            #n = butter_bandpass_filter(n, lowcut, highcut, fs, order=3)
            
            x = []
            for row in x_read:
                x.append([float(row[0]), float(row[1])])
            x = np.array(x)
            t = x[:,1]
            x = x[:,0]
            x = (x - stats.mode(x)[0])*accel_vol

            y = []
            for row in y_read:
                y.append([float(row[0]), float(row[1])])
            y = np.array(y)[:,0]
            y = (y - stats.mode(y)[0])*accel_vol
            
            z = []
            for row in z_read:
                z.append([float(row[0]), float(row[1])])
            z = np.array(z)[:,0]
            z = (z - stats.mode(z)[0])*accel_vol
            
            #if True:
            if not os.path.exists(os.path.join(os.getcwd(), "video_analysis.csv")):
                video = Bird_Video_Reanalysis(morph,trial_folder,"video.avi")
                video.run()
                bird_info = np.array(video.bird_info)

            else:
                bird_info = []
                with open("video_analysis.csv",'r') as video_file:
                    #print("video analysis exist")
                    video_reader = csv.reader(video_file)
                    next(video_reader) 
                    bird_list = list(video_reader)
                    bird_info = np.array(bird_list).astype("float")
            
            accel_raw = np.vstack((x, y, z))
            
            # find the indice of consecutive bird absence time.
            def find_bird_absence(data):
                zeros = np.where(data==0)[0]
                zeros_split = np.split(zeros, np.where(np.diff(zeros) != 1)[0]+1)
                zerorun= [array for array in zeros_split]
                return np.concatenate(zerorun)

            zeros = find_bird_absence(bird_info[:,6])
            # count all visits with bird presence greater than 0.2 s
            #visit_time = [[bird_info[i+1,1], bird_info[j-1,1], sum(bird_info[i:j,-1])] for i, j in zip(zeros, zeros[1:]) if (j-1)-i > 2]
            visits = []
            for i, j in zip(zeros, zeros[1:]):
                if (j-1)-i > 2:
                    num_frames = j-1-i
                    bird_mean = np.mean(bird_info[i+1:j,2:], axis=0)
                    visit_time = [bird_info[i+1,1], bird_info[j,1],num_frames,sum(bird_info[i+1:j,-1])] + bird_mean.tolist()
                    visits.append(visit_time)
                    #print(bird_info[i+1:j,3])
                    #print(visit_time)
                    #break
            
            visit_num =  len(visits)
            #print(visits)
            visit_write.writerow(["trial_name", "trial_number", "visit_number",'hit_count', 'lick_count','start_end','visit_start', 'visit_end', 'frame_count','perch_count','contour_area','centroid_x','centroid_y','closest_distance','intersection_area', 'ellipse_x', 'ellipse_y',  'ellipse_ma','ellipse_mi','angle','perch'])
            visit_count = 0
            
            for visit in visits:
                visit_count += 1
                #print(visit)
                xyz_start = (np.abs(t - visit[0])).argmin()
                xyz_end  = (np.abs(t- visit[1])).argmin()
                xyz_index = range(xyz_start,xyz_end)
                t_visit = t[xyz_index]

                
                if len(t_visit) == 0:
                    continue
                accel_visit = np.sqrt((np.power(accel_raw[0,xyz_index],2) + np.power(accel_raw[1,xyz_index],2) + np.power(accel_raw[2,xyz_index],2))) 
                
                if max(accel_visit) == 0:
                    hit_count = []
                else:
                    hit_count = peakutils.indexes(accel_visit, thres=PeakThreshold/max(accel_visit), min_dist=Dist)
                #plt.figure(1)
                #pplot(t_visit,accel_visit,hit_count)
                #plt.show()
                
                # reduce high frequency noise and get rid of super low values, then smooth the flat top peaks. Use 1% percentile instead of mode or median.
                n_visit = n[xyz_index] - stats.scoreatpercentile(n[xyz_index],1,interpolation_method='higher')
                plt.plot(t_visit,n[xyz_index])
                n_visit[n_visit<2] = 0
                n_visit = smooth(n_visit, window_len=51)
                #n_visit = n[xyz_index]
                if max(n_visit) == 0:
                    lick_count = []
                else:
                    lick_count = peakutils.indexes(n_visit, thres=NectarThreshold/max(n_visit), min_dist=NectarDist)
     
                #plt.figure(2)
                #print(lick_count)
                #pplot(t_visit,n_visit[0:-50],lick_count[0:-1])
                #plt.show()

                this_visit = [trial_folder, trial_count, visit_count,len(hit_count), len(lick_count), visit[1]-visit[0]] + visit
                visit_info.append(this_visit)
                visit_write.writerow(this_visit)

            array = np.array(visit_info)
            e_file.close()
            v_file.close()
            x_file.close()
            y_file.close()
            z_file.close()
            visit_file.close()
            
            with open("video_analysis.csv",'r') as video_file:
                video_reader = csv.reader(video_file)
                next(video_reader) 
                bird_list = list(video_reader)
                [row.insert(0,trial_folder) for row in bird_list]
            bird_full_info.extend(bird_list)
            
            perch_frame_folder = os.path.join(morph_path, morph+'_perch_frames')
            if not os.path.exists(perch_frame_folder):
                os.makedirs(perch_frame_folder)

                
            perch_frame = bird_info[bird_info[:,12] == 1]
            perch_frame_time = perch_frame[:,0]
            print(len(perch_frame_time))
            cwd = os.getcwd()
            img_dir = os.listdir(os.path.join(cwd,"BirdImgs"))
            for i in img_dir:
                i_split = i.split('~')
                img_time = float(i_split[-2])
                
                if img_time in perch_frame_time:
                    shutil.copy2(os.path.join(cwd,"BirdImgs/"+i),perch_frame_folder)
                    
            
            #break

    os.chdir(morph_path)
    with open(("{0}_BasedOnVideo.csv").format(morph), "w") as morph_file:
        morph_write = csv.writer(morph_file,delimiter=',', lineterminator='\n')
        morph_write.writerow(["trial_name", "trial_number", "visit_number",'hit_count', 'lick_count','start_end','visit_start', 'visit_end', 'frame_count','perch_count','contour_area','centroid_x','centroid_y','closest_distance','intersection_area', 'ellipse_x', 'ellipse_y',  'ellipse_ma','ellipse_mi','angle','perch'])
        
        for item in visit_info:
            morph_write.writerow(item)


    with open(("{0}_VideoAnalysis.csv").format(morph), "w") as video_analysis_full:
        video_full_write = csv.writer(video_analysis_full,delimiter=',', lineterminator='\n')
        video_full_write.writerow(["trial_name",'frame_count','time','contour_area','centroid_x','centroid_y','closest_distance','intersection_area', 'ellipse_x', 'ellipse_y',  'ellipse_ma','ellipse_mi','angle','perch'])
        
        for item in bird_full_info:
            video_full_write.writerow(item)
    print(morph+" time: ",time.time()-morph_start_time)
    os.chdir(os.path.dirname(morph_path))
print("Execution time is:", time.time()-program_start_time) 