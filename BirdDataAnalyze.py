import os
import csv
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from peakutils.plot import plot as pplot
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from scipy import stats
from scipy import integrate

accel_vol = 9.8/12 #9.8 kg/s^2 per 12 voltage value changes
sample_freq = 1000 # Hz
PeakThreshold = 9.8*3 # the threshold for peak detection
NectarThreshold = 0.3
Dist = 10 #the minnimum distance between two peaks for accelerometer
NectarDist = 20

morph = "a0A60c-1C4"
#morph = str(raw_input("Which morphology is it?\n")) + "l070r1.5R025v020p000"
morph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),morph)
trial_dir = os.listdir(morph_path)

visit_info = []
trial_count = 0
for item in trial_dir:
    trial_path = os.path.join(morph_path, item)
    if os.path.isdir(trial_path):
        trial_count += 1
        trial_folder = os.path.basename(trial_path)
        print(trial_folder)
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
        visit_file = open('visit_data.csv', 'w')
        visit_write = csv.writer(visit_file, delimiter=',', lineterminator='\n')
        
        # find all the nectar empty events
        empty_time = []
        fill_time = []
        e_data=[]
        for row in e_read:
            e_data.append([int(row[0]),float(row[1])])
        
        # check if  e_data is correct. 
        for i in range(0,len(e_data)-1):
            if e_data[i][0] == 0: 
                empty_time.append(round(float(e_data[i][1]),2))
        
        # read animal present/absent data, and x, y, z data
        animal = []
        for row in v_read:
            animal.append([float(row[0]), float(row[2])])
        animal = np.array(animal)
        
        n = []
        for row in n_read:
            n.append([float(row[0]), float(row[1])])
        n = np.array(n)[:,0]
        
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
        

        accel_raw = np.vstack((x, y, z))
       #accel= np.sqrt((np.power(accel_raw[0,:],2) + np.power(accel_raw[1,:],2) + np.power(accel_raw[2,:],2))) 
        #hit_count1 = peakutils.indexes(accel, thres=PeakThreshold /max(accel), min_dist=10)

        
        # define visits based on nectar empty event
        visit_time = []
        
        for index, time_stamp in enumerate(empty_time):
            idx = (np.abs(animal[:,0] - time_stamp)).argmin() 
            
            if animal[idx,0] < time_stamp:
                visit_start = animal[idx, 0]
                visit_end = visit_start + animal[idx,1]/sample_freq
                visit_time.append([round(visit_start,2), time_stamp, round(visit_end,2)])
            else:
                print("Reading start and end time error occurs in {0}".format(time_stamp))
        
        
        visit_num =  len(visit_time)
        visit_write.writerow(["trial_name", "trial_number", "visit_number",'visit_start', 'nectar_empty', 'visit_end', 'start_empty', 'start_end', 'hit_count', 'lick_count', 'hit/lick'])
        visit_count = 0
        
        lick_count_all = []
        for visit in visit_time:
            visit_count += 1
            xyz_start = (np.abs(t - visit[0])).argmin()
            xyz_end  = (np.abs(t- visit[2])).argmin()
            xyz_index = range(xyz_start,xyz_end)
            t_visit = t[xyz_index]
         
            accel_visit = np.sqrt((np.power(accel_raw[0,xyz_index],2) + np.power(accel_raw[1,xyz_index],2) + np.power(accel_raw[2,xyz_index],2))) 
            hit_count = peakutils.indexes(accel_visit, thres=PeakThreshold /max(accel_visit), min_dist=Dist)
            #pplot(t_visit,accel_visit,hit_count)
            #plt.show()
            
            n_visit = n[xyz_index] - stats.mode(n[xyz_index])[0]
            lick_count = peakutils.indexes(n_visit, thres=NectarThreshold, min_dist=NectarDist)
            lick_count_all.append(lick_count)
            #pplot(t_visit,n_visit,lick_count)
            #plt.show()
            
            this_visit = [trial_folder, trial_count, visit_count, visit[0],visit[1],visit[2], visit[1]-visit[0], visit[2]-visit[0], len(hit_count), len(lick_count), np.float64(len(hit_count))/np.float64(len(lick_count))]
            visit_info.append(this_visit)
            visit_write.writerow(this_visit)

        array = np.array(visit_info)
        e_file.close()
        v_file.close()
        x_file.close()
        y_file.close()
        z_file.close()
        visit_file.close()
        
        
        
        #break

os.chdir(morph_path)
with open(("{0}.csv").format(morph), "w") as morph_file:
    morph_write = csv.writer(morph_file,delimiter=',', lineterminator='\n')
    morph_write.writerow(["trial_name", "trial_number", "visit_number", 'visit_start', 'nectar_empty', 'visit_end', 'start_empty', 'start_end', 'hit_count', 'lick_count', 'hit/lick'])
    
    for item in visit_info:
        morph_write.writerow(item)
    