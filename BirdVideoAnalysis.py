import numpy as np
import cv2
import os
import csv
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import shutil

class Bird_Video_Reanalysis:
    
    def __init__(self, morph, trial_name, video_name):
               
        self.cap = cv2.VideoCapture(video_name)
        self.morph = morph
        self.trial_name = trial_name
        self.i = 0
        self.sensor_position = (291,129)
        self.flower_top_line_p1 = (325,71) #(302, 108)
        self.flower_top_line_p2 = (267,170) #(276, 156) 
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.perch_hover_model = pickle.load(open(os.path.join(self.path,"perch_hover_SVM.txt"), 'rb'))
        self.perch_hover_scaler = pickle.load(open(os.path.join(self.path,"perch_hover_scaler.txt"), 'rb'))
        self.perch_hover_pca = pickle.load(open(os.path.join(self.path,"perch_hover_pca.txt"), 'rb')) 
        self.bird_info = []
        self.rep_saved = False
        self.img_folder = os.path.join(os.getcwd(), "BirdImgs")
        
        
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
        else:
            shutil.rmtree(self.img_folder)
            os.makedirs(self.img_folder)

    def cnt_fitelps(self,img,cnt):
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)
        return ellipse

    def cnt_fitline(self,img,cnt):
        rows,cols = img.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

    def image_part(self,img,x,y,w,h):
        part = img[y:y + h, x:x + w]
        return part

    def clean_circle(self,img):
        # function to clean the circle drawn in every frame
        blank = np.zeros((480, 640, 1), dtype=np.uint8)
        cv2.circle(blank, (300, 250), 150, (255, 255, 255), 2) # use (300,300) for c-1C0 trials before 6_7_14_6
        dst = cv2.inpaint(img, blank, 3, cv2.INPAINT_TELEA)
        return dst
    
    def perch_hover(self,img):
        img = cv2.resize(img,(70,80))
        img_flat = img.reshape(-1,5600)
        img_flat = img_flat.astype(np.float64)
        img_scaler = self.perch_hover_scaler.transform(img_flat)
        img_pca = self.perch_hover_pca.transform(img_scaler)        
        return self.perch_hover_model.predict(img_pca)
        

    def run(self):
    
        with open("m_data.csv","r") as m_file, open("video_analysis.csv","w") as video_analysis:
            video_analysis_writer = csv.writer(video_analysis, delimiter=',', lineterminator='\n')
            video_analysis_writer.writerow(['frame_count','time','contour_area','centroid_x','centroid_y','closest_distance','intersection_area', 'ellipse_x', 'ellipse_y',  'ellipse_ma','ellipse_mi','angle','perch'])
            m_read =  csv.reader(m_file)
            bird_m = []
            for row in m_read:
                bird_m.append([float(row[0]), float(row[1])])
            bird_m = np.array(bird_m)
            self.frame_count = bird_m.shape[0]
            
            while(self.i < self.frame_count):
                #print(i)
                self.i += 1

                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    print("Reading video error for trial {}".format(self.trial_name))
                    break

                # pre-processing image and find bird contour
                self.frame = self.clean_circle(self.frame)
                self.frame_part = self.image_part(self.frame, 0,80,350,400) # use 0,100,350,400 for c-1C0 trials before 6_7_14_6
                self.blw = cv2.cvtColor(self.frame_part,cv2.COLOR_RGB2GRAY)
                self.ret, self.thr_blw = cv2.threshold(self.blw,80,255,cv2.THRESH_BINARY_INV)
                _, self.cnts, _ = cv2.findContours(self.thr_blw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                self.blank = np.zeros(self.thr_blw.shape, dtype="uint8")

                # check bird position
                if self.cnts:
                    self.largest = sorted(self.cnts, key=cv2.contourArea, reverse=True)[0]
                    self.bird_img = cv2.drawContours(self.blank.copy(), [self.largest], -1, 255, -1)
                   
                    # normal area size should be between 3000 and 15000
                    self.area = cv2.contourArea(self.largest)
                    if self.area > 15000 or self.area < 2000:
                        self.row = [self.i,bird_m[self.i-1,1], self.area,0,0,0,0,0,0,0,0,0,0]
                        video_analysis_writer.writerow(self.row)
                        self.bird_info.append(self.row)
                        continue

                    # check whether the bird's body intersect with flower top plane
                    self.line = cv2.line(self.blank.copy(), self.flower_top_line_p1, self.flower_top_line_p2, 255, 3)
                    self.intersect = cv2.bitwise_and(self.bird_img, self.line)
                    self.intersect_area = np.count_nonzero(self.intersect)
                    
                    if self.intersect_area == 0:
                        self.row = [self.i,bird_m[self.i-1,1], self.area,0,0,0,0,0,0,0,0,0,0]
                        video_analysis_writer.writerow(self.row)
                        self.bird_info.append(self.row)
                        continue
                    
                    # find the centroid
                    self.M = cv2.moments(self.largest)
                    self.cx = int(self.M['m10'] / self.M['m00'])
                    self.cy = int(self.M['m01'] / self.M['m00'])
                    
                    # check the distance from the sensor to the closest bird contour pixel
                    self.distance = round(cv2.pointPolygonTest(self.largest, self.sensor_position, measureDist=True),2)

                    # fit the bird with an ellipse to estimate the orientation
                    self.ellipse = self.cnt_fitelps(self.blw,self.largest)

                    # determine if the bird is perching or hovering
                    self.perch = self.perch_hover(self.bird_img)[0]
                    
                    self.row = [self.i,bird_m[self.i-1,1], self.area, self.cx,self.cy,self.distance,self.intersect_area,self.ellipse[0][0], self.ellipse[0][1], self.ellipse[1][0], self.ellipse[1][1], self.ellipse[2], self.perch]
                    video_analysis_writer.writerow(self.row)
                    self.bird_info.append(self.row)
                    
                    # save the image if it depicts bird touching flower
                    img_file_name = self.img_folder+"\{0}~{1}~{2}~{3}.jpg".format(self.morph,self.trial_name,self.i,bird_m[self.i-1,1])
                    cv2.imwrite(img_file_name, self.bird_img)

                    if not self.rep_saved:
                        cv2.drawContours(self.blw, [self.largest], -1, 255, -1)
                        cv2.line(self.blw, self.flower_top_line_p1, self.flower_top_line_p2, 255, 3)
                        cv2.circle(self.blw,self.sensor_position,1,0,-1)
                        cv2.putText(self.blw,"Distance:{0}    Intersect:{1}".format(round(self.distance,1), self.intersect_area),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
                        cv2.imwrite("rep_img.jpg", self.blw)
                        self.rep_saved = True
                        
                    #cv2.imshow("orig", self.blw )
                    #cv2.imshow("intersect",self.intersect)
                    
                    #k = cv2.waitKey(300) & 0xff
                    #if k == 27:
                        #break

            if self.i != self.frame_count:
                print("Video error: frame number does not match in trial {}".format(self.trial_name))
            self.cap.release()
            cv2.destroyAllWindows()

if __name__=="__main__":
    v = Bird_Video_Reanalysis("0","test","video.avi")
    v.run()