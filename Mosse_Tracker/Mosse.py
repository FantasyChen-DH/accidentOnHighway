

import numpy as np
import cv2




class MOSSE:
    def __init__(self, frame, cut_size,num_of_traning_imgs = 10,learning_rate = 0.225,psrGoodness = 10):
        #get the xmin,ymin, xmax ,ymax for all the corners in the cut_Size
        xmin, ymin, xmax, ymax = cut_size
        xmin -= 0
        ymin -= 0
        xmax += 0
        ymax += 0
        self.dx =[]
        self.dy = []
        self.centers = []
        self.learning_rate = learning_rate
        self.num_of_traning_imgs = num_of_traning_imgs
        self.psr_goodness = psrGoodness
        self.updated_last_time = True

        #get width and height of the cut_size
        #cv2.getoptimaldftsize faster the tracker according to the opencv document
        self.width, self.height = map(cv2.getOptimalDFTSize, [xmax - xmin, ymax - ymin])
        # self.width = xmax - xmin
        # self.height = ymax - ymin
        self.area = self.width * self.height
        #calculate the center you of the cut image frame
        self.center = x, y = xmin + 0.5 * (self.width - 1), ymin + 0.5 * (self.height - 1)
        self.size = self.width, self.height

        #take a capture of the frame
        img = cv2.getRectSubPix(frame, (self.width, self.height), (x, y))

        #creating window of the cut_size
        self.win = cv2.createHanningWindow((self.width, self.height), cv2.CV_32F)
        g = np.zeros((self.height, self.width), np.float32)

        g[int(self.height/2), int(self.width/2)] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 3.0) #2.0
        g = g / g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)

        self.prepareInitialTracking(frame,img)


    def prepareInitialTracking(self, frame, cut_image):
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        cut_image= cv2.GaussianBlur(cut_image,(3,3),3)

        for index in range(self.num_of_traning_imgs):
            #make random rotations
            random_rotation = self.randomRotation(cut_image)
            H1, H2 = self.computeNumAndDen(random_rotation)
            self.H1 += H1
            self.H2 += H2


        self.updateFilter()
        self.updateTracking(frame,False)

    def updateTracking(self, frame,is_stopped):
        (x, y), (w, h) = self.center, self.size

        if is_stopped and  self.updated_last_time:
            dx = sum(self.dx[-3:]) / 3
            dy = sum(self.dy[-3:]) / 3
            self.center = x + dx, y + dy

            self.dx.append(dx)
            self.dy.append(dy)
            self.updated_last_time = False
        else:

            self.updated_last_time = True

            self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))

            img= cv2.GaussianBlur(img,(3,3),3)
            img = self.preprocess(img)

            self.psr, self.last_resp, (dx, dy) = self.correlateNewImg(img)
            self.good = self.psr > self.psr_goodness
            if not self.good:
                if len(self.dx) == 0:
                    self.dx.append(0)
                    self.dy.append(0)
                else:
                    self.dx.append(self.dx[-1])
                    self.dy.append(self.dy[-1])
                #this is the new center
                self.center = x + self.dx[-1], y + self.dy[-1]

            else:
                # self.learning_rate = max(min(abs(100-self.good)/100)  -0.8 , 0.125)

                self.dx.append(dx)
                self.dy.append(dy)

                #this is the new center
                self.center = x + dx, y + dy

                #cut same width and height for the new img
                self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.center)
                #calcultate num and denumentator for the new image
                H1,H2 = self.computeNumAndDen(img)
                #update the num and den with learning rate to decay old one
                self.H1 = self.H1 * (1.0-self.learning_rate) + H1 * self.learning_rate
                self.H2 = self.H2 * (1.0-self.learning_rate) + H2 * self.learning_rate
                #update the kernal
                self.updateFilter()

        x_new, y_new = self.center
        self.centers.append((x_new, y_new))


    def preprocess(self, img):
        #to get good results with low contrast imgs
        img = np.log(np.float32(img)+1.0)
        mean = img.mean()
        std_deviation = img.std()
        img = (img-mean) / (std_deviation+1e-5)

        #to gradually reduces the pixel values near the edge to zero
        #and focus more on the center
        preprocessed_img = img*self.win
        return preprocessed_img

    def correlateNewImg(self, img):
        F = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        C = cv2.mulSpectrums(F, self.H, 0, conjB=True)
        response = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = response.shape
        _, max_peak_value, _, max_peak_location = cv2.minMaxLoc(response)
        mx,my = max_peak_location

        #calculate dx and dy
        dx = mx - int(w/2)
        dy = my - int(h / 2)

        side_resp = response.copy()
        side_resp = cv2.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        mean = side_resp.mean()
        standard_deviation =side_resp.std()
        psr = (max_peak_value-mean) / (standard_deviation+1e-5)


        return psr,response, (dx,dy)

    def updateFilter(self):
        self.H = self.HFilter(self.H1, self.H2)
        self.H[...,1] *= -1

    def computeNumAndDen(self,img):
        f = self.preprocess(img)
        F = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, F, 0, conjB=True)
        H2 = cv2.mulSpectrums(F, F, 0, conjB=True)
        return H1,H2

    def randomRotation(self,cut_img):
        # get width and height of the img
        height, width = cut_img.shape[:2]
        # make transformation matrix of size 2*3
        transformation = np.zeros((2, 3))

        perentage = 0.2
        # get angle rotation from 0 to 1
        ang = (np.random.rand() - 0.5) * perentage

        cos, sin = np.cos(ang), np.sin(ang)
        transformation[:2, :2] = [[cos, -sin], [sin, cos]]
        transformation[:2, :2] += (np.random.rand(2, 2) - 0.5) * perentage

        # adding the last of transformation to the last index
        transformation[:, 2] = (width / 2, height / 2) - np.dot(transformation[:2, :2], (width / 2, height / 2))
        # look for that :https://www.youtube.com/watch?v=il6Z5LCykZk
        transformed_img = cv2.warpAffine(cut_img, transformation, (width, height), borderMode=cv2.BORDER_REFLECT)
        return transformed_img

    def HFilter(self,Num, Den):
        # applying the eq in the paper to  get the hfilter
        Num_real, Num_imaginary = Num[..., 0], Num[..., 1]
        Den_real, Den_imaginary = Den[..., 0], Den[..., 1]

        h_filter = (Num_real + 1j * Num_imaginary) / (Den_real + 1j * Den_imaginary)
        h_filter = np.dstack([np.real(h_filter), np.imag(h_filter)]).copy()
        return h_filter
    def getCutFramePosition(self,center = -1):
        if center == -1:
            center = self.center
        x = center[0]
        y = center[1]
        xmin = int(x - 0.5*(self.width-1))
        ymin = int(y - 0.5*(self.height-1))
        xmax = int(self.width+xmin)
        ymax = int(self.height+ymin)
        cut_size = [xmin,ymin,xmax,ymax]
        return cut_size

    #return size of the tracker
    def getSizeOfTracker(self):
        return self.width,self.height

    #return center of the tracker
    def getCenterOfTracker(self):
        return self.center

    def getLearningRate(self):
        return self.learning_rate
    def getPsr(self):
        return self.psr

    #is tracking working or lost the car
    def isGood(self):
        return self.good

