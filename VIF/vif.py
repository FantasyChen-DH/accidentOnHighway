import pickle

import numpy as np
import cv2

import math
import glob, os

from VIF.HornSchunck import HornSchunck


class VIF:
    def __init__(self):
        self.subSampling = 3
        self.rows = 100
        self.cols = 134
        self.hs = HornSchunck()
        self.clf = pickle.load(open(os.path.dirname(os.path.realpath(__file__))+"\model-svm1.sav", 'rb'))
        self.no_crash = 0
        self.crash = 0


    def createBlockHist(self, flow, N, M):

        height, width = flow.shape
        B_height = int(math.floor((height - 11) / N))
        B_width = int(math.floor((width - 11 ) / M))

        frame_hist = []

        for y in np.arange(6, height - B_height - 5, B_height):
            for x in np.arange(6, width - B_width - 5, B_width):
                block_hist = self.createHist(flow[y:y + B_height - 1, x:x + B_width - 1])
                #print(block_hist)
                frame_hist.append(block_hist)

        return np.array(frame_hist).flatten()

    def createHist(self, mini_flow):
        H = np.histogram(mini_flow, np.arange(0, 1, 0.05))
        H = H[0]/float(np.sum(H[0]))
        return H

    def process(self, frames):
        flow = np.zeros([self.rows, self.cols]) #row, cols
        index = 0
        N = 4
        M = 4
        shape = (self.cols,self.rows)

        for i in range(0, len(frames) - self.subSampling - 5, self.subSampling * 2):

            index += 1
            prevFrame = frames[i + self.subSampling]
            currFrame = frames[i + self.subSampling * 2]
            nextFrame = frames[i + self.subSampling * 3]

            prevFrame = cv2.resize(prevFrame, shape)
            currFrame = cv2.resize(currFrame, shape)
            nextFrame = cv2.resize(nextFrame, shape)

            u1, v1, m1 = self.hs.process(prevFrame, currFrame)
            u2, v2, m2 = self.hs.process(currFrame, nextFrame)

            delta = abs(m1 - m2)
            flow = flow + (delta > np.mean(delta))

        flow = flow.astype(float)
        if index > 0:
            flow = flow/index

        feature_vec = self.createBlockHist(flow, N, M)


        return feature_vec











