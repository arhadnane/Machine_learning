# -*- coding: utf-8 -*-
"""
By Adnane ARHARBI
"""
import math
import random
from PIL import Image
from PIL import ImageFilter
import numpy as np

random.seed(0)

res = {(0,0,0,0):"0",(0,0,0,1):"1",(0,0,1,0):"2",(0,0,1,1):"3",(0,1,0,0):"4",(0,1,0,1):"5",(0,1,1,0):"6",(0,1,1,1):"7",(1,0,0,0):"8",(1,0,0,1):"9"}

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

def traite_img ():
        llj =[]
        for lsimage in ['0','1','2','3','4','5','6','7','8','9']:
            img= Image.open("number/"+lsimage+".jpg").convert("L")
            img=img.filter(ImageFilter.DETAIL)
            img=img.resize((10,10))
            img=np.asarray(img)  # Transform picture to numpy table
            img=img.flatten() 
            img=[0 if n<=1 else n/255 for n in img]
            llj.append(img)
        #print(llj)
        return llj
    
class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))
    
    def verif(self, patterns):
        lls = []
        #print("p  = ", patterns)
        #print(patterns, '->', self.update(patterns))
        pp =  self.update(patterns)
        for i in range(len(pp)):
            if pp[i] >= 0.60:
                lls.append(1)
            else:
                lls.append(0)
       
        
        print (res[tuple(lls)])

        
    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=900, N=0.1, M=0.01):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('errorTrain %-.5f' % error)
                
                

def demo():

    pat = [ 
     [lstdemo[0],[0,0,0,0]],
     [lstdemo[1],[0,0,0,1]],
     [lstdemo[2],[0,0,1,0]],
     [lstdemo[3],[0,0,1,1]],
     [lstdemo[4],[0,1,0,0]],
     [lstdemo[5],[0,1,0,1]],
     [lstdemo[6],[0,1,1,0]],
     [lstdemo[7],[0,1,1,1]],
     [lstdemo[8],[1,0,0,0]],
     [lstdemo[9],[1,0,0,1]]
     ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(100,10,4)
    # train it with some patterns
    n.train(pat)
    # test it
    print("*************************TEST******************************")
    n.test(pat)
    print("***********************************************************")
    n.verif(lstdemo[0])
    n.verif(lstdemo[1])
    n.verif(lstdemo[2])
    n.verif(lstdemo[3])
    n.verif(lstdemo[4])
    n.verif(lstdemo[5])
    n.verif(lstdemo[6])
    n.verif(lstdemo[7])
    n.verif(lstdemo[8])
    n.verif(lstdemo[9])

    
#programme principal
if __name__ == '__main__':
    lstdemo =  traite_img()
    #predictImg= traie_img_Predic()
    demo()

