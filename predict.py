# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:31:31 2019

@author: zun
"""

# -*- coding: utf-8 -*-
from django.http import HttpResponse
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
min_, max_ = 0, 19
class rbf_bp:
    # 对输入值进行径向基向量计算
    def kernel_(self, x_):
        #函数：两点之间的欧式距离
        self.distant_ = lambda x1,x2:np.sqrt(np.sum(np.square(x1-x2)))
        #函数：高斯核
        #self.Gaussian = lambda x:np.exp(-np.power(x/self.gamma,2))  
        self.Gaussian = lambda x:x**self.gamma 
        mount_ = x_.shape[0]
        x_dis = np.zeros((mount_,self.num_)) #中间矩阵:存储两点之间的距离
        matrix_ = np.zeros((mount_,self.num_)) #距离，进行高斯核变换
        for i in range(mount_):
            for j in range(self.num_):
                x_dis[i,j]=self.distant_(x_[i],self.x_nodes[j])
                matrix_[i,j]=self.Gaussian(x_dis[i,j])
        return matrix_
 
    def __init__(self,x_nodes,y_nodes,gamma):
        #节点的x坐标值
        self.x_nodes = x_nodes
        #高斯系数
        self.gamma = gamma
        self.num_ = len(y_nodes) #节点数
        matrix_ = self.kernel_(x_nodes)
        #计算初始化权重weights_
        weights_ = np.dot(np.linalg.pinv(matrix_),y_nodes.copy())
        #定义一个两层的网络，第1层为高斯核函数节点的输出，第2层为回归的值
        self.x_ = tf.placeholder(tf.float32,shape=(None,x_nodes.shape[0]),name="x_")
        self.y_ = tf.placeholder(tf.float32,shape=(None),name="y_")
        weights_ = weights_[:, np.newaxis]
        self.weights = tf.Variable(weights_,name = "weights", dtype=tf.float32)
        self.biaes = tf.Variable(0.0,name = "biaes", dtype=tf.float32)
        self.predict_ = tf.matmul(self.x_,self.weights) + self.biaes
        self.loss = tf.reduce_mean(tf.square(self.y_-self.predict_))
        self.err_rate = tf.reduce_mean(tf.abs((self.y_-self.predict_)/self.y_))
 
    def train(self,x_train,y_train,x_test,y_test,batch_size,learn_rate,circles_):
       
        x_train = self.kernel_(x_train)
        x_test = self.kernel_(x_test)
        self.train_ = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        saver = tf.train.Saver()
        size_ = x_train.shape[0] #训练集的数量
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for step_ in range(circles_): #训练次数
                start = int((step_*batch_size)%(size_-1))
                end = start+batch_size
                if end<(size_-1):
                    in_x = x_train[start:end,:]
                    in_y = y_train[start:end]
                else:
                    end_ = end%(size_-1)
                    in_x = np.concatenate((x_train[start:size_-1,:],x_train[0:end_,:]))
        
                sess.run(self.train_, feed_dict={self.x_:in_x,self.y_:in_y})
           # os.mkdir("./Model")
            saver.save(sess,"Model/model.ckpt")
 
    def predict(self,x_data,y_data):
        x_data = self.kernel_(x_data)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,"Model/model.ckpt")
            prediction = sess.run(self.predict_, feed_dict={self.x_:x_data,self.y_:y_data})
        return prediction
    

def getnums():
    nums=[[],[],[],[],[]]#每位 21期数据
   
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3676.400 QQBrowser/10.4.3469.400}',
           'Host':'zst.aicai.com'}
    r = requests.get('http://zst.aicai.com/gaopin_cqssc/',headers = headers)

    soup = BeautifulSoup(r.text,"html.parser")
    num = soup.find_all("td",class_="c_fbf5e3 bd_rt_a")
   
    
    for k in range(0,30):
        if(k>8):
            #只保存最新21期
            sub = num[k].find_next_sibling()
            nums[0].append(int(sub.text))
            for i in range(1,5):
                sub = sub.find_next_sibling()
                nums[i].append(int(sub.text))
    
        

    return nums


def getline(x1,x2,y1,y2):#计算两点之间直线
    k=(y2-y1)/(x2-x1)
    b=y1-k*x1
    return k,b
def intergral(k,B,a,b):#计算定积分值
    return k*(a**2)/2+B*a- k*(b**2)/2-B*b
def adjust(nums):
    #调整数据，使数据更平滑
    turn=[-2,-1,1,2] #调整范围数组，一个数的左右两位
    flag=np.zeros(20) #1不许修改，-1，0 需要修改
    for i in range(0,19):
   
        if(flag[i]!=-1):
            if(abs(nums[i]-nums[i+1])<6):#相差不超过五的数则绑定在一起不用修改
                flag[i+1]=1
                flag[i]=1
            else:
                if(flag[i]==0):#当前未绑定，进入下一位数
                    continue
                else:#当前已经绑定，进入下下位数
                    flag[i+1]=-1
                    
                    
                    
                #flag 为1则不用修改 ，其余值需要调整
    count=0
    for i in range(0,20):
       if(flag[i]!=1):
      
            count+=1
            if(i==0):#第一位需要修改的情况
                mini=10
                t=0
                for j in range(0,4):
                    sub=abs((nums[0]+turn[j]+10)%10-nums[1])
                    if (sub<mini):
                        mini=sub
                        t=j
                nums[0]=(nums[0]+turn[t]+10)%10
            elif(i==19):#最后一位需要修改的情况
                 mini=10
                 t=0
                 for j in range(0,4):
                     sub=abs((nums[19]+turn[j]+10)%10-nums[18])
                     if (sub<mini):
                         mini=sub
                         t=j
                
                 nums[19]=(nums[19]+turn[t]+10)%10
            else:#中间的数字需要修改的情况
                mini=10
                t=0
                for j in range(0,4):
          
                    sub=abs((nums[i]+turn[j]+10)%10-(nums[i-1]+nums[i+1])/2)
         
                    if (sub<mini):
                        mini=sub
                        t=j
             
                nums[i]=(nums[i]+turn[t]+10)%10
    
    return nums

def polish(y):
        #一次磨光函数
    newy=np.zeros(len(y))
    newy[0]=y[0]
    newy[len(y)-1]=y[len(y)-1]
    for i in range(0,len(y)-2):
       k1,b1= getline(i,i+1,y[i],y[i+1])# 计算折线一
       k2,b2= getline(i+1,i+2,y[i+1],y[i+2]) #计算折线二
       h=0.5
       newy[i+1]=intergral(k1,b1,i+1,i+1-h)/(2*h)+intergral(k2,b2,i+1+h,i+1)/(2*h)#根据公式得出磨光后的角点值
    return newy
            
def getPrediction(nums):
     #生成对应函数表达式
    x = np.linspace(min_,max_,1000)#横坐标
    y = np.array(nums)
   # y=polish(adjust(y[0:20]))
    y = np.array(nums[0:20])

    
    #确定多项式的次数
    maxy = y[np.argmax(y)]
    miny = y[np.argmin(y)]
    derta = maxy - miny
    if(0<=derta<3):
        exponent = 3
    elif(3<=derta<7):
        exponent = 5
    else:
        exponent = 7
    
    z1 = np.polyfit(np.linspace(min_,max_,max_-min_+1),y,exponent)#求得二次多项式系数
    p1 = np.poly1d(z1)#将系数代入方程，得到函式p1
 
    # 横坐标数据集
    x_input = np.linspace(min_,max_,1000)[:, np.newaxis]
    # 纵坐标(即彩票号码数据集)
    y_temp = np.zeros((x.shape[0]))
    for i in range(1000):
        x_ = x[i]
        y_temp[i] = p1(x_)
    y_input = y_temp[:, np.newaxis]
    
    # 训练集和数据集的划分
    #x_train,y_train,x_test,y_test = x_input[0:900,:],y_input[0:900,:],x_input[900:1000,:],y_input[900:1000,:]
    x_input = np.linspace(min_,max_,20)[:, np.newaxis]
    y_input = y[:,np.newaxis]
    x_train,y_train,x_test,y_test = x_input[0:18,:],y_input[0:18,:],x_input[18:20,:],y_input[18:20,:]
    #生成100个构造rbf神经网络的节点
    ##x_nodes = np.linspace(min_,max_,50).reshape((50,1))
    x_nodes = np.random.uniform(min_,max_,size=[100,1])
    y_nodes = np.zeros((x_nodes.shape[0]))
    for i in range(100):
        x_ = x_nodes[i]
        y_nodes[i] = p1(x_) 


    rbf_ = rbf_bp(x_nodes,y_nodes,1)#构造rbf神经网络
    rbf_.train(x_train,y_train,x_test,y_test,10,0.0001,1000)#训练rbf
    pResult = rbf_.predict(np.array([21]),y_test[0:20])##！预测时的结果与y_test无关
    #print(pResult[0][0])
    #绘制结点拟合曲线
    return int(pResult[0][0])%10

def calculate(request):
    nums=getnums()
    r=[]
  
    for i in range(0,5):
        r.append(getPrediction(adjust(nums[i])))
     
  
    data='{"result":'+tostr(r)+'}'
    return (data)

