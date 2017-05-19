import tensorflow as tf
import numpy as np
import math
import timeit
from random import randint
tf.set_random_seed(0)

def load_train(filename):
    rval=[]
    handle=open(filename)
    line_i=0
    for line in handle:
        line_i=line_i+1
        line=line.strip()
        temp_store=line.split()
        if line_i==1:
            Hdim=len(temp_store)-1
        xdata=temp_store[:Hdim]
        if(temp_store[Hdim]==0):
            ydata=[1,0]
        else:
            ydata=[0,1]
        rval.append((xdata,ydata))
		
    handle.close()
    return rval
	
def load_test(filename):
    xdata_temp=[]
    ydata_temp=[]
    handle=open(filename)
    line_i=0
    for line in handle:
        line_i=line_i+1
        line=line.strip()
        temp_store=line.split()
        if line_i==1:
            Hdim=len(temp_store)-1
        xdata_temp.append(temp_store[:Hdim])
        if(temp_store[Hdim]==0):
            ydata_temp.append([1,0])
        else:
            ydata_temp.append([0,1])
    xdata = np.array(xdata_temp, dtype='float64')
    ydata = np.array(ydata_temp, dtype='int32')
    rval=(xdata,ydata)
    handle.close()
    return rval
		
train_tuple=load_train('train.dat')
testX,testY=load_test('test.dat')
handle1=open('result_train.dat','w')
handle2=open('result_test.dat','w')
handle1.write('%-7s  %-5s  %-10s  %-10s \n'%('iterate','epoch','accuracy','cross_E'))
handle2.write('%-7s  %-5s  %-10s  %-10s \n'%('iterate','epoch','accuracy','cross_E'))

ndata_train=len(train_tuple)
ndata_test=testX.shape[0]
Hdim=testX.shape[1]
batch_size=30
niter=2000

X = tf.placeholder(tf.float32, [None, Hdim])
Y_ = tf.placeholder(tf.float32, [None,2])
lr = tf.placeholder(tf.float32)
L1 = 6
W1 = tf.Variable(tf.truncated_normal([Hdim, L1], stddev=0.1)) 
B1 = tf.Variable(tf.ones([L1])/10)
W2 = tf.Variable(tf.truncated_normal([L1, 2], stddev=0.1))  
B2 = tf.Variable(tf.zeros([2]))
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Ylogits = tf.matmul(Y1, W2) + B2
Y = tf.nn.softmax(Ylogits) 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*batch_size
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer= tf.train.AdamOptimizer(lr)
train_step=optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0 

start_time = timeit.default_timer()

for icount in range (niter):
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-icount/decay_speed)

    batch_X_temp=[] 
    batch_Y_temp=[]
    for i in range (batch_size):
        arg=randint(1,ndata_train)
        myX,myY=train_tuple[arg-1]
        batch_X_temp.append(myX)
        batch_Y_temp.append(myY)
        batch_X=np.array(batch_X_temp,dtype='float64')
        batch_Y=np.array(batch_Y_temp,dtype='int32')
		
    if icount%10==0 or icount<10:
        a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y})
        epoch=icount//(ndata_train/batch_size)+1
        print(str(icount) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        handle1.write('%6d  %4d  %9.6f  %9.5f\n'%(icount,epoch,a,c))
		
    if icount%50==0 or icount<50:
        a, c = sess.run([accuracy, cross_entropy], {X: testX, Y_: testY})
        epoch=icount//(ndata_train/batch_size)+1
        print(str(icount) + ": ***** epoch " + str(epoch) + " ***** test accuracy:" + str(a) + " test loss: " + str(c))
        handle2.write('%6d  %4d  %9.6f  %9.5f\n'%(icount,epoch,a,c))
	
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate})	
	
end_time = timeit.default_timer()
print(('The training time = %.2fm' % ((end_time - start_time) / 60.)))
handle1.close()
handle2.close()
		
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		

















