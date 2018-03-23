#predict house prices based on house sizes#
#TODO np.array np.asarray np.asanyarray
#after closing one plot the following code will execute?

#data,inference,loss calculation,optimize
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#create house sizes in range(1000,3500)#
num_house = 160
np.random.seed(40)
house_size = np.random.randint(low=1000,high=3500,size=num_house)

#generate house prices based on house sizes with some noise#
np.random.seed(42)
house_price = house_size*100 + np.random.randint(low=20000,high=70000,size=num_house)

#plot generated house and price#
plt.plot(house_size,house_price,"bx")#rx red x, bx blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()


#nomalize to prevent under/overflow#
def normalize(array):
    return (array-array.mean())/array.std()

#define training size
num_train_samples = math.floor(num_house*0.7)

#define training data
train_size = np.array(house_size[:num_train_samples]) #np.asarray
train_price = np.array(house_price[:num_train_samples]) #np.asanyarray

train_size_norm = normalize(train_size)
train_price_norm = normalize(train_price)

#define test data
test_size = np.array(house_size[num_train_samples:])
test_price = np.array(house_price[num_train_samples:])

test_size_norm = normalize(test_size)
test_price_norm = normalize(test_price)

#placeholders
tf_house_size = tf.placeholder("float",name="house_size")#name,the name in the graph
tf_price = tf.placeholder("float",name="price")

#define variables holding factor and offset
factor = tf.Variable(np.random.randn(),name="factor")
offset = tf.Variable(np.random.randn(),name="offset")

#define operation
price_pred = tf.add(tf.multiply(factor,tf_house_size),offset)

#define loss function
cost = tf.reduce_sum(tf.pow(price_pred-tf_price,2))/(2*num_train_samples)

#optimizer learning rate
learning_rate = 0.1

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#initialize
init = tf.global_variables_initializer()

#launch a session
with tf.Session() as sess:
    sess.run(init)

    display_every = 2
    num_training_iter = 50

    for iteration in range(num_training_iter):
        for(x,y) in zip(train_size_norm,train_price_norm):
            #attention feed_dict = {}
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price:y})

        #display
        if(iteration +1)%display_every == 0:

            c = sess.run(cost,feed_dict = {tf_house_size : train_size_norm, tf_price : train_price_norm})
            print("iteration #:" , '%04d'%(iteration + 1),"cost=","{:.9f}".format(c),\
                  "size_factor=",sess.run(factor),"price_offset=",sess.run(offset))



    print("success")
    training_cost = sess.run(cost,feed_dict={tf_house_size:train_size_norm,tf_price:train_price_norm})
    print("cost=",training_cost,"factor=",sess.run(factor),"offset=",sess.run(offset))

    train_house_size_mean = train_size.mean()
    train_house_size_std = train_size.std()
    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    #plot
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("price")
    plt.xlabel("size(sq.ft)")
    plt.plot(train_size,train_price,'go',label='Training data')
    plt.plot(test_size,test_price,'mo',label='Testing data')
   #set it to the real value, not the nomalized ones
    plt.plot(train_size_norm*train_house_size_std + train_house_size_mean,(sess.run(factor)*train_size_norm+sess.run(offset))*train_price_std+train_price_mean,label = 'Learned Regresstion')

    plt.legend(loc = 'upper left')
    plt.show()