import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

tensors=[]
temp=np.array([])
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.ckpt-20000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    var = [v for v in tf.global_variables()]
    print (tf.global_variables())
    # x = np.zeros([11,11,3,32])    
    # var[0]=tf.reshape(var[0],[11,11,3,32])
    for i, x in enumerate(var):
        temp=sess.run(var[i])
        tensors = tensors + [temp]

    names = [tensor.name for tensor in tf.global_variables()]
    # print (tensors[0].shape)
    # print (var)
    index = names.index("alexnet_v2/conv1/weights:0")
    # tensors[0] = tensors[0][:,:,:,::2]
    tensors[index] = tensors[index][:,:,:,::2]
    # print (names)
    # print (tf.trainable_variables())
tf.reset_default_graph()
# saver=tf.train.Saver()
with tf.Session() as sess:

    for i, x in enumerate(tensors):
        if (names[i]=="global_step:0"):
            my_variable = tf.get_variable(names[i][:-2],initializer=tensors[i],trainable=False)    
        else:
            my_variable = tf.get_variable(names[i][:-2],initializer=tensors[i])
    sess.run(tf.global_variables_initializer())
    # save_path= saver.save(sess,"hello.ckpt")
    # print(tf.global_variables())
    saver=tf.train.Saver(tf.global_variables())
    save_path= saver.save(sess,"/home/advait/Desktop/block_1_50/block_1_50.ckpt")
    