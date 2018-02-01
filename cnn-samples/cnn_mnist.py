import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time

# 计算开始时间
start = time.clock()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# sess = tf.InteractiveSession()


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  # 参数ksize是要执行取最值的切片在各个维度上的尺寸，四维数组意义为[batch, height, width, channels]
                          strides=[1, 2, 2, 1], padding='SAME')  # 参数strides是取切片的步长，四维数组意义为四个方向的步长，这里height和width方向都为2，例如原本8x8的矩阵，用2x2切片去pool，会获得5x5的矩阵输出（SAME模式），有效的减少特征维度。


# 第一层卷积
# 现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。
# 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
# 而对于每一个输出通道都有一个对应的偏置量
# 5,5表示patch的大小，1输入的通道数目(彩色图片有r,g,b三个通道)，32表示有多少个神经元（特征）
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
# 最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
# -1代表任何维度，这里是样本数量，MNIST的图像大小为28*28，由于是黑白的，只有一个in_channel。
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 第二层卷积

# 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5, 5, 32, 64])  # 这里32是指上一层的输出通道数目就是这一层的输入的通道数目
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# 输入的是32个14x14的矩阵，权重体现了这层要输出的矩阵个数为64。
# 卷积输出64个12x12的矩阵，因为(14+2−4)/1
# 池化输出64个7x7的矩阵，因为(12+2)/2

# 密集连接层

# 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # tf.matmul（）表示矩阵相乘


# Dropout
# 为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。这样我们可以在训练过程中启用dropout，
# 在测试过程中关闭dropout。
# TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 输出层

# 最后，我们添加一个softmax层，把向量化后的图片x和权重矩阵W相乘，加上偏置b，然后计算每个分类的softmax概率值。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# 类别预测
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数
# 可以很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。


# 训练和评估模型


# 损失函数
# 可以很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))   # 计算交叉熵


# 使用adam优化器来以0.0001的学习率来进行微调

#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)

# 判断预测标签和实际标签是否匹配
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 启动创建的模型，并初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)    # batch 大小设置为50
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(session=sess, feed_dict={
                   x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# 计算程序结束时间
end = time.clock()
print("running time is %g s" % (end - start))
