import paddle
import paddle.fluid as fluid
from PIL import Image
import os
import numpy as np

from deepsort_net import Mylayer

# 指定参数
train_dir = "dataset/people_reid/train"
test_dir = "dataset/people_reid/val"
BATCH_SIZE = 128
NUM_EPOCHES = 1000
RANDOM_POOL_SIZE = 100000  # 随机池
num_classes = 751
# 指定gpu
place = fluid.CUDAPlace(0)
fluid.enable_imperative(place)


# 指定投入训练数据的reader生成器
def train_generater():
    def __reader__():
        imageclass = 0
        for class_name in os.listdir(train_dir):

            for image_name in os.listdir(train_dir + "/" + class_name):
                image_path = train_dir + "/" + class_name + "/" + image_name
                image = Image.open(image_path).resize((64, 128), Image.ANTIALIAS)
                image = np.array(image).astype(np.float32)
                image = np.reshape(image, [3, 64, 128])

                yield image / 255.0 * 2.0 - 1.0, imageclass
            imageclass += 1

    return __reader__


# 指定投入预测数据的reader生成器
def test_generater():
    def __reader__():
        imageclass = 0
        for class_name in os.listdir(test_dir):

            for image_name in os.listdir(test_dir + "/" + class_name):
                image_path = test_dir + "/" + class_name + "/" + image_name
                image = Image.open(image_path).resize((64, 128), Image.ANTIALIAS)
                image = np.array(image).astype(np.float32)
                image = np.reshape(image, [3, 64, 128])

                yield image / 255.0 * 2.0 - 1.0, imageclass
            imageclass += 1

    return __reader__


# 使用sample数据生成器作为DataLoader的数据源
train_loader = fluid.io.DataLoader.from_generator(capacity=10)
train_loader.set_sample_generator(paddle.reader.shuffle(train_generater(), RANDOM_POOL_SIZE), batch_size=BATCH_SIZE,
                                  places=place)

test_loader = fluid.io.DataLoader.from_generator(capacity=10)
test_loader.set_sample_generator(test_generater(), batch_size=BATCH_SIZE, places=place)

# 初始化神经网络
DeepSortNet = Mylayer()

# 添加优化器
adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001, parameter_list=DeepSortNet.parameters())

with fluid.dygraph.guard():
    # 执行训练/预测
    for epoch_idx in range(NUM_EPOCHES):

        # 训练
        DeepSortNet.train()
        # 添加准确度管理器
        accuracy_manager = fluid.metrics.Accuracy()
        print("epoch  %d  " % epoch_idx)
        for data in train_loader():
            # 投入数据
            image, label = data
            label = label.astype(np.int64)
            image = fluid.dygraph.to_variable(image)
            label = fluid.dygraph.to_variable(label)
            # 执行前向
            acc, loss = DeepSortNet(image, label)
            # 执行反向
            loss.backward()
            # 梯度更新
            adam.minimize(loss)
            # 清除梯度
            adam.clear_gradients()
            accuracy_manager.update(acc.numpy(), BATCH_SIZE)

        print("train accuracy: %.6f , loss %.2f" % ( accuracy_manager.eval(), loss))

        # 评估
        DeepSortNet.eval()
        accuracy_manager = fluid.metrics.Accuracy()
        for data in test_loader():
            # 投入数据
            image, label = data
            label = label.astype(np.int64)
            image = fluid.dygraph.to_variable(image)
            label = fluid.dygraph.to_variable(label)
            # 执行前向
            acc, loss = DeepSortNet(image, label)

            accuracy_manager.update(acc.numpy(), BATCH_SIZE)
            # 每隔十次训练,输出一次准确率

        print("test accuracy: %.6f , loss %.2f" % (accuracy_manager.eval(), loss))
