import cv2
import paddle
import paddle.fluid as fluid
import os
import numpy as np
from paddle.fluid.dygraph import to_variable, TracedLayer
from tqdm import trange
import matplotlib.pyplot as plt
from deepsort_net import Mylayer

# 指定参数
train_dir = "dataset/people_reid/train"
test_dir = "dataset/people_reid/val"
BATCH_SIZE = 64
NUM_EPOCHES = 200
RANDOM_POOL_SIZE = 10000  # 随机池
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
                image = cv2.imread(image_path)
                image = cv2.resize(image, (64, 128))
                image = np.transpose(image, (2, 1, 0)).astype(np.float32)

                image = image / 255 * 2.0 - 1.0
                yield image, imageclass

            imageclass += 1

    return __reader__


# 指定投入预测数据的reader生成器
def test_generater():
    def __reader__():
        imageclass = 0
        for class_name in os.listdir(test_dir):

            for image_name in os.listdir(test_dir + "/" + class_name):
                image_path = test_dir + "/" + class_name + "/" + image_name
                image = cv2.imread(image_path)
                image = cv2.resize(image, (64, 128))
                image = np.transpose(image, (2, 1, 0)).astype(np.float32)

                image = image / 255 * 2.0 - 1.0
                yield image, imageclass
            imageclass += 1

    return __reader__


# 使用sample数据生成器作为DataLoader的数据源
train_loader = fluid.io.DataLoader.from_generator(capacity=10)
train_loader.set_sample_generator(paddle.reader.shuffle(train_generater(), RANDOM_POOL_SIZE), batch_size=BATCH_SIZE,
                                  places=place)

test_loader = fluid.io.DataLoader.from_generator(capacity=10)
test_loader.set_sample_generator(test_generater(), batch_size=BATCH_SIZE,
                                 places=place)

# optimizer = fluid.optimizer.AdamOptimizer(
#     learning_rate=learning_rate,
#     parameter_list=DeepSortNet.parameters(),
#     regularization=fluid.regularizer.L2Decay(regularization_coeff=5e-4)
# )

# plot figure
x_epoch = []
record = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="acc")


def draw_curve(epoch, train_loss, train_acc, test_loss, test_acc):
    global record
    record['train_loss'].append(train_loss)
    record['train_acc'].append(train_acc)
    record['test_loss'].append(test_loss)
    record['test_acc'].append(test_acc)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_acc'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_acc'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train_plt.jpg")


with fluid.dygraph.guard():
    # 初始化神经网络
    DeepSortNet = Mylayer()

    learning_rate = 0.001
    boundaries = [30, 60]
    values = [0.1, 0.01, 0.001]
    # 添加优化器
    optimizer = fluid.optimizer.AdamOptimizer(
        # learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
        learning_rate=learning_rate,
        parameter_list=DeepSortNet.parameters(),
        regularization=fluid.regularizer.L2Decay(regularization_coeff=5e-6)
    )
    # 执行训练/预测
    for epoch_idx in trange(NUM_EPOCHES):
        print(optimizer.current_step_lr())
        # 训练
        DeepSortNet.train()
        # 添加准确度管理器
        accuracy_manager = fluid.metrics.Accuracy()
        print("epoch  %d  " % epoch_idx)
        total_train_loss = 0
        for data in train_loader():
            # 投入数据
            image, label = data
            label = label.astype(np.int64)
            image = fluid.dygraph.to_variable(image)
            label = fluid.dygraph.to_variable(label)
            # 执行前向
            train_acc, train_loss = DeepSortNet(image, label)
            # 执行反向
            train_loss.backward()
            # 梯度更新
            optimizer.minimize(train_loss)
            # 清除梯度
            optimizer.clear_gradients()
            total_train_loss += np.mean(train_loss.numpy())
            accuracy_manager.update(train_acc.numpy(), BATCH_SIZE)
        print("train accuracy: %.6f , loss %.2f" % (accuracy_manager.eval(), total_train_loss))

        # 评估
        DeepSortNet.eval()
        total_test_loss = 0
        test_accuracy_manager = fluid.metrics.Accuracy()
        for data in test_loader():
            # 投入数据
            image, label = data
            label = label.astype(np.int64)
            image = fluid.dygraph.to_variable(image)
            label = fluid.dygraph.to_variable(label)
            # 执行前向
            test_acc, test_loss = DeepSortNet(image, label)
            total_test_loss += np.mean(test_loss.numpy())
            test_accuracy_manager.update(test_acc.numpy(), BATCH_SIZE)

        print("test accuracy: %.6f , loss %.2f" % (test_accuracy_manager.eval(), total_test_loss))
        # 绘制图
        draw_curve(epoch_idx, total_train_loss, accuracy_manager.eval(), total_test_loss,
                   test_accuracy_manager.eval())

    # 保存特征模型
    image = np.random.random([1, 3, 128, 64]).astype('float32')
    image = fluid.dygraph.to_variable(image)
    out_dygraph, static_layer = TracedLayer.trace(DeepSortNet, inputs=[image])
    static_layer.save_inference_model('infer_model_to_feature')

    # 保存预测模型
    for data in train_loader():
        # 投入数据
        image, label = data
        break
    label = label.astype(np.int64)
    image = fluid.dygraph.to_variable(image)
    label = fluid.dygraph.to_variable(label)
    out_dygraph, static_layer = TracedLayer.trace(DeepSortNet, inputs=[image, label])
    static_layer.save_inference_model('infer_model_to_eval')
