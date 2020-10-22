import paddle
from paddle import fluid
from paddle.fluid.dygraph.nn import *


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, is_downsample):
        super(BottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        # 创建第一个卷积层 3x3
        # 如果下采样，则步长为2，降低特征图大小
        if is_downsample:
            self.conv0 = fluid.dygraph.Sequential(
                Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=3, stride=2, padding=1,
                       bias_attr=False),
                BatchNorm(num_filters, act="relu"),
            )
        # 不下采样则步长为1
        else:
            self.conv0 = fluid.dygraph.Sequential(
                Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=3, stride=1, padding=1,
                       bias_attr=False),
                BatchNorm(num_filters, act="relu"),
            )

        # 创建第二个卷积层 3x3
        self.conv1 = fluid.dygraph.Sequential(
            Conv2D(num_channels=num_filters, num_filters=num_filters, filter_size=3, stride=1, padding=1,
                   bias_attr=False),
            BatchNorm(num_filters),
        )

        # 如果下采样,则输入shortcut前,要先用步长为2的1x1卷积下降特征图大小为[原长/2,原宽/2]
        if is_downsample:
            self.downsample = fluid.dygraph.Sequential(
                Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=1, stride=2,
                       bias_attr=False),
                BatchNorm(num_filters),
            )

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.is_downsample:
            downsample = self.downsample(inputs)
        else:
            downsample = inputs

        # 输入和最后一层卷积的输出相加
        y = fluid.layers.elementwise_add(x=downsample, y=conv1, act="relu")
        return y


#  构建命令式编程模式（动态图）网络
class Mylayer(fluid.dygraph.Layer):

    def __init__(self, num_classes=751):
        super(Mylayer, self).__init__()
        self.num_classes = num_classes
        self.conv = fluid.dygraph.Sequential(
            Conv2D(num_channels=3, num_filters=32, filter_size=3, stride=1, padding=1),
            # Conv2D(num_channels=32, num_filters=32, filter_size=3, stride=1, padding=1),
            BatchNorm(32, act="relu", in_place=True),
            Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        )

        self.layer1 = BottleneckBlock(32, 32, False)
        self.layer2 = BottleneckBlock(32, 32, False)
        self.layer3 = BottleneckBlock(32, 64, True)
        self.layer4 = BottleneckBlock(64, 64, False)
        self.layer5 = BottleneckBlock(64, 128, True)
        self.layer6 = BottleneckBlock(128, 128, False)
        self.global_avg_pool = Pool2D(pool_type='avg', global_pooling=True)

        self.classifier = fluid.dygraph.Sequential(
            Linear(128, 128),
            BatchNorm(128, act='relu', in_place=True),
            Dropout(),
            # Linear(128, num_classes)

        )
        self.fc = Linear(128, self.num_classes, bias_attr=False)
        self.scale = paddle.fluid.layers.create_parameter(shape=[self.num_classes], dtype="float32")

    # 传label进来就用于训练，不传就只输出特征
    def forward(self, x, label=None):

        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # GAP层，全局平均最大池化
        x = self.global_avg_pool(x)
        # 把(1,1,x,x)压缩为(x,x)
        x = fluid.layers.flatten(x, axis=1)

        # if label is not None:
        x = self.classifier(x)

        # 深度余弦度量
        scale = paddle.fluid.layers.l2_normalize(self.scale, axis=-1, epsilon=0.1)
        scale = paddle.fluid.layers.softplus(scale)
        x = self.fc(x)
        x = paddle.fluid.layers.elementwise_mul(scale, x)

        if label is not None:
            # 计算准确率和loss
            label = fluid.layers.unsqueeze(label, axes=[1])
            loss = fluid.layers.softmax_with_cross_entropy(x, label)
            avg_loss = fluid.layers.mean(loss)
            label = fluid.layers.reshape(label, [label.shape[0], 1])
            acc = fluid.layers.accuracy(x, label)
            return acc, avg_loss
        else:
            return x
