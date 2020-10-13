from paddle import fluid
from paddle.fluid.dygraph.nn import *


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, groups=1, act=None):
        """
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()
        # 创建卷积层,bias为false,不在后面加激活层
        self.conv = Conv2D(num_channels=num_channels, num_filters=num_filters,
                           filter_size=filter_size, stride=stride, padding=(filter_size - 1) // 2,
                           groups=groups, act=None, bias_attr=False)
        # 创建BatchNorm层
        self.batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.batch_norm(y)
        return y


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()
        self.shortcut = shortcut

        # 创建第一个卷积层 3x3
        if not shortcut:
            self.conv0 = ConvBNLayer(num_channels=num_channels, num_filters=num_filters, filter_size=3, stride=2,
                                     act='relu')

        else:
            self.conv0 = ConvBNLayer(num_channels=num_channels, num_filters=num_filters, filter_size=3, stride=1,
                                     act='relu')

        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(num_channels=num_filters, num_filters=num_filters, filter_size=3,
                                 stride=1, act=None)

        # 创建第三个卷积 1x1
        self.conv2 = ConvBNLayer(num_channels=num_filters, num_filters=num_filters, filter_size=1,
                                 act=None)
        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels, num_filters=num_filters,
                                     filter_size=1, stride=stride)

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        # 输入和最后一层卷积的输出相加
        y = fluid.layers.elementwise_add(x=short, y=conv2)
        y = fluid.layers.relu(y)

        return y


#  构建命令式编程模式（动态图）网络
class Mylayer(fluid.dygraph.Layer):

    def __init__(self, num_classes=751):
        super(Mylayer, self).__init__()
        self.num_classes = num_classes
        self.conv = fluid.dygraph.Sequential(
            Conv2D(num_channels=3, num_filters=64, filter_size=3, stride=1, padding=1),
            BatchNorm(64, act="relu", in_place=True),
            Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        )

        self.layer1 = BottleneckBlock(64, 64, 2, True)
        self.layer2 = BottleneckBlock(64, 128, 2, False)
        self.layer3 = BottleneckBlock(128, 256, 2, False)
        self.layer4 = BottleneckBlock(256, 512, 2, False)
        self.avg_pool = Pool2D(pool_size=(4, 8), pool_stride=1, pool_type='avg')

        self.classifier = fluid.dygraph.Sequential(
            Linear(512, 256),
            BatchNorm(256, in_place=True, act='relu'),
            Dropout(0.5),
            Linear(256, num_classes, act='softmax'),
        )

    # 传label进来就用于训练，不传就只输出特征
    def forward(self, x, label=None):
        # torch.Size([64, 3, 128, 64])
        # input shape [16, 64, 128, 3]

        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = fluid.layers.reshape(x, [x.shape[0], x.shape[1]])
        # B x 128
        if label is not None:
            # classifier
            x = self.classifier(x)
            loss = fluid.layers.cross_entropy(x, label)

            avg_loss = fluid.layers.mean(loss)
            label = fluid.layers.reshape(label, [label.shape[0], 1])
            acc = fluid.layers.accuracy(x, label)
            return acc, avg_loss
        else:
            # x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
