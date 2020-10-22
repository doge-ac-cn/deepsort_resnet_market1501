import os

import cv2
import numpy as np
#  这一段必须放在fluid.dygraph.guard()外运行，否则会报错
from paddle import fluid
from paddle.fluid.contrib import summary

train_dir = "dataset/people_reid/train"
test_dir = "dataset/people_reid/val"
model_dir = "infer_model_to_feature"

def calculate_avg_cos_simility(feature_list):
    count = 0
    output = 0
    output_list = []
    for first_index in range(0, len(feature_list)):
        for second_index in range(0, len(feature_list)):
            if second_index != first_index:
                simility = \
                    exe2.run(
                        feed={"x": np.array([feature_list[first_index]]), "y": np.array(feature_list[second_index])},
                        fetch_list=[out])[0][0]
                output += simility
                output_list.append(simility)
                count += 1
    return output / count, output_list


def cal_counts(output_list):
    count90 = 0
    count91 = 0
    count92 = 0
    count88 = 0
    count89 = 0
    count80 = 0
    for sim in output_list:

        if sim > 0.8:
            count80 += 1
        if sim > 0.88:
            count88 += 1
        if sim > 0.89:
            count89 += 1
        if sim > 0.9:
            count90 += 1
        if sim > 0.92:
            count92 += 1
        if sim > 0.91:
            count91 += 1
    count80 = str(count80 / len(output_list))
    count88 = str(count88 / len(output_list))
    count89 = str(count89 / len(output_list))
    count90 = str(count90 / len(output_list))
    count91 = str(count91 / len(output_list))
    count92 = str(count92 / len(output_list))
    print("80 : "+count80+"88 : " + count88 + " 89 : " + count89 + " 90 :" + count90 + " 91 : " + count91 + "  92 :" + count92)


place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

image = np.random.random([1, 3, 128, 64]).astype('float32')
program, feed_vars, fetch_vars = fluid.io.load_inference_model(model_dir, exe)
# print(feed_vars, fetch_vars)
fetch, = exe.run(program, feed={feed_vars[0]: image}, fetch_list=fetch_vars)
# print(fetch)

x = fluid.layers.data(name='x', shape=[128, 1], dtype='float32', append_batch_size=False)
y = fluid.layers.data(name='y', shape=[128, 1], dtype='float32', append_batch_size=False)
out = fluid.layers.cos_sim(x, y)
exe2 = fluid.Executor(place)
exe2.run(fluid.default_startup_program())

test_feature_list = []
for class_name in os.listdir(test_dir):

    for image_name in os.listdir(test_dir + "/" + class_name):
        image_path = test_dir + "/" + class_name + "/" + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 128))
        image = np.transpose(image, (2, 1, 0)).astype(np.float32)

        image = image / 255 * 2.0 - 1.0
        image = np.array([image])

        feature = exe.run(program, feed={feed_vars[0]: image}, fetch_list=fetch_vars)[0]
        feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        test_feature_list.append(feature)
    if len(test_feature_list) > 100:
        break
# 计算余弦相似度均值
avg_simility, test_output_list = calculate_avg_cos_simility(test_feature_list)
# 计算不同余弦相似度的占比
cal_counts(test_output_list)

cos_simility_list = []
total_output_list = []
for class_name in os.listdir(train_dir):
    single_class_feature_list = []

    # 获取每个类别每张图的128维特征图输出
    for image_name in os.listdir(train_dir + "/" + class_name):
        image_path = train_dir + "/" + class_name + "/" + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 128))
        image = np.transpose(image, (2, 1, 0)).astype(np.float32)
        image = image / 255 * 2.0 - 1.0
        image = np.array([image])
        feature = exe.run(program, feed={feed_vars[0]: image}, fetch_list=fetch_vars)[0]
        feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        single_class_feature_list.append(feature)

    if len(single_class_feature_list) > 1:
        # 计算每个类别的余弦相似度均值
        avg_simility, output_list = calculate_avg_cos_simility(single_class_feature_list)
        total_output_list += output_list
        cos_simility_list.append(avg_simility)
        cal_counts(total_output_list)

print(cos_simility_list)
