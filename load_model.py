import numpy as np
#  这一段必须放在fluid.dygraph.guard()外运行，否则会报错
from paddle import fluid
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
image = np.random.random([128,3,128, 64]).astype('float32')
program, feed_vars, fetch_vars = fluid.io.load_inference_model('infer_model', exe)
print(feed_vars,fetch_vars)
fetch, = exe.run(program, feed={feed_vars[0]: image}, fetch_list=fetch_vars)
print(fetch.shape)