import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose

def check_multi_softmax_with_shape(shape, xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SoftmaxOutput(data=X, label=L, multi_output=True)
    x = mx.random.uniform(-1, 1, shape, ctx = xpu)
    l = mx.nd.empty((shape[0], shape[2]), ctx = xpu)
    l[:] = np.random.randint(0, shape[1]-1, (shape[0], shape[2]))
    print l.asnumpy()
    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward()
    print(exec1.outputs[0].asnumpy())
    exec1.backward()
    print(grad.asnumpy())

def test_3d_convolution_with_type(shape, kernel, xpu):
    num_filter = 1
    num_group = 1
    kernel = (3, 3, 3)
    shape = (3, 4, 9, 9, 9)

    x = mx.sym.Variable('x')
    w = mx.sym.Variable('w')
    b = mx.sym.Variable('b')
    y1 = mx.sym.Convolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group, kernel=kernel)

    exe1 = y1.simple_bind(mx.gpu(), x=shape)

    for arr1 in zip(exe1.arg_arrays ):
        arr1[:] = np.random.normal(size=arr1.shape)
    exe1.forward(is_train=True)
    exe1.backward(exe1.outputs[0])

    for arr1 in zip(exe1.outputs + exe1.grad_arrays):
        print arr1.asnumpy()

def test_3d_convolution():

    shape = (3, 1, 240, 10, 10)
    kernel = (24, 5, 5)
    stride = (24, 1, 1)
    num_filter = 10
    weight_shape = (10, 1)+kernel
    xpu = mx.gpu(1)
    X = mx.sym.Variable('X')
    W = mx.sym.Variable('W')
    B = mx.sym.Variable('B')
    # data3d = mx.symbol.Variable('data')
    conv3d = mx.symbol.Convolution(data=X, weight=W, bias=B, num_filter=num_filter, kernel=kernel, stride=stride)
    x = mx.random.uniform(-1, 1, shape=shape, ctx=xpu)
    w = mx.random.uniform(-1, 1, shape=weight_shape, ctx=xpu)
    b = mx.random.uniform(-1, 1, shape=(num_filter,), ctx=xpu)

    exec1 = conv3d.bind(xpu, args=[x, w, b])
    exec1.forward()

    print exec1.outputs[0].asnumpy().shape

def test_convolution_grouping():
    num_filter = 4
    num_group = 2
    kernel = (3, 3)
    shape = (1, 4, 9, 9)

    x = mx.sym.Variable('x')
    w = mx.sym.Variable('w')
    b = mx.sym.Variable('b')
    y1 = mx.sym.Convolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group, kernel=kernel)
    xslice = mx.sym.SliceChannel(data=x, num_outputs=num_group, axis=1)
    wslice = mx.sym.SliceChannel(data=w, num_outputs=num_group, axis=0)
    bslice = mx.sym.SliceChannel(data=b, num_outputs=num_group, axis=0)
    y2 = mx.sym.Concat(*[mx.sym.Convolution(data=xslice[i], weight=wslice[i], bias=bslice[i],
                                            num_filter=num_filter//num_group, kernel=kernel)
                       for i in range(num_group)])

    exe1 = y1.simple_bind(mx.cpu(), x=shape)
    exe2 = y2.simple_bind(mx.cpu(), x=shape, w=(num_filter, shape[1]//num_group, kernel[0], kernel[1]), b=(num_filter,))
    for arr1, arr2 in zip(exe1.arg_arrays, exe2.arg_arrays):
        arr1[:] = np.random.normal(size=arr1.shape)
        arr2[:] = arr1
    exe1.forward(is_train=True)
    exe1.backward(exe1.outputs[0])
    exe2.forward(is_train=True)
    exe2.backward(exe2.outputs[0])

    for arr1, arr2 in zip(exe1.outputs + exe1.grad_arrays, exe2.outputs + exe2.grad_arrays):
        np.testing.assert_allclose(arr1.asnumpy(), arr2.asnumpy(), rtol=1e-3)

if __name__ == '__main__':
    test_convolution_grouping()
    test_3d_convolution()
