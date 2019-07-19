# How to build your own conv in topi.
import numpy as np

import tvm
import topi
import topi.generic

def topi_conv(N, Cin, Cout, H, W, Kh, Kw):
    X = tvm.placeholder((N, Cin, H, W), name='X')
    K = tvm.placeholder((Cout, Cin, Kh, Kw), name='W')
    # def conv2d(input, filter, strides, padding, dilation, layout='NCHW', out_dtype=None):
    with tvm.target.create("cuda"):
        out = topi.nn.conv2d(X, K, (1, 1), (1, 1), (1, 1))
        sconv = topi.generic.nn.schedule_conv2d_nchw(out)
        print(tvm.lower(sconv, [X, K], simple_mode=True))

    act = np.random.randn(N, Cin, H, W).astype(np.float32)
    wei = np.random.randn(Cout, Cin, Kh, Kw).astype(np.float32)
    nout = np.zeros_like(act).astype(np.float32)

    ctx = tvm.gpu(0)
    xx = tvm.nd.array(act, ctx)
    ww = tvm.nd.array(wei, ctx)
    oo = tvm.nd.array(nout, ctx)
    return sconv, [X, K, out], (xx, ww, oo)


if __name__ == '__main__':
    sconv, [X, K, out], (x, w, oo) = topi_conv(1, 5, 5, 12, 12, 3, 3)
    func = tvm.build(sconv, [X, K, out], 'cuda')
    print(func)
    print(func(x, w, oo))
