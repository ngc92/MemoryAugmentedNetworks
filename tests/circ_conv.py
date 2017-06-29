
import numpy as np

def circ_conv_(x, y):
    n = x.shape[1]
    w = np.zeros((len(x), n))
    for i in range(n):
        for j in range(n):
            w[:, i] += x[:, j] * y[:, (i - j) % n]
    return w

def circ_conv_numpy(w, shifts, shiftws):
    n = w.shape[1]
    sv = np.zeros(w.shape)
    sv[:, np.array(shifts) % n] = shiftws
    return circ_conv_(w, sv)


def circ_conv_numpy_test(): 
    for i in range(10):
        n_batches = np.random.randint(20)
        n_dims    = np.random.randint(100)

        x = np.random.random((n_batches, n_dims))

        shifts = [-1, 0, 1]

        s1 = [1., 0., 0.]
        s2 = [0., 1., 0.]
        s3 = [0., 0., 1.]

        assert (circ_conv_numpy(x, shifts, s1) == np.roll(x, -1, axis=1)).all()
        assert (circ_conv_numpy(x, shifts, s2) == x).all()
        assert (circ_conv_numpy(x, shifts, s3) == np.roll(x, 1, axis=1)).all()

circ_conv_numpy_test()
