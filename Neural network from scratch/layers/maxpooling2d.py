import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), mode="max"):
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mode = mode
    
    def target_shape(self, input_shape):
        H = int(1 + (input_shape[0] - self.kernel_size[0]) / self.stride[0])
        W = n_W = int(1 + (input_shape[1] - self.kernel_size[1]) / self.stride[1])
        return H, W
    
    def forward(self, A_prev):
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (f_h, f_w) = self.kernel_size
        strideh, stridew = self.stride
        H, W = self.target_shape((H_prev, W_prev))
        A = np.zeros((batch_size, H, W, C_prev))
        for i in range(batch_size):
            for h in range(H):
                h_start = strideh * h
                h_end = h_start + f_h
                for w in range(W):
                    w_start = stridew * w
                    w_end = w_start + f_w
                    for c in range(C_prev):
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.average(a_prev_slice)
                        else:
                            raise ValueError("kir khar")

        return A
    
    def create_mask_from_window(self, x):
        mask = (x == np.max(x))
        return mask
    
    def distribute_value(self, dz, shape):
        (n_H, n_W) = shape
        average = np.prod(shape)
        a = np.ones(shape) * (dz/average)
        return a
    
    def backward(self, dA, A_prev):
       pass
    
    
