import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), initialize_method="random"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.initialize_method = initialize_method

        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.parameters = [self.initialize_weights(), self.initialize_bias()]


    def initialize_weights(self):
        if self.initialize_method == "random":
            return np.random.randn((self.kernel_size[0], self.kernel_size[1], self.in_channels, self.out_channels))*0.01
        if self.initialize_method == "xavier":
            return None
        if self.initialize_method == "he":
            return None
        else:
            raise ValueError("Invalid initialization method")
    
    def initialize_bias(self):
        return np.zeros((1, 1, 1, self.out_channels))
    
    def target_shape(self, input_shape):
        input_height, input_width = input_shape
        padding_top, padding_right = self.padding
        kernel_height, kernel_width = self.kernel_size
        H = (input_height + 2*padding_top - kernel_height)/self.stride[0] + 1
        W = (input_width + 2*padding_right - kernel_width)/self.stride[1] + 1
        return H, W
    
    def pad(self, A, padding, pad_value=0):
        A_padded = np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode="constant", constant_values=(pad_value, pad_value))
        return A_padded
    
    def single_step_convolve(self, a_slic_prev, W, b):
        S = np.multiply(a_slic_prev, W)
        Z = np.sum(S)
        Z = Z + b.astype(float)
        return Z

    def forward(self, A_prev):
        W, b = self.parameters
        (batch_size, H_prev, W_prev, C_prev) = A_prev.shape
        (kernel_size_h, kernel_size_w, C_prev, C) = W.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        H, W = self.target_shape(A_prev.shape)
        Z = np.zeros((batch_size, H, W, C))
        A_prev_pad = self.pad(A_prev, self.padding)
        for i in range(batch_size):
            for h in range(H_prev):
                h_start = h * stride_h
                h_end = h_start + kernel_size_h
                for w in range(W_prev):
                    w_start = w * stride_w
                    w_end = w_start + kernel_size_w
                    for c in range(C_prev):
                        a_slice_prev = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                        Z[i, h, w, c] = self.single_step_convolve(a_slice_prev, W, b)
        return Z

    def backward(self, dZ, A_prev):
        # bro I just can't be bother with this
        pass
    
    def update_parameters(self, optimizer, grads):
       pass