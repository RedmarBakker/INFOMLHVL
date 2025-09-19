import numpy as np
from Layer import Layer

class ConvLayer(Layer):

    def __init__(self, filters=None, stride=(1, 1)):
        if filters is None:
            filters = []

        self.filters = filters
        self.stride = stride

    def add_filters(self, kernels):
        if isinstance(kernels, np.ndarray):
            kernels = [kernels]

        for kernel in kernels:
            if kernel.ndim == 2:
                kernel = kernel.reshape(kernel.shape[0], kernel.shape[1], 1)
            elif kernel.ndim == 3:
                pass
            else:
                raise ValueError("Kernel must be 2D or 3D")

            self.filters.append(kernel)

        return self

    def process(self, input: np.ndarray) -> np.ndarray:
        # Assert the input shape and conform it to H,W,D for future uniform processing
        if input.ndim == 2:
            H, W = input.shape
            D = 1
            input = input.reshape((H, W, D))
        elif input.ndim == 3:
            H, W, D = input.shape
        else:
            raise ValueError("Input must be 2D or 3D (H, W) or (H, W, D)")

        # Assert kernel dimensions (from the first kernel)
        kh, kw, kd = self.filters[0].shape
        assert kd == D, f"Filter depth {kd} must match input depth {D}"

        # Assert the stride and output dimensions
        sh, sw = self.stride
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1
        num_filters = len(self.filters)

        output = np.zeros((out_h, out_w, num_filters))

        # Flatten the kernels for vector multiplication
        kernels = np.zeros((num_filters, kh * kw * D))
        for idx in range(num_filters):
            kernels[idx, :] = self.filters[idx].reshape(kh * kw * D)

        # Slice the input into kernel-size patches, flatten and vector-multiply with the kernel
        for i in range(out_h):
            for j in range(out_w):
                # Slice
                region = input[i * sh:i * sh + kh, j * sw:j * sw + kw, :]

                # reshape into vector
                patch_vector = region.reshape(kh * kw * D)  # shape (kh*kw*D,)

                # multiply with all transposed kernels
                result = patch_vector @ kernels.T

                # store in output
                output[i, j, :] = result

        return output

class ReLULayer(Layer):
    def process(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def process(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 2:
            input = input.reshape(input.shape[0], input.shape[1], 1)

        H, W, D = input.shape
        ph, pw = self.pool_size
        sh, sw = self.stride

        out_h = (H - ph) // sh + 1
        out_w = (W - pw) // sw + 1
        output = np.zeros((out_h, out_w, D))

        for d in range(D):
            for i in range(out_h):
                for j in range(out_w):
                    region = input[i*sh:i*sh+ph, j*sw:j*sw+pw, d]
                    output[i, j, d] = np.max(region)

        if output.shape[-1] == 1:
            return output.squeeze()

        return output

class NormalizeLayer(Layer):
    def process(self, input):
        # Normalize for each channel using axis=(0, 1)
        mean = input.mean(axis=(0, 1))
        std = input.std(axis=(0, 1))

        # Ensure std is not zero to avoid division by zero
        std[std == 0] = 1

        return (input - mean) / std

class FlattenLayer(Layer):
    def process(self, input):
        result = 1

        for dim in input.shape:
            result *= dim

        return input.reshape(result)


class SoftmaxLayer(Layer):
    def process(self, input):
        assert input.ndim == 1

        exps = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
