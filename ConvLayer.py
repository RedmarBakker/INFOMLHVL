import numpy as np

class ConvLayer:

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

    def process(self, input):
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
