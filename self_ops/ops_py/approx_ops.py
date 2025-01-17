import torch
from torch.autograd import Function
import approx_ops
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm as tqdm


# global variables
# lut_appmult (torch.Tensor): lookup table for approximate multiplication
# lut_grad_a (torch.Tensor): lookup table for gradient of approximate multiplication w.r.t. the first operand
# lut_grad_b (torch.Tensor): lookup table for gradient of approximate multiplication w.r.t. the second operand
lut_appmult = None
lut_grad_a = None
lut_grad_b = None


def init_lookup_tables(device, file_name, quantization_bit):
    # lookup table parameters
    # must match the CUDA code settings, do not touch!!!!!!!!
    LUT_MAXVAL = 2**quantization_bit - 1
    if quantization_bit == 7:
        # LUT_MAXVAL = 127                               # maximum value of input operands 
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 638   # padding for multiple-thread 4-element loading
    elif quantization_bit == 8:
        # LUT_MAXVAL = 255                               # maximum value of input operands 
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM
    elif quantization_bit == 4:
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 718   # padding for multiple-thread 4-element loading
    elif quantization_bit == 6:
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 830   # padding for multiple-thread 4-element loading
    else:
        raise NotImplementedError(f'quantization_bit = {quantization_bit} is not supported')

    # load lookup tables from file
    print(f'Initializing lookup tables from {file_name}...')

    # global variables
    global lut_appmult, lut_grad_a, lut_grad_b

    # parse file
    state = 'init'
    lut_appmult = torch.zeros(LUT_ELEM_NUM, dtype=torch.uint16, device=device) 
    lut_grad_a = torch.zeros(LUT_ELEM_NUM, dtype=torch.int16, device=device)
    lut_grad_b = torch.zeros(LUT_ELEM_NUM, dtype=torch.int16, device=device)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('LUT for approximate multiplier:'):
                state = 'load_appmult'
                continue
            elif line.startswith('LUT for the gradient of the approximate multiplier w.r.t. the first operand:'):
                state = 'load_grad_a'
                continue
            elif line.startswith('LUT for the gradient of the approximate multiplier w.r.t. the second operand:'):
                state = 'load_grad_b'
                continue
            else:
                if state == 'init': # skip header
                    continue
            # parse lines; in each line, the first element is operand a, the second element is operand b, and the third element is the result
            a, b, res = line.split()
            a, b, res = int(a), int(b), int(res)
            if state == 'load_appmult':
                lut_appmult[a * LUT_COL_NUM + b] = res
            elif state == 'load_grad_a':
                lut_grad_a[a * LUT_COL_NUM + b] = res
            elif state == 'load_grad_b':
                lut_grad_b[a * LUT_COL_NUM + b] = res
            else:
                raise ValueError('Unknown state')

    # # STE (only for test)
    # FACTOR = 16.0
    # for a in range(LUT_ROW_NUM):
    #     for b in range(LUT_COL_NUM):
    #         lut_grad_a[a * LUT_COL_NUM + b] = b * FACTOR
    #         lut_grad_b[a * LUT_COL_NUM + b] = a * FACTOR


class AccGemm(Function):
    @staticmethod
    def forward(ctx, a, b):
        """gemm function forward.
        Args:
            a (torch.Tensor): [M, K]
            b (torch.Tensor): [K, N]
        
        Returns:
            c (torch.Tensor): [M, N]
        """
        # prepare input & output tensors 
        # a_cont, b_cont = a.contiguous(), b.contiguous()
        a_cont, b_cont = a.contiguous().to(torch.float32), b.contiguous().to(torch.float32)
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        # c = a.new_zeros(M, N).to(torch.uint32)
        c = a.new_zeros(M, N).to(torch.float32)

        # call gemm forward function
        approx_ops.acc_gemm_forward_fp32(a_cont, b_cont, c)

        # convert output dtype 
        c = c.to(torch.float32)

        # save for backward
        ctx.save_for_backward(a_cont, b_cont)
        ctx.ori_shape = (M, K, N)

        return c

    @staticmethod
    def backward(ctx, g_c):
        """gemm function backward.
        Args:
            g_c (torch.Tensor): [M, N], float32
        
        Returns:
            g_a (torch.Tensor): [M, K], float32
            g_b (torch.Tensor): [K, N], float32
        """
        # check input dtype
        assert g_c.dtype == torch.float32

        # get saved tensors
        a_cont, b_cont, = ctx.saved_tensors
        M, K, N = ctx.ori_shape

        # prepare output tensors
        g_a, g_b = g_c.new_zeros(M, K), g_c.new_zeros(K, N)

        # call gemm backward function
        g_a = torch.matmul(g_c, b_cont.t())
        g_b = torch.matmul(a_cont.t(), g_c)

        return g_a, g_b

acc_gemm_op = AccGemm.apply


class ApproxGemm(Function):
    @staticmethod
    def forward(ctx, a, b):
        """gemm function forward.
        Args:
            a (torch.Tensor): [M, K]
            b (torch.Tensor): [K, N]
        
        Returns:
            c (torch.Tensor): [M, N]
        """
        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        a, b = a.to(torch.uint8), b.to(torch.uint8)

        # prepare input & output tensors 
        a_cont, b_cont = a.contiguous(), b.contiguous()
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        c = a.new_zeros(M, N).to(torch.uint32)

        # call gemm forward function
        global lut_appmult
        approx_ops.approx_gemm_forward(a_cont, b_cont, c, lut_appmult)

        # convert output dtype 
        c = c.to(torch.float32)

        # save for backward
        ctx.save_for_backward(a_cont, b_cont)
        ctx.ori_shape = (M, K, N)

        return c

    @staticmethod
    def backward(ctx, g_c):
        """gemm function backward.
        Args:
            g_c (torch.Tensor): [M, N], float32
        
        Returns:
            g_a (torch.Tensor): [M, K], float32
            g_b (torch.Tensor): [K, N], float32
        """
        # check input dtype
        assert g_c.dtype == torch.float32

        # get saved tensors
        a_cont, b_cont, = ctx.saved_tensors
        M, K, N = ctx.ori_shape

        # prepare output tensors
        g_a, g_b = g_c.new_zeros(M, K), g_c.new_zeros(K, N)

        # call gemm backward function
        global lut_grad_a, lut_grad_b
        approx_ops.approx_gemm_backward(g_c.contiguous(), a_cont, b_cont, g_a, g_b, lut_grad_a, lut_grad_b)

        return g_a, g_b

approx_gemm_op = ApproxGemm.apply


class ApproxGemmBaseline(Function):
    @staticmethod
    def forward(ctx, a, b):
        """gemm function forward.
        Args:
            a (torch.Tensor): [M, K]
            b (torch.Tensor): [K, N]
        
        Returns:
            c (torch.Tensor): [M, N]
        """
        # save for backward
        ctx.save_for_backward(a, b)

        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        a, b = a.to(torch.uint8), b.to(torch.uint8)

        # prepare input & output tensors 
        a_cont, b_cont = a.contiguous(), b.contiguous()
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        c = a.new_zeros(M, N).to(torch.uint32)

        # call gemm forward function
        global lut_appmult
        approx_ops.approx_gemm_forward(a_cont, b_cont, c, lut_appmult)

        # convert output dtype 
        c = c.to(torch.float32)

        return c

    @staticmethod
    def backward(ctx, g_c):
        """gemm function backward.
        Args:
            g_c (torch.Tensor): [M, N], float32
        
        Returns:
            g_a (torch.Tensor): [M, K], float32
            g_b (torch.Tensor): [K, N], float32
        """
        # check input dtype
        assert g_c.dtype == torch.float32

        # get saved tensors
        a, b, = ctx.saved_tensors

        # call straight-through estimator
        g_a = torch.matmul(g_c.contiguous(), b.t().contiguous())
        g_b = torch.matmul(a.t().contiguous(), g_c.contiguous())

        return g_a, g_b

approx_gemm_baseline_op = ApproxGemmBaseline.apply


class ApproxBmm(Function):
    @staticmethod
    def forward(ctx, a, b):
        """bmm function forward.
        Args:
            a (torch.Tensor): [BATCH_SIZE, M, K]
            b (torch.Tensor): [BATCH_SIZE, K, N]
        
        Returns:
            c (torch.Tensor): [BATCH_SIZE, M, N]
        """
        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        a, b = a.to(torch.uint8), b.to(torch.uint8)

        # prepare input & output tensors 
        a_cont, b_cont = a.contiguous(), b.contiguous()
        BATCH_SIZE, M, K, N = a.shape[0], a.shape[1], a.shape[2], b.shape[2]
        c = a.new_zeros(BATCH_SIZE, M, N).to(torch.uint32)

        # call gemm forward function
        global lut_appmult
        approx_ops.approx_bmm_forward(a_cont, b_cont, c, lut_appmult)

        # convert output dtype 
        c = c.to(torch.float32)

        # save for backward
        ctx.save_for_backward(a_cont, b_cont)
        ctx.ori_shape = (BATCH_SIZE, M, K, N)

        return c

    @staticmethod
    def backward(ctx, g_c):
        """bmm function backward.
        Args:
            g_c (torch.Tensor): [BATCH_SIZE, M, N], float32
        
        Returns:
            g_a (torch.Tensor): [BATCH_SIZE, M, K], float32
            g_b (torch.Tensor): [BATCH_SIZE, K, N], float32
        """
        # check input dtype
        assert g_c.dtype == torch.float32

        # get saved tensors
        a_cont, b_cont, = ctx.saved_tensors
        BATCH_SIZE, M, K, N = ctx.ori_shape

        # prepare output tensors
        g_a, g_b = g_c.new_zeros(BATCH_SIZE, M, K), g_c.new_zeros(BATCH_SIZE, K, N)

        # call bmm backward function
        global lut_grad_a, lut_grad_b
        approx_ops.approx_bmm_backward(g_c.contiguous(), a_cont, b_cont, g_a, g_b, lut_grad_a, lut_grad_b)

        return g_a, g_b

approx_bmm_op = ApproxBmm.apply


class AccConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        """conv2d function forward.
        Args:
            input (torch.Tensor): [batch, inC, inH, inW]
            weight (torch.Tensor): [outC, inC, kernelH, kernelW]
            stride (int)
            padding (int)
            dilation (int)
            
        Returns:
            output (torch.Tensor): [batch, outC, outH, outW]
        """
        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        # input, weight = input.to(torch.uint8), weight.to(torch.uint8)
        
        # prepare input & output tensors
        input_cont, weight_cont = input.contiguous(), weight.contiguous()
        batch, inC, inH, inW = input.shape
        outC, _, kernelH, kernelW = weight.shape
        assert(kernelH == kernelW)
        outH = (inH + 2 * padding - dilation * (kernelH - 1) - 1) // stride + 1
        outW = (inW + 2 * padding - dilation * (kernelH - 1) - 1) // stride + 1
        output = input.new_zeros(batch, outC, outH, outW).to(torch.float32)
        
        # call conv2d forward function
        # global lut_appmult
        # approx_ops.approx_conv2d_forward(input_cont, weight_cont, output, lut_appmult, stride, padding, dilation)
        # approx_ops.acc_conv2d_forward_fp32(input_cont, weight_cont, output, stride, padding, dilation)
        output = torch.nn.functional.conv2d(input_cont, weight_cont, None, stride, padding, dilation, 1)

        # convert output dtype
        # output = output.to(torch.float32)
        
        # save for backward
        ctx.save_for_backward(input_cont, weight_cont)
        ctx.parameters = (batch, inC, inH, inW, outC, kernelH, outH, outW, stride, padding, dilation)

        return output
    
    @staticmethod
    def backward(ctx, g_out):
        """conv2d function backward.
        Args:
            g_out (torch.Tensor): [batch, outC, outH, outW]
            
        Returns:
            g_input (torch.Tensor): [batch, inC, inH, inW]
            g_weight (torch.Tensor): [outC, inC, kernel, kernel]
        """
        input_cont, weight_cont, = ctx.saved_tensors
        batch, inC, inH, inW, outC, kernelH, outH, outW, stride, padding, dilation = ctx.parameters

        g_input = torch.nn.grad.conv2d_input(input_cont.shape, weight_cont, g_out, stride, padding, dilation)
        g_weight = torch.nn.grad.conv2d_weight(input_cont, weight_cont.shape, g_out, stride, padding, dilation)
        
        return g_input, g_weight, None, None, None

acc_conv2d_op = AccConv2d.apply


class ApproxConv2dLegacy0(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        """conv2d function forward.
        Args:
            input (torch.Tensor): [batch, inC, inH, inW]
            weight (torch.Tensor): [outC, inC, kernelH, kernelW]
            stride (int)
            padding (int)
            dilation (int)
            
        Returns:
            output (torch.Tensor): [batch, outC, outH, outW]
        """
        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        input, weight = input.to(torch.uint8), weight.to(torch.uint8)
        
        # prepare input & output tensors
        input_cont, weight_cont = input.contiguous(), weight.contiguous()
        batch, inC, inH, inW = input.shape
        outC, _, kernelH, kernelW = weight.shape
        assert(kernelH == kernelW)
        outH = (inH + 2 * padding - dilation * (kernelH - 1) - 1) // stride + 1
        outW = (inW + 2 * padding - dilation * (kernelH - 1) - 1) // stride + 1
        output = input.new_zeros(batch, outC, outH, outW).to(torch.uint32)
        
        # call conv2d forward function
        global lut_appmult
        approx_ops.approx_conv2d_forward(input_cont, weight_cont, output, lut_appmult, stride, padding, dilation)

        # convert output dtype
        output = output.to(torch.float32)
        
        # save for backward
        ctx.save_for_backward(input_cont, weight_cont)
        ctx.parameters = (batch, inC, inH, inW, outC, kernelH, outH, outW, stride, padding, dilation)

        return output
    
    @staticmethod
    def backward(ctx, g_out):
        """conv2d function backward.
        Args:
            g_out (torch.Tensor): [batch, outC, outH, outW]
            
        Returns:
            g_input (torch.Tensor): [batch, inC, inH, inW]
            g_weight (torch.Tensor): [outC, inC, kernel, kernel]
            None, None, None
        """
        input_cont, weight_cont, = ctx.saved_tensors
        batch, inC, inH, inW, outC, kernelH, outH, outW, stride, padding, dilation = ctx.parameters

        g_input = g_out.new_zeros(batch, inC, inH, inW)
        g_weight = g_out.new_zeros(outC, inC, kernelH, kernelH)
        approx_ops.acc_conv2d_backward_fp32(input_cont.to(torch.float32), weight_cont.to(torch.float32), g_out.contiguous(), stride, padding, dilation, g_input, g_weight)

        # g_input = torch.nn.grad.conv2d_input(input_cont.shape, weight_cont.float(), g_out, stride, padding, dilation)
        g_input = F.conv_transpose2d(g_out, weight_cont.to(torch.float32), stride=stride, padding=padding)

        # g_weight = torch.nn.grad.conv2d_weight(input_cont.float(), weight_cont.shape, g_out, stride, padding, dilation)
        g_weight = F.conv2d(input_cont.to(torch.float32).transpose(0, 1), g_out.transpose(0, 1), stride=dilation, padding=padding).transpose(0, 1)
        
        return g_input, g_weight, None, None, None

        
class ApproxConv2dLegacy1(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        """conv2d function forward.
        Args:
            input (torch.Tensor): [batch_size, in_channels, in_height, in_width]
            weight (torch.Tensor): [out_channels, in_channels, kernel_height, kernel_width]
            stride (int)
            padding (int)
            dilation (int)
            
        Returns:
            output (torch.Tensor): [batch_size, out_channels, out_height, out_width]
        """
        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        # input, weight = input.to(torch.uint8), weight.to(torch.uint8)
        
        # prepare input & output tensors
        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
        out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

        # pad the input (zero padding)
        input_padded = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)        

        # Use unfold to create sliding windows
        input_unfolded = F.unfold(input_padded, kernel_size=(kernel_height, kernel_width), dilation=(dilation, dilation), stride=(stride, stride))

        # Reshape for matrix multiplication
        input_unfolded = input_unfolded.transpose(1, 2)  # Shape: (batch_size, num_windows, in_channels * kernel_height * kernel_width)
        input_unfolded = input_unfolded.reshape(-1, in_channels * kernel_height * kernel_width)  # Shape: (batch_size * num_windows, in_channels * kernel_height * kernel_width)
        weight_reshaped = weight.view(out_channels, -1).t()  # Shape: (in_channels * kernel_height * kernel_width, out_channels)

        # Perform the matrix multiplication using torch.mm
        # output = torch.mm(input_unfolded, weight_reshaped)  # Shape: (batch_size * num_windows, out_channels)
        output = approx_gemm_op(input_unfolded, weight_reshaped)

        # Reshape the output to (batch_size, out_channels, out_height, out_width)
        output = output.view(batch_size, out_height * out_width, out_channels).transpose(1, 2)  # Shape: (batch_size, out_channels, num_windows)
        output = output.view(batch_size, out_channels, out_height, out_width)  # Shape: (batch_size, out_channels, out_height, out_width)

        # # Reshape for matrix multiplication
        # input_unfolded = input_unfolded.view(batch_size, in_channels * kernel_height * kernel_width, -1)
        # weight_reshaped = weight.view(out_channels, -1)
        # print(f"input_unfolded shape = {input_unfolded.shape}, input_shape = {input.shape}, weight_shape = {weight.shape}, output_shape = {(batch_size, out_channels, out_height, out_width)}, weight_reshaped shape = {weight_reshaped.shape}")

        # # Perform the matrix multiplication
        # output = torch.matmul(weight_reshaped, input_unfolded)
        # # output = input.new_zeros(batch_size, out_channels, out_height, out_width).to(torch.uint32)

        # # Reshape the output to (batch_size, out_channels, out_height, out_width)
        # output = output.view(batch_size, out_channels, out_height, out_width)

        return output
    
    @staticmethod
    def backward(ctx, g_out):
        """conv2d function backward.
        Args:
            g_out (torch.Tensor): [batch, outC, outH, outW]
            
        Returns:
            g_input (torch.Tensor): [batch, inC, inH, inW]
            g_weight (torch.Tensor): [outC, inC, kernel, kernel]
            None, None, None
        """
        return None, None, None, None, None


def approx_conv2d_op(input, weight, bias, stride, padding, dilation, groups):
    """conv2d function forward.
    Args:
        input (torch.Tensor): [batch_size, in_channels, in_height, in_width]
        weight (torch.Tensor): [out_channels, in_channels, kernel_height, kernel_width]
        bias (torch.Tensor): [out_channels]
        stride (int)
        padding (int)
        dilation (int)
        groups (int)
        
    Returns:
        output (torch.Tensor): [batch_size, out_channels, out_height, out_width]
    """
    # check
    assert bias is None, 'bias is not supported in approximate convolution'
    assert groups == 1, 'groups is not supported in approximate convolution'
    stride, padding, dilation = stride[0], padding[0], dilation[0]

    # prepare input & output tensors
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

    # pad the input (zero padding)
    input_padded = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)        

    # Use unfold to create sliding windows
    input_unfolded = F.unfold(input_padded, kernel_size=(kernel_height, kernel_width), dilation=(dilation, dilation), stride=(stride, stride))

    # Reshape for matrix multiplication
    input_unfolded = input_unfolded.transpose(1, 2)  # Shape: (batch_size, num_windows, in_channels * kernel_height * kernel_width)
    input_unfolded = input_unfolded.reshape(-1, in_channels * kernel_height * kernel_width)  # Shape: (batch_size * num_windows, in_channels * kernel_height * kernel_width)
    weight_reshaped = weight.view(out_channels, -1).t()  # Shape: (in_channels * kernel_height * kernel_width, out_channels)

    # Perform the matrix multiplication
    output = approx_gemm_op(input_unfolded, weight_reshaped) # Shape: (batch_size * num_windows, out_channels)

    # Reshape the output to (batch_size, out_channels, out_height, out_width)
    output = output.view(batch_size, out_height * out_width, out_channels).transpose(1, 2)  # Shape: (batch_size, out_channels, num_windows)
    output = output.view(batch_size, out_channels, out_height, out_width)  # Shape: (batch_size, out_channels, out_height, out_width)

    return output

    
def approx_conv2d_baseline_op(input, weight, bias, stride, padding, dilation, groups):
    """conv2d function forward.
    Args:
        input (torch.Tensor): [batch_size, in_channels, in_height, in_width]
        weight (torch.Tensor): [out_channels, in_channels, kernel_height, kernel_width]
        bias (torch.Tensor): [out_channels]
        stride (int)
        padding (int)
        dilation (int)
        groups (int)
        
    Returns:
        output (torch.Tensor): [batch_size, out_channels, out_height, out_width]
    """
    # check
    assert bias is None, 'bias is not supported in approximate convolution'
    assert groups == 1, 'groups is not supported in approximate convolution'
    stride, padding, dilation = stride[0], padding[0], dilation[0]

    # prepare input & output tensors
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

    # pad the input (zero padding)
    input_padded = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)        

    # Use unfold to create sliding windows
    input_unfolded = F.unfold(input_padded, kernel_size=(kernel_height, kernel_width), dilation=(dilation, dilation), stride=(stride, stride))

    # Reshape for matrix multiplication
    input_unfolded = input_unfolded.transpose(1, 2)  # Shape: (batch_size, num_windows, in_channels * kernel_height * kernel_width)
    input_unfolded = input_unfolded.reshape(-1, in_channels * kernel_height * kernel_width)  # Shape: (batch_size * num_windows, in_channels * kernel_height * kernel_width)
    weight_reshaped = weight.view(out_channels, -1).t()  # Shape: (in_channels * kernel_height * kernel_width, out_channels)

    # Perform the matrix multiplication
    output = approx_gemm_baseline_op(input_unfolded, weight_reshaped) # Shape: (batch_size * num_windows, out_channels)

    # Reshape the output to (batch_size, out_channels, out_height, out_width)
    output = output.view(batch_size, out_height * out_width, out_channels).transpose(1, 2)  # Shape: (batch_size, out_channels, num_windows)
    output = output.view(batch_size, out_channels, out_height, out_width)  # Shape: (batch_size, out_channels, out_height, out_width)

    return output