#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>


void approx_gemm_forward_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &c_tensor, const at::Tensor &lut_appmult_tensor);
void acc_gemm_forward_fp32_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &c_tensor);
void approx_gemm_backward_gpu(const at::Tensor &gc_tensor, const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &ga_tensor, at::Tensor &gb_tensor, const at::Tensor &lut_grad_a_tensor, const at::Tensor &lut_grad_b_tensor);
// void approx_bmm_forward_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &c_tensor, const at::Tensor &lut_appmult_tensor);
// void approx_bmm_backward_gpu(const at::Tensor &gc_tensor, const at::Tensor &a_tensor, const at::Tensor &b_tensor, at::Tensor &ga_tensor, at::Tensor &gb_tensor, const at::Tensor &lut_grad_a_tensor, const at::Tensor &lut_grad_b_tensor);
// void approx_conv2d_forward_gpu(const at::Tensor &input_tensor, const at::Tensor &weight_tensor, at::Tensor &output_tensor, const at::Tensor &lut_appmult_tensor, const int stride, const int padding, const int dilation);
// void acc_conv2d_forward_fp32_gpu(const at::Tensor &input_tensor, const at::Tensor &weight_tensor, at::Tensor &output_tensor, const int stride, const int padding, const int dilation);
// void acc_conv2d_backward_fp32_gpu(const at::Tensor &input_tensor, const at::Tensor &weight_tensor, const at::Tensor &grad_output_tensor, const int stride, const int padding, const int dilation, const at::Tensor &grad_input_tensor, const at::Tensor &grad_weight_tensor);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("approx_gemm_forward", &approx_gemm_forward_gpu, "approximate gemm forward (CUDA)");
    m.def("acc_gemm_forward_fp32", &acc_gemm_forward_fp32_gpu, "accurate gemm forward FP32 (CUDA)");
    m.def("approx_gemm_backward", &approx_gemm_backward_gpu, "approximate gemm backward (CUDA)");
    // m.def("approx_bmm_forward", &approx_bmm_forward_gpu, "approximate bmm forward (CUDA)");
    // m.def("approx_bmm_backward", &approx_bmm_backward_gpu, "approximate bmm backward (CUDA)");
    // m.def("approx_conv2d_forward", &approx_conv2d_forward_gpu, "approximate conv2d forward (CUDA)");
    // m.def("acc_conv2d_forward_fp32", &acc_conv2d_forward_fp32_gpu, "accurate conv2d forward FP32 (CUDA)");
    // m.def("acc_conv2d_backward_fp32", &acc_conv2d_backward_fp32_gpu, "accurate conv2d backward FP32 (CUDA)");
}