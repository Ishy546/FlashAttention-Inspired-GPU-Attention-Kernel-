// C++ binding layer: input validation + pybind11 registration.
#include <torch/extension.h>
 
at::Tensor attention_naive_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal);
at::Tensor attention_flash_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal);
 
static void check(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  for (const auto& t : {q, k, v}) {
    TORCH_CHECK(t.is_cuda(), "tensors must be CUDA tensors");
    TORCH_CHECK(t.scalar_type() == at::kFloat, "only float32 is supported");
    TORCH_CHECK(t.dim() == 4, "expected shape [B, H, N, D]");
    TORCH_CHECK(t.is_contiguous(), "tensors must be contiguous");
  }
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v must have the same shape");
}
 
at::Tensor attention_naive(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal) {
  check(q, k, v);
  return attention_naive_cuda(q, k, v, causal);
}
 
at::Tensor attention_flash(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal) {
  check(q, k, v);
  const auto d = q.size(3);
  TORCH_CHECK(d == 32 || d == 64 || d == 128, "head dim must be 32, 64, or 128");
  return attention_flash_cuda(q, k, v, causal);
}
 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_naive", &attention_naive, "Baseline attention (materializes N x N scores)");
  m.def("attention_flash", &attention_flash, "Tiled attention with online softmax");
}