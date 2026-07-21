#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace {

constexpr int64_t kMaxFanout1Hop = 1024;
constexpr int64_t kMax2HopSharedEntries = 12000;

void check_graph_inputs(const at::Tensor& rowptr,
                        const at::Tensor& col,
                        const at::Tensor& x,
                        const at::Tensor& frontier) {
  TORCH_CHECK(rowptr.is_cuda() && col.is_cuda() && x.is_cuda() && frontier.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(rowptr.device() == col.device() && rowptr.device() == x.device() &&
                  rowptr.device() == frontier.device(),
              "all inputs must be on the same CUDA device");
  TORCH_CHECK(rowptr.scalar_type() == at::kInt && col.scalar_type() == at::kInt &&
                  frontier.scalar_type() == at::kInt,
              "rowptr, col, and frontier must be int32");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(rowptr.dim() == 1 && col.dim() == 1 && x.dim() == 2 &&
                  frontier.dim() == 1,
              "expected rowptr[N+1], col[E], x[N,D], and frontier[B]");
  TORCH_CHECK(rowptr.is_contiguous() && col.is_contiguous() && x.is_contiguous() &&
                  frontier.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(rowptr.numel() == x.size(0) + 1, "rowptr length must equal N+1");
}

void check_fanout_1hop(int64_t fanout) {
  TORCH_CHECK(fanout >= 0, "fanout must be non-negative");
  TORCH_CHECK(fanout <= kMaxFanout1Hop, "fanout exceeds the supported maximum of ",
              kMaxFanout1Hop);
}

void check_fanout_2hop(int64_t fanout1, int64_t fanout2) {
  TORCH_CHECK(fanout1 >= 0 && fanout2 >= 0, "fanouts must be non-negative");
  TORCH_CHECK(fanout1 == 0 ||
                  fanout2 <=
                      (std::numeric_limits<int64_t>::max() - fanout1) / fanout1,
              "fanout pair is too large");
  const int64_t entries = fanout1 + fanout1 * fanout2;
  TORCH_CHECK(entries <= kMax2HopSharedEntries,
              "fanout pair requires too much shared memory; reduce fanout1 or fanout2");
}

}  // namespace

void fused_sample_agg_cuda_forward(const int32_t*, const int32_t*, const float*,
                                   int64_t, const int32_t*, int64_t, int,
                                   uint64_t, float*, int32_t*, int32_t*,
                                   cudaStream_t);

void fused_sample_agg_cuda_backward(const int32_t*, const int32_t*, const float*,
                                    int64_t, int64_t, int, float*, cudaStream_t);

void fused_sample_agg_2hop_cuda_forward(const int32_t*, const int32_t*, const float*,
                                        int64_t, const int32_t*, int64_t, int, int,
                                        uint64_t, float*, int32_t*, int32_t*,
                                        cudaStream_t);

void fused_sample_agg_2hop_cuda_backward(const float*, const int32_t*, const int32_t*,
                                         int64_t, int64_t, int, int, float*,
                                         cudaStream_t);

at::Tensor fused_sample_agg_forward(at::Tensor rowptr, at::Tensor col, at::Tensor x,
                                    at::Tensor frontier, int64_t fanout,
                                    uint64_t seed) {
  check_graph_inputs(rowptr, col, x, frontier);
  check_fanout_1hop(fanout);
  c10::cuda::CUDAGuard device_guard(x.device());
  const auto batch_size = frontier.size(0);
  const auto feature_dim = x.size(1);
  auto out = at::zeros({batch_size, feature_dim}, x.options());
  if (batch_size == 0 || fanout == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_sample_agg_cuda_forward(
      rowptr.data_ptr<int32_t>(), col.data_ptr<int32_t>(), x.data_ptr<float>(),
      feature_dim, frontier.data_ptr<int32_t>(), batch_size, static_cast<int>(fanout),
      seed, out.data_ptr<float>(), nullptr, nullptr, stream.stream());
  return out;
}

std::vector<at::Tensor> fused_sample_agg_forward_with_samples(
    at::Tensor rowptr, at::Tensor col, at::Tensor x, at::Tensor frontier,
    int64_t fanout, uint64_t seed) {
  check_graph_inputs(rowptr, col, x, frontier);
  check_fanout_1hop(fanout);
  c10::cuda::CUDAGuard device_guard(x.device());
  const auto batch_size = frontier.size(0);
  const auto feature_dim = x.size(1);
  auto out = at::zeros({batch_size, feature_dim}, x.options());
  auto samples = at::full({batch_size, fanout}, -1, rowptr.options());
  auto takes = at::zeros({batch_size}, rowptr.options());
  if (batch_size == 0 || fanout == 0) {
    return {out, samples, takes};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_sample_agg_cuda_forward(
      rowptr.data_ptr<int32_t>(), col.data_ptr<int32_t>(), x.data_ptr<float>(),
      feature_dim, frontier.data_ptr<int32_t>(), batch_size, static_cast<int>(fanout),
      seed, out.data_ptr<float>(), samples.data_ptr<int32_t>(),
      takes.data_ptr<int32_t>(), stream.stream());
  return {out, samples, takes};
}

at::Tensor fused_sample_agg_backward(at::Tensor samples, at::Tensor takes,
                                     at::Tensor grad_out, int64_t num_nodes) {
  TORCH_CHECK(samples.is_cuda() && takes.is_cuda() && grad_out.is_cuda(),
              "backward inputs must be CUDA tensors");
  TORCH_CHECK(samples.scalar_type() == at::kInt && takes.scalar_type() == at::kInt,
              "samples and takes must be int32");
  TORCH_CHECK(samples.device() == takes.device() &&
                  samples.device() == grad_out.device(),
              "backward inputs must be on the same CUDA device");
  TORCH_CHECK(grad_out.scalar_type() == at::kFloat && grad_out.dim() == 2,
              "grad_out must be a float32 matrix");
  TORCH_CHECK(samples.dim() == 2 && takes.dim() == 1 &&
                  samples.size(0) == grad_out.size(0) &&
                  takes.size(0) == grad_out.size(0),
              "saved sample shapes do not match grad_out");
  TORCH_CHECK(samples.is_contiguous() && takes.is_contiguous() &&
                  grad_out.is_contiguous(),
              "backward inputs must be contiguous");
  TORCH_CHECK(num_nodes >= 0, "num_nodes must be non-negative");
  c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_x = at::zeros({num_nodes, grad_out.size(1)}, grad_out.options());
  if (grad_out.size(0) == 0 || samples.size(1) == 0) {
    return grad_x;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_sample_agg_cuda_backward(
      samples.data_ptr<int32_t>(), takes.data_ptr<int32_t>(),
      grad_out.data_ptr<float>(), grad_out.size(0), grad_out.size(1),
      samples.size(1), grad_x.data_ptr<float>(), stream.stream());
  return grad_x;
}

at::Tensor fused_sample_agg_2hop_forward(at::Tensor rowptr, at::Tensor col,
                                         at::Tensor x, at::Tensor frontier,
                                         int64_t fanout1, int64_t fanout2,
                                         uint64_t seed) {
  check_graph_inputs(rowptr, col, x, frontier);
  check_fanout_2hop(fanout1, fanout2);
  c10::cuda::CUDAGuard device_guard(x.device());
  const auto batch_size = frontier.size(0);
  const auto feature_dim = x.size(1);
  auto out = at::zeros({batch_size, feature_dim}, x.options());
  if (batch_size == 0 || fanout1 == 0 || fanout2 == 0) {
    return out;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_sample_agg_2hop_cuda_forward(
      rowptr.data_ptr<int32_t>(), col.data_ptr<int32_t>(), x.data_ptr<float>(),
      feature_dim, frontier.data_ptr<int32_t>(), batch_size,
      static_cast<int>(fanout1), static_cast<int>(fanout2), seed,
      out.data_ptr<float>(), nullptr, nullptr, stream.stream());
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_sample_agg_2hop_forward_with_samples(
    at::Tensor rowptr, at::Tensor col, at::Tensor x, at::Tensor frontier,
    int64_t fanout1, int64_t fanout2, uint64_t seed) {
  check_graph_inputs(rowptr, col, x, frontier);
  check_fanout_2hop(fanout1, fanout2);
  c10::cuda::CUDAGuard device_guard(x.device());
  const auto batch_size = frontier.size(0);
  const auto feature_dim = x.size(1);
  auto out = at::zeros({batch_size, feature_dim}, x.options());
  auto samples1 = at::full({batch_size, fanout1}, -1, col.options());
  auto samples2 = at::full({batch_size, fanout1, fanout2}, -1, col.options());
  if (batch_size == 0 || fanout1 == 0 || fanout2 == 0) {
    return {out, samples1, samples2};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_sample_agg_2hop_cuda_forward(
      rowptr.data_ptr<int32_t>(), col.data_ptr<int32_t>(), x.data_ptr<float>(),
      feature_dim, frontier.data_ptr<int32_t>(), batch_size,
      static_cast<int>(fanout1), static_cast<int>(fanout2), seed,
      out.data_ptr<float>(), samples1.data_ptr<int32_t>(),
      samples2.data_ptr<int32_t>(), stream.stream());
  return {out, samples1, samples2};
}

at::Tensor fused_sample_agg_2hop_backward(at::Tensor grad_out,
                                          at::Tensor samples1,
                                          at::Tensor samples2,
                                          int64_t num_nodes) {
  TORCH_CHECK(grad_out.is_cuda() && samples1.is_cuda() && samples2.is_cuda(),
              "backward inputs must be CUDA tensors");
  TORCH_CHECK(grad_out.scalar_type() == at::kFloat &&
                  samples1.scalar_type() == at::kInt &&
                  samples2.scalar_type() == at::kInt,
              "invalid backward input dtypes");
  TORCH_CHECK(grad_out.dim() == 2 && samples1.dim() == 2 &&
                  samples2.dim() == 3 &&
                  samples1.size(0) == grad_out.size(0) &&
                  samples2.size(0) == grad_out.size(0) &&
                  samples2.size(1) == samples1.size(1),
              "saved sample shapes do not match grad_out");
  TORCH_CHECK(grad_out.device() == samples1.device() &&
                  grad_out.device() == samples2.device(),
              "backward inputs must be on the same CUDA device");
  TORCH_CHECK(grad_out.is_contiguous() && samples1.is_contiguous() &&
                  samples2.is_contiguous(),
              "backward inputs must be contiguous");
  TORCH_CHECK(num_nodes >= 0, "num_nodes must be non-negative");
  c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto grad_x = at::zeros({num_nodes, grad_out.size(1)}, grad_out.options());
  if (grad_out.size(0) == 0 || samples1.size(1) == 0 || samples2.size(2) == 0) {
    return grad_x;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  fused_sample_agg_2hop_cuda_backward(
      grad_out.data_ptr<float>(), samples1.data_ptr<int32_t>(),
      samples2.data_ptr<int32_t>(), grad_out.size(0), grad_out.size(1),
      samples1.size(1), samples2.size(2), grad_x.data_ptr<float>(),
      stream.stream());
  return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_sample_agg_forward", &fused_sample_agg_forward);
  module.def("fused_sample_agg_forward_with_samples",
             &fused_sample_agg_forward_with_samples);
  module.def("fused_sample_agg_backward", &fused_sample_agg_backward);
  module.def("fused_sample_agg_2hop_forward", &fused_sample_agg_2hop_forward);
  module.def("fused_sample_agg_2hop_forward_with_samples",
             &fused_sample_agg_2hop_forward_with_samples);
  module.def("fused_sample_agg_2hop_backward",
             &fused_sample_agg_2hop_backward);
}
