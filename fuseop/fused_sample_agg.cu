#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 128;

__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

__device__ __forceinline__ uint64_t next_random(uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 2685821657736338717ULL;
}

__device__ __forceinline__ uint32_t bounded_random(uint64_t& state,
                                                   uint32_t bound) {
  return static_cast<uint32_t>(__umul64hi(next_random(state), bound));
}

__device__ __forceinline__ uint64_t make_seed(uint64_t base, uint64_t root,
                                              uint64_t hop,
                                              uint64_t position) {
  return splitmix64(base ^ splitmix64(root) ^ splitmix64(hop) ^
                    splitmix64(position));
}

__device__ void reservoir_sample(const int32_t* neighbors, int degree, int fanout,
                                 uint64_t seed, int32_t* output) {
  const int take = min(degree, fanout);
  for (int i = 0; i < take; ++i) {
    output[i] = neighbors[i];
  }
  uint64_t state = seed == 0 ? 0x9E3779B97F4A7C15ULL : seed;
  for (int i = take; i < degree; ++i) {
    const uint32_t selected = bounded_random(state, static_cast<uint32_t>(i + 1));
    if (selected < static_cast<uint32_t>(take)) {
      output[selected] = neighbors[i];
    }
  }
  for (int i = take; i < fanout; ++i) {
    output[i] = -1;
  }
}

__global__ void sample_agg_1hop_forward(
    const int32_t* __restrict__ rowptr, const int32_t* __restrict__ col,
    const float* __restrict__ x, int64_t feature_dim,
    const int32_t* __restrict__ frontier, int64_t batch_size, int fanout,
    uint64_t seed, float* __restrict__ out,
    int32_t* __restrict__ saved_samples, int32_t* __restrict__ saved_takes) {
  extern __shared__ int32_t shared_samples[];
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t batch_index =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  if (batch_index >= batch_size) {
    return;
  }

  int32_t* samples = shared_samples + warp * fanout;
  const int32_t root = frontier[batch_index];
  const int32_t begin = rowptr[root];
  const int32_t degree = rowptr[root + 1] - begin;
  const int take = min(degree, fanout);

  if (lane == 0) {
    reservoir_sample(col + begin, degree, fanout,
                     make_seed(seed, static_cast<uint64_t>(root), 0,
                               static_cast<uint64_t>(batch_index)),
                     samples);
    if (saved_takes != nullptr) {
      saved_takes[batch_index] = take;
    }
    if (saved_samples != nullptr) {
      for (int i = 0; i < fanout; ++i) {
        saved_samples[batch_index * fanout + i] = samples[i];
      }
    }
  }
  __syncwarp();

  for (int64_t feature = lane; feature < feature_dim; feature += kWarpSize) {
    float sum = 0.0F;
    for (int i = 0; i < take; ++i) {
      sum += x[static_cast<int64_t>(samples[i]) * feature_dim + feature];
    }
    out[batch_index * feature_dim + feature] =
        take > 0 ? sum / static_cast<float>(take) : 0.0F;
  }
}

__global__ void sample_agg_1hop_backward(
    const int32_t* __restrict__ samples, const int32_t* __restrict__ takes,
    const float* __restrict__ grad_out, int64_t batch_size,
    int64_t feature_dim, int fanout, float* __restrict__ grad_x) {
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t batch_index =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  if (batch_index >= batch_size) {
    return;
  }

  const int take = takes[batch_index];
  if (take == 0) {
    return;
  }
  for (int64_t feature = lane; feature < feature_dim; feature += kWarpSize) {
    const float contribution =
        grad_out[batch_index * feature_dim + feature] / static_cast<float>(take);
    for (int i = 0; i < take; ++i) {
      const int32_t neighbor = samples[batch_index * fanout + i];
      atomicAdd(&grad_x[static_cast<int64_t>(neighbor) * feature_dim + feature],
                contribution);
    }
  }
}

__global__ void sample_agg_2hop_forward(
    const int32_t* __restrict__ rowptr, const int32_t* __restrict__ col,
    const float* __restrict__ x, int64_t feature_dim,
    const int32_t* __restrict__ frontier, int64_t batch_size, int fanout1,
    int fanout2, uint64_t seed, float* __restrict__ out,
    int32_t* __restrict__ saved_samples1,
    int32_t* __restrict__ saved_samples2) {
  const int64_t batch_index = blockIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  extern __shared__ int32_t shared[];
  int32_t* samples1 = shared;
  int32_t* samples2 = shared + fanout1;
  const int32_t root = frontier[batch_index];

  if (threadIdx.x == 0) {
    const int32_t begin = rowptr[root];
    const int32_t degree = rowptr[root + 1] - begin;
    reservoir_sample(col + begin, degree, fanout1,
                     make_seed(seed, static_cast<uint64_t>(root), 0,
                               static_cast<uint64_t>(batch_index)),
                     samples1);

    for (int i = 0; i < fanout1; ++i) {
      const int32_t middle = samples1[i];
      int32_t* second_hop = samples2 + i * fanout2;
      if (middle < 0) {
        for (int j = 0; j < fanout2; ++j) {
          second_hop[j] = -1;
        }
      } else {
        const int32_t middle_begin = rowptr[middle];
        const int32_t middle_degree = rowptr[middle + 1] - middle_begin;
        reservoir_sample(col + middle_begin, middle_degree, fanout2,
                         make_seed(seed, static_cast<uint64_t>(root), 1,
                                   static_cast<uint64_t>(i)),
                         second_hop);
      }
    }

    if (saved_samples1 != nullptr) {
      for (int i = 0; i < fanout1; ++i) {
        saved_samples1[batch_index * fanout1 + i] = samples1[i];
      }
    }
    if (saved_samples2 != nullptr) {
      for (int i = 0; i < fanout1 * fanout2; ++i) {
        saved_samples2[batch_index * fanout1 * fanout2 + i] = samples2[i];
      }
    }
  }
  __syncthreads();

  for (int64_t feature = threadIdx.x; feature < feature_dim;
       feature += blockDim.x) {
    float outer_sum = 0.0F;
    int nonempty_middle_nodes = 0;
    for (int i = 0; i < fanout1; ++i) {
      float inner_sum = 0.0F;
      int valid_neighbors = 0;
      for (int j = 0; j < fanout2; ++j) {
        const int32_t neighbor = samples2[i * fanout2 + j];
        if (neighbor >= 0) {
          inner_sum +=
              x[static_cast<int64_t>(neighbor) * feature_dim + feature];
          ++valid_neighbors;
        }
      }
      if (valid_neighbors > 0) {
        outer_sum += inner_sum / static_cast<float>(valid_neighbors);
        ++nonempty_middle_nodes;
      }
    }
    out[batch_index * feature_dim + feature] =
        nonempty_middle_nodes > 0
            ? outer_sum / static_cast<float>(nonempty_middle_nodes)
            : 0.0F;
  }
}

__global__ void sample_agg_2hop_backward(
    const float* __restrict__ grad_out,
    const int32_t* __restrict__ samples1,
    const int32_t* __restrict__ samples2, int64_t batch_size,
    int64_t feature_dim, int fanout1, int fanout2,
    float* __restrict__ grad_x) {
  const int64_t batch_index = blockIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const int64_t first_base = batch_index * fanout1;
  const int64_t second_base = first_base * fanout2;
  int nonempty_middle_nodes = 0;
  for (int i = 0; i < fanout1; ++i) {
    if (samples1[first_base + i] < 0) {
      continue;
    }
    for (int j = 0; j < fanout2; ++j) {
      if (samples2[second_base + i * fanout2 + j] >= 0) {
        ++nonempty_middle_nodes;
        break;
      }
    }
  }
  if (nonempty_middle_nodes == 0) {
    return;
  }

  for (int64_t feature = threadIdx.x; feature < feature_dim;
       feature += blockDim.x) {
    const float outer_grad =
        grad_out[batch_index * feature_dim + feature] /
        static_cast<float>(nonempty_middle_nodes);
    for (int i = 0; i < fanout1; ++i) {
      int valid_neighbors = 0;
      for (int j = 0; j < fanout2; ++j) {
        valid_neighbors +=
            samples2[second_base + i * fanout2 + j] >= 0 ? 1 : 0;
      }
      if (valid_neighbors == 0) {
        continue;
      }
      const float inner_grad =
          outer_grad / static_cast<float>(valid_neighbors);
      for (int j = 0; j < fanout2; ++j) {
        const int32_t neighbor = samples2[second_base + i * fanout2 + j];
        if (neighbor >= 0) {
          atomicAdd(
              &grad_x[static_cast<int64_t>(neighbor) * feature_dim + feature],
              inner_grad);
        }
      }
    }
  }
}

}  // namespace

void fused_sample_agg_cuda_forward(
    const int32_t* rowptr, const int32_t* col, const float* x,
    int64_t feature_dim, const int32_t* frontier, int64_t batch_size,
    int fanout, uint64_t seed, float* out, int32_t* samples, int32_t* takes,
    cudaStream_t stream) {
  const int warps_per_block = kThreads / kWarpSize;
  const int64_t blocks = (batch_size + warps_per_block - 1) / warps_per_block;
  const size_t shared_bytes =
      static_cast<size_t>(warps_per_block * fanout) * sizeof(int32_t);
  sample_agg_1hop_forward<<<static_cast<unsigned int>(blocks), kThreads,
                             shared_bytes, stream>>>(
      rowptr, col, x, feature_dim, frontier, batch_size, fanout, seed, out,
      samples, takes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_sample_agg_cuda_backward(
    const int32_t* samples, const int32_t* takes, const float* grad_out,
    int64_t batch_size, int64_t feature_dim, int fanout, float* grad_x,
    cudaStream_t stream) {
  const int warps_per_block = kThreads / kWarpSize;
  const int64_t blocks = (batch_size + warps_per_block - 1) / warps_per_block;
  sample_agg_1hop_backward<<<static_cast<unsigned int>(blocks), kThreads, 0,
                              stream>>>(samples, takes, grad_out, batch_size,
                                       feature_dim, fanout, grad_x);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_sample_agg_2hop_cuda_forward(
    const int32_t* rowptr, const int32_t* col, const float* x,
    int64_t feature_dim, const int32_t* frontier, int64_t batch_size,
    int fanout1, int fanout2, uint64_t seed, float* out, int32_t* samples1,
    int32_t* samples2, cudaStream_t stream) {
  const size_t shared_bytes =
      static_cast<size_t>(fanout1 + fanout1 * fanout2) * sizeof(int32_t);
  sample_agg_2hop_forward<<<static_cast<unsigned int>(batch_size), kThreads,
                             shared_bytes, stream>>>(
      rowptr, col, x, feature_dim, frontier, batch_size, fanout1, fanout2,
      seed, out, samples1, samples2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_sample_agg_2hop_cuda_backward(
    const float* grad_out, const int32_t* samples1, const int32_t* samples2,
    int64_t batch_size, int64_t feature_dim, int fanout1, int fanout2,
    float* grad_x, cudaStream_t stream) {
  sample_agg_2hop_backward<<<static_cast<unsigned int>(batch_size), kThreads, 0,
                              stream>>>(grad_out, samples1, samples2,
                                       batch_size, feature_dim, fanout1,
                                       fanout2, grad_x);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
