#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

#define NEG_INF (-INFINITY)

// ============================================================================
// Baseline: three kernels, materializes S = QK^T and P = softmax(S) in global
// memory. O(N^2) extra memory and lots of HBM traffic.
// ============================================================================

// S[bh,i,j] = scale * dot(Q[bh,i,:], K[bh,j,:])
__global__ void qk_kernel(const float* Q, const float* K, float* S,
                          int N, int D, float scale, bool causal) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int bh = blockIdx.z;
  if (i >= N || j >= N) return;
  const float* q = Q + ((size_t)bh * N + i) * D;
  const float* k = K + ((size_t)bh * N + j) * D;
  float acc = 0.f;
  for (int d = 0; d < D; ++d) acc += q[d] * k[d];
  S[((size_t)bh * N + i) * N + j] = (causal && j > i) ? NEG_INF : acc * scale;
}

// Row-wise softmax, one block per row (in place).
constexpr int SM_THREADS = 256;
__global__ void softmax_kernel(float* S, int N) {
  __shared__ float red[SM_THREADS];
  size_t row = (size_t)blockIdx.y * N + blockIdx.x;  // bh * N + i
  float* s = S + row * N;
  int t = threadIdx.x;

  float m = NEG_INF;
  for (int j = t; j < N; j += SM_THREADS) m = fmaxf(m, s[j]);
  red[t] = m;
  __syncthreads();
  for (int off = SM_THREADS / 2; off > 0; off >>= 1) {
    if (t < off) red[t] = fmaxf(red[t], red[t + off]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();

  float sum = 0.f;
  for (int j = t; j < N; j += SM_THREADS) {
    float e = expf(s[j] - m);
    s[j] = e;
    sum += e;
  }
  red[t] = sum;
  __syncthreads();
  for (int off = SM_THREADS / 2; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    __syncthreads();
  }
  float inv = 1.f / red[0];
  for (int j = t; j < N; j += SM_THREADS) s[j] *= inv;
}

// O[bh,i,d] = sum_j P[bh,i,j] * V[bh,j,d]
__global__ void pv_kernel(const float* P, const float* V, float* O, int N, int D) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int bh = blockIdx.z;
  if (i >= N || d >= D) return;
  const float* p = P + ((size_t)bh * N + i) * N;
  const float* v = V + (size_t)bh * N * D;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) acc += p[j] * v[(size_t)j * D + d];
  O[((size_t)bh * N + i) * D + d] = acc;
}

at::Tensor attention_naive_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal) {
  int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
  int BH = B * H;
  auto S = at::empty({B, H, N, N}, q.options());
  auto O = at::empty_like(q);
  float scale = 1.f / sqrtf((float)D);
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 blk(16, 16);
  qk_kernel<<<dim3((N + 15) / 16, (N + 15) / 16, BH), blk, 0, stream>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), S.data_ptr<float>(), N, D, scale, causal);
  softmax_kernel<<<dim3(N, BH), SM_THREADS, 0, stream>>>(S.data_ptr<float>(), N);
  pv_kernel<<<dim3((D + 15) / 16, (N + 15) / 16, BH), blk, 0, stream>>>(
      S.data_ptr<float>(), v.data_ptr<float>(), O.data_ptr<float>(), N, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return O;
}

// ============================================================================
// Tiled / FlashAttention-style kernel.
//  - Each thread owns one query row (q and the O accumulator live in registers).
//  - K/V are streamed through shared memory in tiles of BC rows.
//  - Online softmax keeps a running max (m) and denominator (l), so the N x N
//    score matrix is never written to global memory: O(N) extra memory.
//  - Causal mode skips key tiles entirely above the diagonal.
// ============================================================================
template <int D, int BR, int BC>
__global__ void flash_fwd_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                 const float* __restrict__ V, float* __restrict__ O,
                                 int N, float scale, bool causal) {
  __shared__ float Ks[BC * D];
  __shared__ float Vs[BC * D];

  const int bh = blockIdx.y;
  const int q_idx = blockIdx.x * BR + threadIdx.x;
  const bool valid = q_idx < N;
  const size_t base = (size_t)bh * N * D;
  const float* Kb = K + base;
  const float* Vb = V + base;

  float q[D], o[D];
  #pragma unroll
  for (int d = 0; d < D; ++d) {
    q[d] = valid ? Q[base + (size_t)q_idx * D + d] * scale : 0.f;
    o[d] = 0.f;
  }
  float m = NEG_INF, l = 0.f;

  // Block-uniform loop bound (all threads must hit every __syncthreads).
  int kv_end = N;
  if (causal) kv_end = min(N, (blockIdx.x + 1) * BR);

  for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
    // Cooperative, coalesced load of the K/V tile into shared memory.
    for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
      int r = idx / D, c = idx % D, kr = kv0 + r;
      Ks[idx] = (kr < N) ? Kb[(size_t)kr * D + c] : 0.f;
      Vs[idx] = (kr < N) ? Vb[(size_t)kr * D + c] : 0.f;
    }
    __syncthreads();

    if (valid) {
      float s[BC];
      float m_tile = NEG_INF;
      #pragma unroll
      for (int j = 0; j < BC; ++j) {
        int kj = kv0 + j;
        float acc = NEG_INF;
        if (kj < N && !(causal && kj > q_idx)) {
          acc = 0.f;
          #pragma unroll
          for (int d = 0; d < D; ++d) acc += q[d] * Ks[j * D + d];  // smem broadcast
        }
        s[j] = acc;
        m_tile = fmaxf(m_tile, acc);
      }

      if (m_tile > NEG_INF) {
        float m_new = fmaxf(m, m_tile);
        float corr = expf(m - m_new);  // rescale old stats (0 on first tile)
        l *= corr;
        #pragma unroll
        for (int d = 0; d < D; ++d) o[d] *= corr;
        #pragma unroll
        for (int j = 0; j < BC; ++j) {
          float p = (s[j] == NEG_INF) ? 0.f : expf(s[j] - m_new);
          l += p;
          #pragma unroll
          for (int d = 0; d < D; ++d) o[d] += p * Vs[j * D + d];
        }
        m = m_new;
      }
    }
    __syncthreads();
  }

  if (valid) {
    float inv = 1.f / l;
    #pragma unroll
    for (int d = 0; d < D; ++d) O[base + (size_t)q_idx * D + d] = o[d] * inv;
  }
}

template <int D>
static void launch_flash(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         at::Tensor& O, bool causal) {
  constexpr int BR = 64, BC = 32;
  int BH = q.size(0) * q.size(1), N = q.size(2);
  float scale = 1.f / sqrtf((float)D);
  dim3 grid((N + BR - 1) / BR, BH);
  flash_fwd_kernel<D, BR, BC><<<grid, BR, 0, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), O.data_ptr<float>(),
      N, scale, causal);
}

at::Tensor attention_flash_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal) {
  auto O = at::empty_like(q);
  switch (q.size(3)) {
    case 32:  launch_flash<32>(q, k, v, O, causal); break;
    case 64:  launch_flash<64>(q, k, v, O, causal); break;
    case 128: launch_flash<128>(q, k, v, O, causal); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return O;
}
