/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#pragma once

#include <algorithm>
#include <curand_kernel.h>

#include <vector>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>

class LifMembrane;
class IzhMembrane;


static bool use_permutation = false;
static bool streams_initialized = false;
static std::vector<cudaStream_t> streams;
static std::vector<size_t> group_sizes, group_offsets;
static std::vector<int> group_keys;
static size_t num_groups = 0;
static std::vector<size_t> perm, inv_perm;


__global__ void init_rand_kernel(size_t n, curandState *states, uint64_t seed);

__global__ void lif_adapt_kernel(const float *d_I_t,                                            // inputs
                                 uint8_t *d_S_t, float *d_V_t, float *d_U_t, uint8_t *d_T_t,    // outputs
                                 // global parameters
                                 size_t n, float dt,
                                 // individual parameters
                                 const float *d_reverse, const float *d_resist,
                                 const float *d_V_thresh, const float *d_V_rest, const float *d_adapt_amp,
                                 const uint8_t *d_refract, const float *d_alpha, const float *d_beta,
                                 const float *d_prob, curandState *rand_states);

__global__ void lif_refract_kernel(const float *d_I_t,                                  // inputs
                                   uint8_t *d_S_t, float *d_V_t, uint8_t *d_T_t,        // outputs
                                   // global parameters
                                   size_t n, float dt,
                                   // individual parameters
                                   const float *d_reverse, const float *d_resist,
                                   const float *d_V_thresh, const float *d_V_rest,
                                   const uint8_t *d_refract, const float *d_alpha,
                                   const float* d_prob, curandState *rand_states);

__global__ void lif_kernel(const float *d_I_t,                  // inputs
                           uint8_t *d_S_t, float *d_V_t,        // outputs
                           // global parameters
                           size_t n, float dt,
                           // individual parameters
                           const float *d_reverse, const float *d_resist,
                           const float *d_V_thresh, const float *d_V_rest, const float *d_alpha,
                           const float *d_prob, curandState *rand_states);

__global__ void izh_kernel(const float *d_I_t,                  // inputs
                           uint8_t *d_S_t, float *d_V_t, float *d_U_t,        // outputs
                           // global parameters
                           size_t n, float dt,
                           // individual parameters
                           const float *d_a, const float *d_b, const float *d_c, const float *d_d,
                           const float *d_prob, curandState *rand_states);

void init_cuda_membranes(const std::vector<LifMembrane*>& membranes,
                         float*& d_I_t, uint8_t*& d_S_t, float*& d_V_t, float*& d_U_t, uint8_t*& d_T_t,
                         float& dt,
                         float*& d_reverse, float*& d_resist, float*& d_V_thresh,
                         float*& d_V_rest, float*& d_adapt_amp, uint8_t*& d_refract,
                         float*& d_alpha, float*& d_beta,
                         float*& d_prob, curandState*& d_rand_states, uint64_t custom_seed = static_cast<uint64_t>(-1));

void init_cuda_membranes(const std::vector<IzhMembrane *> &membranes,
                         float *&d_I_t, uint8_t *&d_S_t, float *&d_V_t, float *&d_U_t,
                         float &dt, float *&d_a, float *&d_b, float *&d_c, float *&d_d,
                         float *&d_prob, curandState*& d_rand_states, uint64_t custom_seed);

void update_cuda_lif_membranes(const std::vector<float>& h_I_t,                             // inputs
                               std::vector<uint8_t>& h_S_t, std::vector<float>& h_V_t);

void update_cuda_lif_membranes(const std::vector<float>& h_I_t,                             // inputs
                               std::vector<uint8_t>& h_S_t);

void update_cuda_izh_membranes(const std::vector<float>& h_I_t,
                               std::vector<uint8_t>& h_S_t, std::vector<float>& h_V_t);

void update_cuda_izh_membranes(const std::vector<float>& h_I_t,
                               std::vector<uint8_t>& h_S_t);




// Execution
template <size_t I, size_t N, typename Tuple, typename... Args>
void dispatch(int key, const Tuple& t, Args&&... args) {
    if (key == static_cast<int>(I)) {
        std::get<I>(t)(args...);
        return;
    }
    if constexpr (I + 1 < N) {
        dispatch<I + 1, N>(key, t, std::forward<Args>(args)...);
    } else {
        std::cerr << "Invalid key: " << key << std::endl;
    }
}
template <typename Params, typename... Launchers>
void exec_kernels(std::tuple<Launchers...> launchers_tuple,
                  const std::vector<size_t>& group_sizes,
                  const std::vector<size_t>& group_offsets,
                  const std::vector<int>& group_keys,
                  const Params& params) {
    const size_t num_groups = group_keys.size();
    if (num_groups == 0) return;
    if (streams.size() != num_groups) {
        std::cerr << "Error: Stream count mismatch (init=" << streams.size() << ", groups=" << num_groups << ")" << std::endl;
        return;
    }

    // Compute max_key for validation
    constexpr size_t num_launchers = sizeof...(Launchers);
    const int max_key = *std::ranges::max_element(group_keys);
    if (static_cast<size_t>(max_key) >= num_launchers) {
        std::cerr << "Error: Key " << max_key << " exceeds number of kernels (" << num_launchers << ")." << std::endl;
        return;
    }

    size_t offset = 0;
    for (size_t g = 0; g < num_groups; ++g) {
        constexpr size_t threads = 256;
        const size_t sz = group_sizes[g];
        if (sz == 0) continue;
        const size_t blocks = (sz + threads - 1) / threads;
        const int key = group_keys[g];
        dispatch<0, num_launchers>(key, launchers_tuple, blocks, threads, streams[g], params, offset, sz);
        offset += sz;
    }
}

// Populate
template <size_t... N, typename TupleType, typename... Targets>
void assign_to_targets(std::index_sequence<N...>, size_t i, const TupleType& vals, Targets&... targets) {
    ((targets[i] = std::get<N>(vals)), ...);
}
template <typename TupleType, typename... Targets>
void assign_to_targets(size_t i, const TupleType& vals, Targets&... targets) {
    assign_to_targets(std::make_index_sequence<sizeof...(Targets)>{}, i, vals, targets...);
}
template <typename Fetch, typename... Targets>
void populate(size_t n, Fetch&& fetch, Targets&... targets) {
    ((targets.resize(n)), ...);

    for (size_t i = 0; i < n; ++i) {
        auto vals = fetch(i);
        assign_to_targets(i, vals, targets...);
    }
}

// Permutation
template <typename KeyFunc>
bool compute_groups(const size_t n, KeyFunc&& key_func, std::vector<size_t>& group_sizes,
                    std::vector<size_t>& group_offsets, std::vector<int>& group_keys) {
    if (n == 0) {
        group_sizes.clear();
        group_offsets.clear();
        group_keys.clear();
        return false;
    }

    std::vector<int> keys(n);
    for (int i = 0; i < n; ++i) {
        keys[i] = key_func(i);
    }

    // Get sorted unique keys
    std::ranges::sort(keys);
    const auto last = std::ranges::unique(keys).begin();
    group_keys = std::vector(keys.begin(), last);
    const size_t num_groups = group_keys.size();

    group_sizes.resize(num_groups);
    group_offsets.resize(num_groups+1);
    group_offsets[0] = 0;
    group_offsets[num_groups] = n;

    // Count members of each group
    auto counts = std::vector<size_t>(num_groups, 0);
    for (int i = 0; i < n; ++i) {
        auto first = std::ranges::lower_bound(group_keys, keys[i]);
        const int group_idx = static_cast<int>(first - group_keys.begin());
        counts[group_idx]++;
    }

    size_t offset = 0;
    for (int g = 0; g < num_groups; ++g) {
        group_sizes[g] = counts[g];
        offset += counts[g];
        group_offsets[g+1] = offset;
    }
    return num_groups > 1;
}

template <typename CompFunc>
void compute_permutation(const size_t n, std::vector<size_t>& perm, std::vector<size_t>& inv_perm, CompFunc compare_func) {
    perm.resize(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::sort(perm, compare_func);

    inv_perm.resize(n);
    for (size_t i = 0; i < n; ++i) {
        inv_perm[perm[i]] = i;
    }
}

template <typename... InputVecs, typename... OutputVecs>
requires (sizeof...(InputVecs) == sizeof...(OutputVecs))
void permute(const std::vector<size_t>& perm, const InputVecs&... inputs, OutputVecs&... outputs) {
    const size_t n = perm.size();
    auto input_tuple = std::tie(inputs...);
    auto output_tuple = std::tie(outputs...);

    // Resize all outputs
    std::apply([&](auto&... out_vecs) {
        ((out_vecs.resize(n)), ...);
    }, output_tuple);

    // Permute everything
    for (size_t i = 0; i < n; ++i) {
        const size_t p = perm[i];
        auto permute_one = [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((std::get<Is>(output_tuple)[i] = std::get<Is>(input_tuple)[p]), ...);
        };
        permute_one(std::make_index_sequence<sizeof...(InputVecs)>{});
    }
}

// Copy memory from device to host
template <typename HostVec, typename DeviceVec, typename... Rest>
void from_device(size_t n, HostVec& h, const DeviceVec* d, Rest&&... rest) {
    using T = typename HostVec::value_type;
    static_assert(std::is_same_v<T, std::remove_pointer_t<std::remove_const_t<DeviceVec>>>,
                  "Type mismatch between host value_type and device pointer");
    const cudaError_t err = cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; }
    if constexpr (sizeof...(Rest) > 0) {
        from_device(n, std::forward<Rest>(rest)...);
    }
}
template <typename HostVec, typename DeviceVec, typename... Rest>
void from_device(const std::vector<size_t>& perm, HostVec& h, const DeviceVec* d, Rest&&... rest) {
    size_t n = perm.size();
    using T = typename HostVec::value_type;
    static_assert(std::is_same_v<T, std::remove_pointer_t<std::remove_const_t<DeviceVec>>>,
                  "Type mismatch between host value_type and device pointer");
    std::vector<T> temp(n);
    const cudaError_t err = cudaMemcpy(temp.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for permuted: " << cudaGetErrorString(err) << std::endl;
    }
    for (size_t gid = 0; gid < n; ++gid) {
        const size_t oid = perm[gid];
        h[oid] = temp[gid];
    }
    if constexpr (sizeof...(Rest) > 0) {
        from_device(perm, std::forward<Rest>(rest)...);
    }
}

// Copy memory from host to device
template <typename HostVec, typename DeviceVec, typename... Rest>
void to_device(size_t n, const HostVec& h, DeviceVec* d, Rest&&... rest) {
    using T = typename HostVec::value_type;
    static_assert(std::is_same_v<std::remove_pointer_t<DeviceVec>, T>, "Type mismatch between host value_type and device pointer");
    const cudaError_t err = cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    if constexpr (sizeof...(Rest) > 0) {
        to_device(n, std::forward<Rest>(rest)...);
    }
}
template <typename HostVec, typename DeviceVec, typename... Rest>
void to_device(const std::vector<size_t>& perm, const HostVec& h, DeviceVec* d, Rest&&... rest) {
    size_t n = perm.size();
    using T = typename HostVec::value_type;
    static_assert(std::is_same_v<std::remove_pointer_t<DeviceVec>, T>, "Type mismatch between host value_type and device pointer");
    std::vector<T> p(n);
    for (size_t i = 0; i < n; ++i) {
        p[i] = h[perm[i]];
    }
    const cudaError_t err = cudaMemcpy(d, p.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for permuted: " << cudaGetErrorString(err) << std::endl;
    }
    if constexpr (sizeof...(Rest) > 0) {
        to_device(perm, std::forward<Rest>(rest)...);
    }
}

// Stream handling
inline void init_streams() {
    if (streams_initialized) {
        for (const auto& s : streams) { if (s) cudaStreamDestroy(s); }
        streams.clear();
        streams_initialized = false;
    }
    streams.resize(num_groups);
    for (auto& s : streams) { cudaStreamCreate(&s); }
    streams_initialized = true;
}
inline void sync_streams() {
    for (const auto& stream : streams) {
        cudaStreamSynchronize(stream);
    }
}
inline void free_streams() {
    for (const auto& stream : streams) {
        if (!stream) { continue; }
        const cudaError_t err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
    streams.clear();
    streams_initialized = false;
    num_groups = 0;
}

// Batch device allocate
template <typename Ptr, typename... Rest>
void allocate_device(size_t n, Ptr*& ptr, Rest&&... rest) {
    using T = std::remove_pointer_t<Ptr>;
    const cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    }
    if constexpr (sizeof...(Rest) > 0) {
        allocate_device(n, std::forward<Rest>(rest)...);
    }
}

// Batch device free
template <typename Ptr, typename... Rest>
void free_device(Ptr* ptr, Rest&&... rest) {
    if (ptr) {
        const cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
    if constexpr (sizeof...(Rest) > 0) {
        free_device(std::forward<Rest>(rest)...);
    }
}