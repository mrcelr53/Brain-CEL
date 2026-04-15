/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2025 Your Name
 */

#include "NeuronGroup.cuh"

#include <chrono>

#include "spiking/Membrane.h"

__global__ void init_rand_kernel(const size_t n, curandState *states, const unsigned long seed) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void lif_adapt_kernel(const float *d_I_t, // inputs
                                 uint8_t *d_S_t, float *d_V_t, float *d_U_t, uint8_t *d_T_t, // outputs
                                 // global parameters
                                 const size_t n, const float dt,
                                 // individual parameters
                                 const float *d_reverse, const float *d_resist,
                                 const float *d_V_thresh, const float *d_V_rest, const float *d_adapt_amp,
                                 const uint8_t *d_refract, const float *d_alpha, const float *d_beta,
                                 const float *d_prob, curandState *rand_states) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    const float reverse = d_reverse[id];
    const float resist = d_resist[id];
    const float thresh = d_V_thresh[id];
    const float rest = d_V_rest[id];
    const float adapt_amp = d_adapt_amp[id];
    const uint8_t refract = d_refract[id];
    const float alpha = d_alpha[id];
    const float beta = d_beta[id];
    const float prob = d_prob[id];

    const float I_0 = d_I_t[id] / dt;
    const float V_0 = d_V_t[id];
    const float U_0 = d_U_t[id];
    const uint8_t T_s = d_T_t[id]; // elapsed ticks since last spike (refractory: 0 - refract | free: refract+1 - 255)
    const float adapt_thresh = thresh + U_0;

    // Reset spike
    d_S_t[id] = 0;

    // Tick refractory counter
    if (T_s == 255u) { d_T_t[id] = refract; } else { d_T_t[id]++; }

    // Refractory period
    if (T_s < refract) {
        d_V_t[id] = rest;
        return;
    }

    // Update membrane potential
    d_V_t[id] = V_0 * alpha + reverse * (1.0f - alpha) + resist * I_0 * (1.0f - alpha);

    // Spike
    const float rand = curand_uniform(&rand_states[id]);
    const bool spike = d_V_t[id] >= adapt_thresh || rand < prob;
    if (spike) {
        d_V_t[id] = rest;
        d_S_t[id] = 1;
        d_T_t[id] = 0; // only counts from 0 after spike
    }

    // Update adaptive threshold
    d_U_t[id] = U_0 * beta + adapt_amp * (1.0f - beta) * static_cast<float>(spike);
}

__global__ void lif_refract_kernel(const float *d_I_t,                                  // inputs
                                   uint8_t *d_S_t, float *d_V_t, uint8_t *d_T_t,        // outputs
                                   // global parameters
                                   const size_t n, const float dt,
                                   // individual parameters
                                   const float *d_reverse, const float *d_resist,
                                   const float *d_V_thresh, const float *d_V_rest,
                                   const uint8_t *d_refract, const float *d_alpha,
                                   const float* d_prob, curandState *rand_states) {

    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    const float reverse     = d_reverse[id];
    const float resist      = d_resist[id];
    const float thresh      = d_V_thresh[id];
    const float rest        = d_V_rest[id];
    const uint8_t refract   = d_refract[id];
    const float alpha       = d_alpha[id];
    const float prob        = d_prob[id];

    const float I_0 = d_I_t[id] / dt;
    const float V_0 = d_V_t[id];
    const uint8_t T_s = d_T_t[id];  // elapsed ticks since last spike (refractory: 0 - refract | free: refract+1 - 255)

    // Reset spike
    d_S_t[id] = 0;

    // Tick counter
    if (T_s == 255u) { d_T_t[id] = refract; }
    else             { d_T_t[id]++; }

    // Refractory period
    if (T_s < refract) {
        d_V_t[id] = rest;
        return;
    }

    // Update membrane potential
    d_V_t[id] = V_0 * alpha + reverse * (1.0f - alpha) + resist * I_0 * (1.0f - alpha);

    // Spike
    const float rand = curand_uniform(&rand_states[id]);
    if (d_V_t[id] >= thresh || rand < prob) {
        d_V_t[id] = rest;
        d_S_t[id] = 1;
        d_T_t[id] = 0; // only counts from 0 after spike
    }
}

__global__ void lif_kernel(const float *d_I_t,                  // inputs
                           uint8_t *d_S_t, float *d_V_t,        // outputs
                           // global parameters
                           const size_t n, const float dt,
                           // individual parameters
                           const float *d_reverse, const float *d_resist,
                           const float *d_V_thresh, const float *d_V_rest, const float *d_alpha,
                           const float *d_prob, curandState *rand_states) {

    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    const float reverse     = d_reverse[id];
    const float resist      = d_resist[id];
    const float thresh      = d_V_thresh[id];
    const float rest        = d_V_rest[id];
    const float alpha       = d_alpha[id];
    const float prob        = d_prob[id];

    const float I_0 = d_I_t[id] / dt;
    const float V_0 = d_V_t[id];

    // Reset spike
    d_S_t[id] = 0;

    // Update membrane potential
    d_V_t[id] = V_0 * alpha + reverse * (1.0f - alpha) + resist * I_0 * (1.0f - alpha);

    // Spike
    const float rand = curand_uniform(&rand_states[id]);
    if (d_V_t[id] >= thresh || rand < prob) {
        d_V_t[id] = rest;
        d_S_t[id] = 1;
    }
}

__global__ void izh_kernel(const float *d_I_t,                                // inputs
                           uint8_t *d_S_t, float *d_V_t, float *d_U_t,        // outputs
                           // global parameters
                           const size_t n, const float dt,
                           // individual parameters
                           const float *d_a, const float *d_b, const float *d_c, const float *d_d,
                           const float *d_prob, curandState *rand_states) {

    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    const float a      = d_a[id];
    const float b      = d_b[id];
    const float c      = d_c[id];
    const float d      = d_d[id];
    const float prob   = d_prob[id];

    const float I_0      = d_I_t[id];
    const float v_0      = d_V_t[id];
    const float u_0      = d_U_t[id];

    // Update membrane potential (2x for higher resolution)
    const float dtt = dt/2;
    float v_t = v_0 + dtt * (0.04f * v_0 * v_0 + 5.f * v_0 + 140.f - u_0 + I_0);
          v_t = v_t + dtt * (0.04f * v_t * v_t + 5.f * v_t + 140.f - u_0 + I_0);  // double integration as in paper

    // Spike
    const float rand = curand_uniform(&rand_states[id]);
    if (v_t >= 10.f || rand < prob) {
        d_V_t[id] = c;
        d_U_t[id] = u_0 + d;
        d_S_t[id] = 1;
    }
    else {
        d_S_t[id] = 0;
        d_V_t[id] = v_t;
        d_U_t[id] = u_0 + dt * a * (b * v_t - u_0);   // uses v_t as in paper
    }
}

struct LifMembraneDeviceParams {
    float dt;

    float* d_I_t;
    uint8_t* d_S_t;
    float* d_V_t;
    float* d_U_t;
    uint8_t* d_T_t;

    const float* d_reverse;
    const float* d_resist;
    const float* d_V_thresh;
    const float* d_V_rest;
    const float* d_adapt_amp;
    const uint8_t* d_refract;
    const float* d_alpha;  // exp(-dt / V_tau)
    const float* d_beta;   // exp(-dt / adapt_tau)
    const float* d_prob;
    curandState* d_rand_states;
};

static bool lif_params_initialized = false;
static LifMembraneDeviceParams lif_params{};

void init_cuda_membranes(const std::vector<LifMembrane*>& membranes,
                         float*& d_I_t, uint8_t*& d_S_t, float*& d_V_t, float*& d_U_t, uint8_t*& d_T_t,
                         float& dt,
                         float*& d_reverse, float*& d_resist, float*& d_V_thresh,
                         float*& d_V_rest, float*& d_adapt_amp, uint8_t*& d_refract,
                         float*& d_alpha, float*& d_beta,
                         float*& d_prob, curandState*& d_rand_states, const uint64_t custom_seed) {
    const size_t n = membranes.size();
    if (n == 0) { return; }

    const LifMembrane* m = membranes[0];
    dt = m->timestep();

    // Initialize host vectors
    std::vector<float> h_V_t, h_U_t(n, 0.f), h_reverse(n);
    std::vector<uint8_t> h_T_t(n, 255u);
    std::vector<float> h_resist(n), h_V_thresh(n), h_V_rest(n), h_adapt_amp(n, 0.f), h_alpha(n), h_beta(n), h_prob(n);
    std::vector<uint8_t> h_refract(n, 0u);
    std::vector<curandState> h_rand_states(n);

    // Initialize device vectors
    allocate_device(n, d_I_t, d_S_t, d_V_t, d_U_t, d_T_t, d_rand_states,
                d_reverse, d_resist, d_V_thresh,
                d_V_rest, d_adapt_amp, d_refract, d_alpha, d_beta, d_prob);

    // Populate vectors
    auto fetch_params = [&](const size_t i) {
        const auto& mem = membranes[i];
        const auto& ms = mem->state();
        return std::make_tuple(mem->potential(), mem->adaption(), mem->refraction(),      // matches order in populate
                               ms.reverse, ms.resist, ms.V_thresh, ms.V_rest, ms.adapt_amp, ms.refract,
                               expf(-dt / ms.V_tau), expf(-dt / ms.adapt_tau), dt * ms.basefire / 1000.0f);
    };
    populate(n, fetch_params,
             h_V_t, h_U_t, h_T_t, h_reverse, h_resist, h_V_thresh, h_V_rest,
             h_adapt_amp, h_refract, h_alpha, h_beta, h_prob);

    // Random generator
    uint64_t seed = custom_seed;
    if (custom_seed == static_cast<uint64_t>(-1)) {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    size_t blocks = (n + 255) / 256;
    init_rand_kernel<<<blocks, 256>>>(n, d_rand_states, seed);
    cudaDeviceSynchronize();

    // Calculate group shapes
    auto sorter = [&h_adapt_amp, &h_refract](const int i) {
        return h_adapt_amp[i] != 0.f ? 2 : (h_refract[i] > 0 ? 1 : 0);
    };
    use_permutation = compute_groups(n, sorter, group_sizes, group_offsets, group_keys);
    num_groups = std::max(static_cast<size_t>(1), group_keys.size());

    // Move to device
    cudaMemset(d_I_t, 0, n * sizeof(float));
    cudaMemset(d_S_t, 0, n * sizeof(uint8_t));
    if (use_permutation) {
        // Permute all vectors into reshaped order
        auto comparator = [&](const int a, const int b) {
            const int ka = sorter(a);
            const int kb = sorter(b);
            return ka < kb || (ka == kb && a < b);
        };
        compute_permutation(n, perm, inv_perm, comparator);

        // Permute random states
        from_device(n, h_rand_states, d_rand_states);

        // From host to device
        to_device(perm, h_V_t, d_V_t,
                 h_U_t,        d_U_t,
                 h_T_t,        d_T_t,
                 h_reverse,    d_reverse,
                 h_resist,     d_resist,
                 h_V_thresh,   d_V_thresh,
                 h_V_rest,     d_V_rest,
                 h_adapt_amp,  d_adapt_amp,
                 h_refract, d_refract,
                  h_alpha, d_alpha,
                  h_beta, d_beta,
                  h_prob, d_prob,
                  h_rand_states, d_rand_states);
    }
    else {
        // No reshaping - copy over directly
        to_device(n, h_V_t, d_V_t,
                  h_U_t, d_U_t,
                  h_T_t, d_T_t,
                  h_reverse, d_reverse,
                  h_resist, d_resist,
                  h_V_thresh, d_V_thresh,
                  h_V_rest, d_V_rest,
                  h_adapt_amp, d_adapt_amp,
                  h_refract, d_refract,
                  h_alpha, d_alpha,
                  h_beta, d_beta,
                  h_prob, d_prob);
    }

    // Initialize streams for parallel kernels
    init_streams();

    // Synchronize device
    cudaDeviceSynchronize();

    // Bundle device params
    lif_params = {dt, d_I_t, d_S_t, d_V_t, d_U_t, d_T_t, d_reverse, d_resist,
                     d_V_thresh, d_V_rest, d_adapt_amp, d_refract, d_alpha, d_beta, d_prob, d_rand_states};
    lif_params_initialized = true;
}

struct IzhMembraneDeviceParams {
    float dt;
    float* d_I_t;
    uint8_t* d_S_t;
    float* d_V_t;
    float* d_U_t;

    const float* d_a;
    const float* d_b;
    const float* d_c;
    const float* d_d;
    const float* d_prob;
    curandState* d_rand_states;
};
static IzhMembraneDeviceParams izh_params{};
static bool izh_params_initialized = false;

void init_cuda_membranes(const std::vector<IzhMembrane*> &membranes,
                         float *&d_I_t, uint8_t *&d_S_t, float *&d_V_t, float *&d_U_t,
                         float &dt, float *&d_a, float *&d_b, float *&d_c, float *&d_d,
                         float *&d_prob, curandState*& d_rand_states, const uint64_t custom_seed) {
    const size_t n = membranes.size();
    if (n == 0) { return; }

    const IzhMembrane* m = membranes[0];
    dt = m->timestep();

    // Initialize host vectors
    std::vector<float> h_V_t(n), h_U_t(n, 0.f);
    std::vector<float> h_a(n), h_b(n), h_c(n), h_d(n), h_prob(n);
    std::vector<curandState> h_rand_states(n);

    // Initialize device vectors
    allocate_device(n, d_I_t, d_S_t, d_V_t, d_U_t, d_rand_states,
                d_a, d_b, d_c, d_d, d_prob);

    // Populate vectors
    auto fetch_params = [&](const size_t i) {
        const auto& mem = membranes[i];
        const auto& ms = mem->state();
        return std::make_tuple(mem->potential(), mem->recovery(),      // matches order in populate
                               ms.U_timescale, ms.U_sensitivity, ms.V_reset, ms.U_reset, dt * ms.basefire / 1000.0f);
    };
    populate(n, fetch_params, h_V_t, h_U_t, h_a, h_b, h_c, h_d, h_prob);

    // Random generator
    uint64_t seed = custom_seed;
    if (custom_seed == static_cast<uint64_t>(-1)) {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    size_t blocks = (n + 255) / 256;
    init_rand_kernel<<<blocks, 256>>>(n, d_rand_states, seed);
    cudaDeviceSynchronize();

    auto sorter = [&](const int i) { return false; };
    use_permutation = compute_groups(n, sorter, group_sizes, group_offsets, group_keys);
    num_groups = std::max(static_cast<size_t>(1), group_keys.size());

    // Move to device
    cudaMemset(d_I_t, 0, n * sizeof(float));
    cudaMemset(d_S_t, 0, n * sizeof(uint8_t));

    to_device(n,
              h_V_t,            d_V_t,
          h_U_t,        d_U_t,
          h_a,          d_a,
          h_b,          d_b,
          h_c,          d_c,
          h_d,          d_d,
          h_prob,       d_prob);

    // Initialize streams for parallel kernels
    init_streams();

    // Synchronize device
    cudaDeviceSynchronize();

    // Bundle device params
    izh_params = {dt, d_I_t, d_S_t, d_V_t, d_U_t, d_a, d_b, d_c, d_d, d_prob, d_rand_states};
    izh_params_initialized = true;
}

void update_cuda_lif_membranes(const std::vector<float>& h_I_t,                             // inputs
                               std::vector<uint8_t>& h_S_t, std::vector<float>& h_V_t) {    // outputs
    const size_t n = h_I_t.size();
    h_S_t.resize(n, 0u);
    h_V_t.resize(n);
    if (n == 0) return;

    auto lif_base = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        lif_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off, n, p.dt,
                                                   p.d_reverse + off, p.d_resist + off,
                                                   p.d_V_thresh + off, p.d_V_rest + off,
                                                   p.d_alpha + off,
                                                   p.d_prob + off, p.d_rand_states + off);
    };
    auto lif_refract = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        lif_refract_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off,
                                                           p.d_T_t + off, n, p.dt,
                                                           p.d_reverse + off, p.d_resist + off,
                                                           p.d_V_thresh + off, p.d_V_rest + off,
                                                           p.d_refract + off, p.d_alpha + off,
                                                           p.d_prob + off,  p.d_rand_states + off);
    };
    auto lif_adapt = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        lif_adapt_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off,
                                                         p.d_U_t + off, p.d_T_t + off, n, p.dt,
                                                         p.d_reverse + off, p.d_resist + off,
                                                         p.d_V_thresh + off, p.d_V_rest + off,
                                                         p.d_adapt_amp + off, p.d_refract + off,
                                                         p.d_alpha + off, p.d_beta + off,
                                                         p.d_prob + off, p.d_rand_states + off);
    };

    // Pipeline
    if (use_permutation) {
        to_device(perm, h_I_t, lif_params.d_I_t);
        exec_kernels(std::make_tuple(lif_base, lif_refract, lif_adapt),
                 group_sizes, group_offsets, group_keys, lif_params);
        from_device(perm, h_S_t, lif_params.d_S_t, h_V_t, lif_params.d_V_t);
    }
    else {
        to_device(n, h_I_t, lif_params.d_I_t);
        exec_kernels(std::make_tuple(lif_base, lif_refract, lif_adapt),
                 group_sizes, group_offsets, group_keys, lif_params);
        from_device(n, h_S_t, lif_params.d_S_t, h_V_t, lif_params.d_V_t);
     }

    sync_streams();
}
void update_cuda_lif_membranes(const std::vector<float>& h_I_t,                             // inputs
                               std::vector<uint8_t>& h_S_t) {                               // outputs
    const size_t n = h_I_t.size();
    h_S_t.resize(n, 0u);
    if (n == 0) return;

    auto lif_base = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        lif_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off, n, p.dt,
                                                   p.d_reverse + off, p.d_resist + off,
                                                   p.d_V_thresh + off, p.d_V_rest + off,
                                                   p.d_alpha + off,
                                                   p.d_prob + off, p.d_rand_states + off);
    };
    auto lif_refract = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        lif_refract_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off,
                                                           p.d_T_t + off, n, p.dt,
                                                           p.d_reverse + off, p.d_resist + off,
                                                           p.d_V_thresh + off, p.d_V_rest + off,
                                                           p.d_refract + off, p.d_alpha + off,
                                                           p.d_prob + off,  p.d_rand_states + off);
    };
    auto lif_adapt = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        lif_adapt_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off,
                                                         p.d_U_t + off, p.d_T_t + off, n, p.dt,
                                                         p.d_reverse + off, p.d_resist + off,
                                                         p.d_V_thresh + off, p.d_V_rest + off,
                                                         p.d_adapt_amp + off, p.d_refract + off,
                                                         p.d_alpha + off, p.d_beta + off,
                                                         p.d_prob + off, p.d_rand_states + off);
    };

    // Pipeline
    if (use_permutation) {
        to_device(perm, h_I_t, lif_params.d_I_t);
        exec_kernels(std::make_tuple(lif_base, lif_refract, lif_adapt),
                 group_sizes, group_offsets, group_keys, lif_params);
        from_device(perm, h_S_t, lif_params.d_S_t);
    }
    else {
        to_device(n, h_I_t, lif_params.d_I_t);
        exec_kernels(std::make_tuple(lif_base, lif_refract, lif_adapt),
                 group_sizes, group_offsets, group_keys, lif_params);
        from_device(n, h_S_t, lif_params.d_S_t);
    }

    sync_streams();
}

void update_cuda_izh_membranes(const std::vector<float>& h_I_t,
                               std::vector<uint8_t>& h_S_t, std::vector<float>& h_V_t) {
    const size_t n = h_I_t.size();
    h_S_t.resize(n, 0u);
    h_V_t.resize(n);
    if (n == 0) return;

    auto izh_calc = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        izh_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off, p.d_U_t + off, n, p.dt,
                                                   p.d_a + off, p.d_b + off, p.d_c + off, p.d_d + off,
                                                   p.d_prob + off, p.d_rand_states + off);
    };

    to_device(n, h_I_t, izh_params.d_I_t);
    exec_kernels(std::make_tuple(izh_calc),
                 group_sizes, group_offsets, group_keys, izh_params);
    from_device(n, h_S_t, izh_params.d_S_t, h_V_t, izh_params.d_V_t);

    sync_streams();
}
void update_cuda_izh_membranes(const std::vector<float>& h_I_t,
                               std::vector<uint8_t>& h_S_t) {
    const size_t n = h_I_t.size();
    h_S_t.resize(n, 0u);
    if (n == 0) return;

    auto izh_calc = [](size_t blocks, size_t threads, cudaStream_t stream, const auto& p, size_t off, const size_t n) {
        izh_kernel<<<blocks, threads, 0, stream>>>(p.d_I_t + off, p.d_S_t + off, p.d_V_t + off, p.d_U_t + off, n, p.dt,
                                                   p.d_a + off, p.d_b + off, p.d_c + off, p.d_d + off,
                                                   p.d_prob + off, p.d_rand_states + off);
    };

    to_device(n, h_I_t, izh_params.d_I_t);
    exec_kernels(std::make_tuple(izh_calc),
                 group_sizes, group_offsets, group_keys, izh_params);
    from_device(n, h_S_t, izh_params.d_S_t);

    sync_streams();
}

