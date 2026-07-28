#pragma once
// Idiomatic C++17 educational port of Kimi K3 float32 reference ops.
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace kimi::k3 {

void rmsnorm(std::span<const float> x, std::span<const float> weight, float eps,
             std::span<float> out);

void situ_and_mul(std::span<const float> x, std::size_t last_dim, float beta,
                  std::optional<float> linear_beta, std::span<float> out);

void linear(std::span<const float> x, std::size_t rows, std::size_t in_dim,
            std::span<const float> weight, std::size_t out_dim, std::span<float> out);

void moe_gate(std::span<const float> hidden, std::size_t n, std::size_t h,
              std::span<const float> weight, std::size_t num_experts,
              std::span<const float> bias, std::size_t top_k, float scale, bool renormalize,
              std::span<long long> topk_idx, std::span<float> topk_weight);

void expert_ffn(std::span<const float> x, std::size_t rows, std::size_t hidden,
                std::span<const float> w1, std::span<const float> w2, std::span<const float> w3,
                std::size_t inter, float beta, std::optional<float> linear_beta,
                std::span<float> out);

void apply_attn_res(std::span<const float> prefix, std::size_t n, std::size_t hidden,
                    std::span<const float> block_residual, std::size_t num_blocks,
                    std::span<const float> proj, std::span<const float> norm, float eps,
                    std::span<float> out);

void short_conv1d_silu(std::span<const float> x, std::size_t batch, std::size_t time,
                       std::size_t channels, std::span<const float> weight, std::size_t kernel,
                       std::span<float> out);

void kda_lowerbound_gate(std::span<const float> g, std::size_t outer, std::size_t heads,
                         std::size_t dim, std::span<const float> a_log,
                         std::span<const float> dt_bias, float lower_bound, std::span<float> out);

void l2_normalize(std::span<const float> x, std::size_t rows, std::size_t dim, float eps,
                  std::span<float> out);

void kda_recurrent(std::span<const float> q, std::span<const float> k, std::span<const float> v,
                   std::span<const float> g, std::span<const float> beta, std::size_t batch,
                   std::size_t time, std::size_t heads, std::size_t dim, bool l2norm_qk,
                   std::span<float> out, std::span<float> final_state);

void mla_eager_attention(std::span<const float> query, std::span<const float> key,
                         std::span<const float> value, std::size_t batch, std::size_t heads,
                         std::size_t time, std::size_t dim, float scaling, bool causal,
                         std::span<float> out);

void gated_mla_output(std::span<const float> attn, std::span<const float> gate_logits,
                      std::span<float> out);

void latent_moe_forward(std::span<const float> hidden, std::size_t batch, std::size_t seq,
                        std::size_t hidden_dim, std::span<const float> router_w,
                        std::span<const float> router_b, std::size_t num_experts, std::size_t top_k,
                        const std::vector<std::vector<float>>& ew1,
                        const std::vector<std::vector<float>>& ew2,
                        const std::vector<std::vector<float>>& ew3, std::size_t latent,
                        std::size_t inter, std::span<const float> routed_down,
                        std::span<const float> routed_up, std::span<const float> routed_norm,
                        std::span<const float> shared_gate, std::span<const float> shared_up,
                        std::span<const float> shared_down, float rms_eps, float beta,
                        std::optional<float> linear_beta, std::span<float> out);

void decoder_block_forward(std::span<const float> hidden, std::size_t batch, std::size_t seq,
                           std::size_t hidden_dim, std::span<const float> block_residual,
                           std::size_t num_blocks, std::span<const float> input_norm,
                           std::span<const float> post_norm, std::span<const float> sa_proj,
                           std::span<const float> sa_norm, std::span<const float> mlp_proj,
                           std::span<const float> mlp_norm, std::size_t layer_idx,
                           std::size_t block_size, float attn_scale, float mlp_scale, float rms_eps,
                           std::span<float> prefix_out, std::vector<float>& br_out,
                           std::size_t& nb_out);

}  // namespace kimi::k3
