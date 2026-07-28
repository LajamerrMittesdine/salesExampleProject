#ifndef KIMI_K3_REF_H
#define KIMI_K3_REF_H
/* Idiomatic C99 educational port of Kimi K3 float32 reference ops. */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void k3_rmsnorm(const float *x, const float *weight, size_t rows, size_t hidden, float eps, float *out);
void k3_situ_and_mul(const float *x, size_t rows, size_t last_dim, float beta, int has_linear_beta,
                     float linear_beta, float *out);
void k3_linear(const float *x, size_t rows, size_t in_dim, const float *weight, size_t out_dim,
               float *out);
void k3_moe_gate(const float *hidden, size_t n, size_t h, const float *weight, size_t num_experts,
                 const float *bias, size_t top_k, float scale, int renormalize, long long *topk_idx,
                 float *topk_weight);
void k3_expert_ffn(const float *x, size_t rows, size_t hidden, const float *w1, const float *w2,
                   const float *w3, size_t inter, float beta, int has_linear_beta, float linear_beta,
                   float *out);
void k3_apply_attn_res(const float *prefix, size_t n, size_t hidden, const float *block_residual,
                       size_t num_blocks, const float *proj, const float *norm, float eps, float *out);
void k3_short_conv1d_silu(const float *x, size_t batch, size_t time, size_t channels,
                          const float *weight, size_t kernel, float *out);
void k3_kda_lowerbound_gate(const float *g, size_t outer, size_t heads, size_t dim, const float *a_log,
                            const float *dt_bias, float lower_bound, float *out);
void k3_l2_normalize(const float *x, size_t rows, size_t dim, float eps, float *out);
void k3_kda_recurrent(const float *q, const float *k, const float *v, const float *g, const float *beta,
                      size_t batch, size_t time, size_t heads, size_t dim, int l2norm_qk, float *out,
                      float *final_state);
void k3_mla_eager_attention(const float *query, const float *key, const float *value, size_t batch,
                            size_t heads, size_t time, size_t dim, float scaling, int causal,
                            float *out);
void k3_gated_mla_output(const float *attn, const float *gate_logits, size_t n, float *out);
void k3_latent_moe_forward(const float *hidden, size_t batch, size_t seq, size_t hidden_dim,
                           const float *router_w, const float *router_b, size_t num_experts,
                           size_t top_k, const float *const *ew1, const float *const *ew2,
                           const float *const *ew3, size_t latent, size_t inter,
                           const float *routed_down, const float *routed_up, const float *routed_norm,
                           const float *shared_gate, const float *shared_up, const float *shared_down,
                           float rms_eps, float beta, int has_linear_beta, float linear_beta,
                           float *out);
void k3_decoder_block_forward(const float *hidden, size_t batch, size_t seq, size_t hidden_dim,
                              const float *block_residual, size_t num_blocks, const float *input_norm,
                              const float *post_norm, const float *sa_proj, const float *sa_norm,
                              const float *mlp_proj, const float *mlp_norm, size_t layer_idx,
                              size_t block_size, float attn_scale, float mlp_scale, float rms_eps,
                              float *prefix_out, float *br_out, size_t *br_out_len);

#ifdef __cplusplus
}
#endif
#endif
