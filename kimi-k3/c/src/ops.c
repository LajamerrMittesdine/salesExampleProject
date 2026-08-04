#include "kimi_k3_ref.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p && n) abort();
  return p;
}

void k3_rmsnorm(const float *x, const float *weight, size_t rows, size_t hidden, float eps, float *out) {
  for (size_t r = 0; r < rows; ++r) {
    float var = 0.f;
    for (size_t j = 0; j < hidden; ++j) {
      float v = x[r * hidden + j];
      var += v * v;
    }
    var /= (float)hidden;
    float inv = 1.f / sqrtf(var + eps);
    for (size_t j = 0; j < hidden; ++j) out[r * hidden + j] = x[r * hidden + j] * inv * weight[j];
  }
}

void k3_situ_and_mul(const float *x, size_t rows, size_t last_dim, float beta, int has_linear_beta,
                     float linear_beta, float *out) {
  size_t d = last_dim / 2;
  for (size_t r = 0; r < rows; ++r) {
    const float *xs = x + r * last_dim;
    float *os = out + r * d;
    for (size_t i = 0; i < d; ++i) {
      float gate = xs[i];
      float up = xs[d + i];
      float situ_a = beta * tanhf(gate / beta) * sigmoidf_(gate);
      if (has_linear_beta) up = linear_beta * tanhf(up / linear_beta);
      os[i] = situ_a * up;
    }
  }
}

void k3_linear(const float *x, size_t rows, size_t in_dim, const float *weight, size_t out_dim,
               float *out) {
  for (size_t r = 0; r < rows; ++r) {
    for (size_t o = 0; o < out_dim; ++o) {
      float s = 0.f;
      const float *w = weight + o * in_dim;
      const float *xr = x + r * in_dim;
      for (size_t j = 0; j < in_dim; ++j) s += xr[j] * w[j];
      out[r * out_dim + o] = s;
    }
  }
}

void k3_moe_gate(const float *hidden, size_t n, size_t h, const float *weight, size_t num_experts,
                 const float *bias, size_t top_k, float scale, int renormalize, long long *topk_idx,
                 float *topk_weight) {
  float *scores = xmalloc(num_experts * sizeof(float));
  float *choice = xmalloc(num_experts * sizeof(float));
  size_t *order = xmalloc(num_experts * sizeof(size_t));
  size_t *chosen = xmalloc(top_k * sizeof(size_t));
  for (size_t row = 0; row < n; ++row) {
    const float *x = hidden + row * h;
    for (size_t e = 0; e < num_experts; ++e) {
      float logit = 0.f;
      const float *w = weight + e * h;
      for (size_t j = 0; j < h; ++j) logit += x[j] * w[j];
      scores[e] = sigmoidf_(logit);
      choice[e] = scores[e] + bias[e];
      order[e] = e;
    }
    for (size_t i = 0; i < num_experts; ++i) {
      for (size_t j = i + 1; j < num_experts; ++j) {
        int swap = 0;
        if (choice[order[j]] > choice[order[i]]) swap = 1;
        else if (choice[order[j]] == choice[order[i]] && order[j] < order[i]) swap = 1;
        if (swap) { size_t t = order[i]; order[i] = order[j]; order[j] = t; }
      }
    }
    for (size_t k = 0; k < top_k; ++k) chosen[k] = order[k];
    for (size_t i = 0; i < top_k; ++i)
      for (size_t j = i + 1; j < top_k; ++j)
        if (chosen[j] < chosen[i]) { size_t t = chosen[i]; chosen[i] = chosen[j]; chosen[j] = t; }
    for (size_t k = 0; k < top_k; ++k) {
      topk_idx[row * top_k + k] = (long long)chosen[k];
      topk_weight[row * top_k + k] = scores[chosen[k]];
    }
    if (top_k > 1 && renormalize) {
      float denom = 1e-20f;
      for (size_t k = 0; k < top_k; ++k) denom += topk_weight[row * top_k + k];
      for (size_t k = 0; k < top_k; ++k) topk_weight[row * top_k + k] /= denom;
    }
    for (size_t k = 0; k < top_k; ++k) topk_weight[row * top_k + k] *= scale;
  }
  free(scores); free(choice); free(order); free(chosen);
}

void k3_expert_ffn(const float *x, size_t rows, size_t hidden, const float *w1, const float *w2,
                   const float *w3, size_t inter, float beta, int has_linear_beta, float linear_beta,
                   float *out) {
  float *gate = xmalloc(rows * inter * sizeof(float));
  float *up = xmalloc(rows * inter * sizeof(float));
  float *gate_up = xmalloc(rows * 2 * inter * sizeof(float));
  float *act = xmalloc(rows * inter * sizeof(float));
  k3_linear(x, rows, hidden, w1, inter, gate);
  k3_linear(x, rows, hidden, w3, inter, up);
  for (size_t r = 0; r < rows; ++r) {
    memcpy(gate_up + r * 2 * inter, gate + r * inter, inter * sizeof(float));
    memcpy(gate_up + r * 2 * inter + inter, up + r * inter, inter * sizeof(float));
  }
  k3_situ_and_mul(gate_up, rows, 2 * inter, beta, has_linear_beta, linear_beta, act);
  k3_linear(act, rows, inter, w2, hidden, out);
  free(gate); free(up); free(gate_up); free(act);
}

static void stable_softmax_rows(const float *x, size_t rows, size_t cols, float *out) {
  for (size_t r = 0; r < rows; ++r) {
    const float *row = x + r * cols;
    float m = -INFINITY;
    for (size_t c = 0; c < cols; ++c) if (row[c] > m) m = row[c];
    float sum = 0.f;
    for (size_t c = 0; c < cols; ++c) { out[r * cols + c] = expf(row[c] - m); sum += out[r * cols + c]; }
    for (size_t c = 0; c < cols; ++c) out[r * cols + c] /= sum;
  }
}

void k3_apply_attn_res(const float *prefix, size_t n, size_t hidden, const float *block_residual,
                       size_t num_blocks, const float *proj, const float *norm, float eps, float *out) {
  size_t blocks = num_blocks + 1;
  float *v = xmalloc(n * blocks * hidden * sizeof(float));
  float *scores = xmalloc(n * blocks * sizeof(float));
  float *probs = xmalloc(n * blocks * sizeof(float));
  for (size_t i = 0; i < n; ++i) {
    for (size_t b = 0; b < num_blocks; ++b)
      memcpy(v + (i * blocks + b) * hidden, block_residual + (i * num_blocks + b) * hidden, hidden * sizeof(float));
    memcpy(v + (i * blocks + num_blocks) * hidden, prefix + i * hidden, hidden * sizeof(float));
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t b = 0; b < blocks; ++b) {
      const float *vv = v + (i * blocks + b) * hidden;
      float var = 0.f;
      for (size_t j = 0; j < hidden; ++j) var += vv[j] * vv[j];
      var /= (float)hidden;
      float inv = 1.f / sqrtf(var + eps);
      float score = 0.f;
      for (size_t j = 0; j < hidden; ++j) score += vv[j] * inv * norm[j] * proj[j];
      scores[i * blocks + b] = score;
    }
  }
  stable_softmax_rows(scores, n, blocks, probs);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < hidden; ++j) {
      float acc = 0.f;
      for (size_t b = 0; b < blocks; ++b) acc += probs[i * blocks + b] * v[(i * blocks + b) * hidden + j];
      out[i * hidden + j] = acc;
    }
  free(v); free(scores); free(probs);
}

void k3_short_conv1d_silu(const float *x, size_t batch, size_t time, size_t channels,
                          const float *weight, size_t kernel, float *out) {
  for (size_t b = 0; b < batch; ++b)
    for (size_t t = 0; t < time; ++t)
      for (size_t c = 0; c < channels; ++c) {
        float acc = 0.f;
        for (size_t i = 0; i < kernel; ++i) {
          int src = (int)t + (int)i - ((int)kernel - 1);
          float xv = src >= 0 ? x[(b * time + (size_t)src) * channels + c] : 0.f;
          acc += xv * weight[c * kernel + i];
        }
        out[(b * time + t) * channels + c] = acc * sigmoidf_(acc);
      }
}

void k3_kda_lowerbound_gate(const float *g, size_t outer, size_t heads, size_t dim, const float *a_log,
                            const float *dt_bias, float lower_bound, float *out) {
  float *a_exp = xmalloc(heads * sizeof(float));
  for (size_t h = 0; h < heads; ++h) a_exp[h] = expf(a_log[h]);
  for (size_t o = 0; o < outer; ++o)
    for (size_t h = 0; h < heads; ++h)
      for (size_t d = 0; d < dim; ++d) {
        size_t idx = (o * heads + h) * dim + d;
        float v = g[idx];
        if (dt_bias) v += dt_bias[h * dim + d];
        out[idx] = lower_bound * sigmoidf_(a_exp[h] * v);
      }
  free(a_exp);
}

void k3_l2_normalize(const float *x, size_t rows, size_t dim, float eps, float *out) {
  for (size_t r = 0; r < rows; ++r) {
    float s = 0.f;
    for (size_t j = 0; j < dim; ++j) { float v = x[r * dim + j]; s += v * v; }
    float inv = 1.f / sqrtf(s + eps);
    for (size_t j = 0; j < dim; ++j) out[r * dim + j] = x[r * dim + j] * inv;
  }
}

void k3_kda_recurrent(const float *q, const float *k, const float *v, const float *g, const float *beta,
                      size_t batch, size_t time, size_t heads, size_t dim, int l2norm_qk, float *out,
                      float *final_state) {
  float scale = 1.f / sqrtf((float)dim);
  size_t nqk = batch * time * heads * dim;
  float *qq = xmalloc(nqk * sizeof(float));
  float *kk = xmalloc(nqk * sizeof(float));
  memcpy(qq, q, nqk * sizeof(float));
  memcpy(kk, k, nqk * sizeof(float));
  if (l2norm_qk) {
    float *tq = xmalloc(nqk * sizeof(float));
    float *tk = xmalloc(nqk * sizeof(float));
    k3_l2_normalize(qq, batch * time * heads, dim, 1e-6f, tq);
    k3_l2_normalize(kk, batch * time * heads, dim, 1e-6f, tk);
    free(qq); free(kk); qq = tq; kk = tk;
  }
  for (size_t i = 0; i < nqk; ++i) qq[i] *= scale;
  size_t sn = batch * heads * dim * dim;
  float *s = xmalloc(sn * sizeof(float));
  memset(s, 0, sn * sizeof(float));
  float *ks = xmalloc(dim * sizeof(float));
  float *delta = xmalloc(dim * sizeof(float));
  for (size_t t = 0; t < time; ++t)
    for (size_t b = 0; b < batch; ++b)
      for (size_t h = 0; h < heads; ++h) {
        size_t q_base = ((b * time + t) * heads + h) * dim;
        size_t s_base = (b * heads + h) * dim * dim;
        float bval = beta[(b * time + t) * heads + h];
        for (size_t kd = 0; kd < dim; ++kd) {
          float eg = expf(g[q_base + kd]);
          for (size_t vd = 0; vd < dim; ++vd) s[s_base + kd * dim + vd] *= eg;
        }
        for (size_t vd = 0; vd < dim; ++vd) {
          float acc = 0.f;
          for (size_t kd = 0; kd < dim; ++kd) acc += kk[q_base + kd] * s[s_base + kd * dim + vd];
          ks[vd] = acc;
          delta[vd] = v[q_base + vd] - ks[vd];
        }
        for (size_t kd = 0; kd < dim; ++kd) {
          float bk = bval * kk[q_base + kd];
          for (size_t vd = 0; vd < dim; ++vd) s[s_base + kd * dim + vd] += bk * delta[vd];
        }
        for (size_t vd = 0; vd < dim; ++vd) {
          float acc = 0.f;
          for (size_t kd = 0; kd < dim; ++kd) acc += qq[q_base + kd] * s[s_base + kd * dim + vd];
          out[q_base + vd] = acc;
        }
      }
  memcpy(final_state, s, sn * sizeof(float));
  free(qq); free(kk); free(s); free(ks); free(delta);
}

void k3_mla_eager_attention(const float *query, const float *key, const float *value, size_t batch,
                            size_t heads, size_t time, size_t dim, float scaling, int causal,
                            float *out) {
  size_t scn = batch * heads * time * time;
  float *scores = xmalloc(scn * sizeof(float));
  float *probs = xmalloc(scn * sizeof(float));
  float *tmp = xmalloc(batch * heads * time * dim * sizeof(float));
  for (size_t b = 0; b < batch; ++b)
    for (size_t h = 0; h < heads; ++h)
      for (size_t qi = 0; qi < time; ++qi)
        for (size_t kj = 0; kj < time; ++kj) {
          float s = 0.f;
          size_t qb = ((b * heads + h) * time + qi) * dim;
          size_t kb = ((b * heads + h) * time + kj) * dim;
          for (size_t d = 0; d < dim; ++d) s += query[qb + d] * key[kb + d];
          s *= scaling;
          if (causal && kj > qi) s = -1e9f;
          scores[(((b * heads + h) * time) + qi) * time + kj] = s;
        }
  stable_softmax_rows(scores, batch * heads * time, time, probs);
  for (size_t b = 0; b < batch; ++b)
    for (size_t h = 0; h < heads; ++h)
      for (size_t qi = 0; qi < time; ++qi)
        for (size_t d = 0; d < dim; ++d) {
          float acc = 0.f;
          for (size_t kj = 0; kj < time; ++kj) {
            float p = probs[(((b * heads + h) * time) + qi) * time + kj];
            acc += p * value[((b * heads + h) * time + kj) * dim + d];
          }
          tmp[((b * heads + h) * time + qi) * dim + d] = acc;
        }
  for (size_t b = 0; b < batch; ++b)
    for (size_t t = 0; t < time; ++t)
      for (size_t h = 0; h < heads; ++h)
        for (size_t d = 0; d < dim; ++d)
          out[((b * time + t) * heads + h) * dim + d] = tmp[((b * heads + h) * time + t) * dim + d];
  free(scores); free(probs); free(tmp);
}

void k3_gated_mla_output(const float *attn, const float *gate_logits, size_t n, float *out) {
  for (size_t i = 0; i < n; ++i) out[i] = attn[i] * sigmoidf_(gate_logits[i]);
}

void k3_latent_moe_forward(const float *hidden, size_t batch, size_t seq, size_t hidden_dim,
                           const float *router_w, const float *router_b, size_t num_experts,
                           size_t top_k, const float *const *ew1, const float *const *ew2,
                           const float *const *ew3, size_t latent, size_t inter,
                           const float *routed_down, const float *routed_up, const float *routed_norm,
                           const float *shared_gate, const float *shared_up, const float *shared_down,
                           float rms_eps, float beta, int has_linear_beta, float linear_beta,
                           float *out) {
  size_t n = batch * seq;
  long long *idx = xmalloc(n * top_k * sizeof(long long));
  float *wt = xmalloc(n * top_k * sizeof(float));
  float *latent_x = xmalloc(n * latent * sizeof(float));
  float *y = xmalloc(n * latent * sizeof(float));
  float *routed = xmalloc(n * hidden_dim * sizeof(float));
  float *shared = xmalloc(n * hidden_dim * sizeof(float));
  float *tmp = xmalloc(latent * sizeof(float));
  float *acc = xmalloc(latent * sizeof(float));
  k3_moe_gate(hidden, n, hidden_dim, router_w, num_experts, router_b, top_k, 1.f, 1, idx, wt);
  k3_linear(hidden, n, hidden_dim, routed_down, latent, latent_x);
  for (size_t t = 0; t < n; ++t) {
    memset(acc, 0, latent * sizeof(float));
    for (size_t k = 0; k < top_k; ++k) {
      size_t e = (size_t)idx[t * top_k + k];
      k3_expert_ffn(latent_x + t * latent, 1, latent, ew1[e], ew2[e], ew3[e], inter, beta,
                    has_linear_beta, linear_beta, tmp);
      for (size_t j = 0; j < latent; ++j) acc[j] += wt[t * top_k + k] * tmp[j];
    }
    memcpy(y + t * latent, acc, latent * sizeof(float));
  }
  if (routed_norm) {
    float *yn = xmalloc(n * latent * sizeof(float));
    k3_rmsnorm(y, routed_norm, n, latent, rms_eps, yn);
    free(y); y = yn;
  }
  k3_linear(y, n, latent, routed_up, hidden_dim, routed);
  k3_expert_ffn(hidden, n, hidden_dim, shared_gate, shared_down, shared_up, inter, beta,
                has_linear_beta, linear_beta, shared);
  for (size_t i = 0; i < n * hidden_dim; ++i) out[i] = routed[i] + shared[i];
  free(idx); free(wt); free(latent_x); free(y); free(routed); free(shared); free(tmp); free(acc);
}

void k3_decoder_block_forward(const float *hidden, size_t batch, size_t seq, size_t hidden_dim,
                              const float *block_residual, size_t num_blocks, const float *input_norm,
                              const float *post_norm, const float *sa_proj, const float *sa_norm,
                              const float *mlp_proj, const float *mlp_norm, size_t layer_idx,
                              size_t block_size, float attn_scale, float mlp_scale, float rms_eps,
                              float *prefix_out, float *br_out, size_t *br_out_len) {
  size_t n = batch * seq;
  size_t nh = n * hidden_dim;
  float *hidden_buf = xmalloc(nh * sizeof(float));
  float *prefix = xmalloc(nh * sizeof(float));
  int has_prefix = 1;
  memcpy(hidden_buf, hidden, nh * sizeof(float));
  memcpy(prefix, hidden, nh * sizeof(float));
  size_t nb = num_blocks;
  float *br = xmalloc((nb + 1) * nh * sizeof(float)); /* room to append one block */
  if (nb) memcpy(br, block_residual, nb * nh * sizeof(float));
  if (nb > 0) {
    float *tmp = xmalloc(nh * sizeof(float));
    k3_apply_attn_res(prefix, n, hidden_dim, br, nb, sa_proj, sa_norm, rms_eps, tmp);
    free(hidden_buf); hidden_buf = tmp;
  }
  if (layer_idx % block_size == 0) {
    memcpy(br + nb * nh, prefix, nh * sizeof(float));
    nb += 1;
    has_prefix = 0;
  }
  float *normed = xmalloc(nh * sizeof(float));
  k3_rmsnorm(hidden_buf, input_norm, n, hidden_dim, rms_eps, normed);
  for (size_t i = 0; i < nh; ++i) normed[i] *= attn_scale;
  float *prefix_sum = xmalloc(nh * sizeof(float));
  if (has_prefix) {
    for (size_t i = 0; i < nh; ++i) prefix_sum[i] = prefix[i] + normed[i];
  } else {
    memcpy(prefix_sum, normed, nh * sizeof(float));
  }
  float *hidden2 = xmalloc(nh * sizeof(float));
  float *post = xmalloc(nh * sizeof(float));
  k3_apply_attn_res(prefix_sum, n, hidden_dim, br, nb, mlp_proj, mlp_norm, rms_eps, hidden2);
  k3_rmsnorm(hidden2, post_norm, n, hidden_dim, rms_eps, post);
  for (size_t i = 0; i < nh; ++i) post[i] *= mlp_scale;
  for (size_t i = 0; i < nh; ++i) prefix_sum[i] += post[i];
  memcpy(prefix_out, prefix_sum, nh * sizeof(float));
  memcpy(br_out, br, nb * nh * sizeof(float));
  *br_out_len = nb * nh;
  free(hidden_buf); free(prefix); free(br); free(normed); free(prefix_sum); free(hidden2); free(post);
}
