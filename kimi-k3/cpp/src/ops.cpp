#include "kimi_k3_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace kimi::k3 {
namespace {

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

void stable_softmax_rows(const float* x, std::size_t rows, std::size_t cols, float* out) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = x + r * cols;
    float m = -INFINITY;
    for (std::size_t c = 0; c < cols; ++c) m = std::max(m, row[c]);
    float sum = 0.f;
    for (std::size_t c = 0; c < cols; ++c) {
      out[r * cols + c] = std::exp(row[c] - m);
      sum += out[r * cols + c];
    }
    for (std::size_t c = 0; c < cols; ++c) out[r * cols + c] /= sum;
  }
}

}  // namespace

void rmsnorm(std::span<const float> x, std::span<const float> weight, float eps,
             std::span<float> out) {
  const std::size_t h = weight.size();
  const std::size_t rows = x.size() / h;
  for (std::size_t r = 0; r < rows; ++r) {
    float var = 0.f;
    for (std::size_t j = 0; j < h; ++j) {
      float v = x[r * h + j];
      var += v * v;
    }
    var /= static_cast<float>(h);
    float inv = 1.f / std::sqrt(var + eps);
    for (std::size_t j = 0; j < h; ++j) out[r * h + j] = x[r * h + j] * inv * weight[j];
  }
}

void situ_and_mul(std::span<const float> x, std::size_t last_dim, float beta,
                  std::optional<float> linear_beta, std::span<float> out) {
  const std::size_t d = last_dim / 2;
  const std::size_t rows = x.size() / last_dim;
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xs = x.data() + r * last_dim;
    float* os = out.data() + r * d;
    for (std::size_t i = 0; i < d; ++i) {
      float gate = xs[i];
      float up = xs[d + i];
      float situ_a = beta * std::tanh(gate / beta) * sigmoid(gate);
      if (linear_beta) up = *linear_beta * std::tanh(up / *linear_beta);
      os[i] = situ_a * up;
    }
  }
}

void linear(std::span<const float> x, std::size_t rows, std::size_t in_dim,
            std::span<const float> weight, std::size_t out_dim, std::span<float> out) {
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t o = 0; o < out_dim; ++o) {
      float s = 0.f;
      const float* w = weight.data() + o * in_dim;
      const float* xr = x.data() + r * in_dim;
      for (std::size_t j = 0; j < in_dim; ++j) s += xr[j] * w[j];
      out[r * out_dim + o] = s;
    }
  }
}

void moe_gate(std::span<const float> hidden, std::size_t n, std::size_t h,
              std::span<const float> weight, std::size_t num_experts,
              std::span<const float> bias, std::size_t top_k, float scale, bool renormalize,
              std::span<long long> topk_idx, std::span<float> topk_weight) {
  std::vector<float> scores(num_experts), choice(num_experts);
  for (std::size_t row = 0; row < n; ++row) {
    const float* x = hidden.data() + row * h;
    for (std::size_t e = 0; e < num_experts; ++e) {
      float logit = 0.f;
      const float* w = weight.data() + e * h;
      for (std::size_t j = 0; j < h; ++j) logit += x[j] * w[j];
      scores[e] = sigmoid(logit);
      choice[e] = scores[e] + bias[e];
    }
    std::vector<std::size_t> order(num_experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      if (choice[a] != choice[b]) return choice[a] > choice[b];
      return a < b;
    });
    std::vector<std::size_t> chosen(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_k));
    std::sort(chosen.begin(), chosen.end());
    for (std::size_t k = 0; k < top_k; ++k) {
      topk_idx[row * top_k + k] = static_cast<long long>(chosen[k]);
      topk_weight[row * top_k + k] = scores[chosen[k]];
    }
    if (top_k > 1 && renormalize) {
      float denom = 1e-20f;
      for (std::size_t k = 0; k < top_k; ++k) denom += topk_weight[row * top_k + k];
      for (std::size_t k = 0; k < top_k; ++k) topk_weight[row * top_k + k] /= denom;
    }
    for (std::size_t k = 0; k < top_k; ++k) topk_weight[row * top_k + k] *= scale;
  }
}

void expert_ffn(std::span<const float> x, std::size_t rows, std::size_t hidden,
                std::span<const float> w1, std::span<const float> w2, std::span<const float> w3,
                std::size_t inter, float beta, std::optional<float> linear_beta,
                std::span<float> out) {
  std::vector<float> gate(rows * inter), up(rows * inter), gate_up(rows * 2 * inter), act(rows * inter);
  linear(x, rows, hidden, w1, inter, gate);
  linear(x, rows, hidden, w3, inter, up);
  for (std::size_t r = 0; r < rows; ++r) {
    std::copy_n(gate.data() + r * inter, inter, gate_up.data() + r * 2 * inter);
    std::copy_n(up.data() + r * inter, inter, gate_up.data() + r * 2 * inter + inter);
  }
  situ_and_mul(gate_up, 2 * inter, beta, linear_beta, act);
  linear(act, rows, inter, w2, hidden, out);
}

void apply_attn_res(std::span<const float> prefix, std::size_t n, std::size_t hidden,
                    std::span<const float> block_residual, std::size_t num_blocks,
                    std::span<const float> proj, std::span<const float> norm, float eps,
                    std::span<float> out) {
  const std::size_t blocks = num_blocks + 1;
  std::vector<float> v(n * blocks * hidden), scores(n * blocks), probs(n * blocks);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t b = 0; b < num_blocks; ++b) {
      std::copy_n(block_residual.data() + (i * num_blocks + b) * hidden, hidden,
                  v.data() + (i * blocks + b) * hidden);
    }
    std::copy_n(prefix.data() + i * hidden, hidden, v.data() + (i * blocks + num_blocks) * hidden);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t b = 0; b < blocks; ++b) {
      const float* vv = v.data() + (i * blocks + b) * hidden;
      float var = 0.f;
      for (std::size_t j = 0; j < hidden; ++j) var += vv[j] * vv[j];
      var /= static_cast<float>(hidden);
      float inv = 1.f / std::sqrt(var + eps);
      float score = 0.f;
      for (std::size_t j = 0; j < hidden; ++j) score += vv[j] * inv * norm[j] * proj[j];
      scores[i * blocks + b] = score;
    }
  }
  stable_softmax_rows(scores.data(), n, blocks, probs.data());
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < hidden; ++j) {
      float acc = 0.f;
      for (std::size_t b = 0; b < blocks; ++b)
        acc += probs[i * blocks + b] * v[(i * blocks + b) * hidden + j];
      out[i * hidden + j] = acc;
    }
  }
}

void short_conv1d_silu(std::span<const float> x, std::size_t batch, std::size_t time,
                       std::size_t channels, std::span<const float> weight, std::size_t kernel,
                       std::span<float> out) {
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t t = 0; t < time; ++t) {
      for (std::size_t c = 0; c < channels; ++c) {
        float acc = 0.f;
        for (std::size_t i = 0; i < kernel; ++i) {
          int src = static_cast<int>(t) + static_cast<int>(i) - (static_cast<int>(kernel) - 1);
          float xv = src >= 0 ? x[(b * time + static_cast<std::size_t>(src)) * channels + c] : 0.f;
          acc += xv * weight[c * kernel + i];
        }
        out[(b * time + t) * channels + c] = acc * sigmoid(acc);
      }
    }
  }
}

void kda_lowerbound_gate(std::span<const float> g, std::size_t outer, std::size_t heads,
                         std::size_t dim, std::span<const float> a_log,
                         std::span<const float> dt_bias, float lower_bound, std::span<float> out) {
  std::vector<float> a_exp(heads);
  for (std::size_t h = 0; h < heads; ++h) a_exp[h] = std::exp(a_log[h]);
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t d = 0; d < dim; ++d) {
        std::size_t idx = (o * heads + h) * dim + d;
        float v = g[idx];
        if (!dt_bias.empty()) v += dt_bias[h * dim + d];
        out[idx] = lower_bound * sigmoid(a_exp[h] * v);
      }
    }
  }
}

void l2_normalize(std::span<const float> x, std::size_t rows, std::size_t dim, float eps,
                  std::span<float> out) {
  for (std::size_t r = 0; r < rows; ++r) {
    float s = 0.f;
    for (std::size_t j = 0; j < dim; ++j) {
      float v = x[r * dim + j];
      s += v * v;
    }
    float inv = 1.f / std::sqrt(s + eps);
    for (std::size_t j = 0; j < dim; ++j) out[r * dim + j] = x[r * dim + j] * inv;
  }
}

void kda_recurrent(std::span<const float> q, std::span<const float> k, std::span<const float> v,
                   std::span<const float> g, std::span<const float> beta, std::size_t batch,
                   std::size_t time, std::size_t heads, std::size_t dim, bool l2norm_qk,
                   std::span<float> out, std::span<float> final_state) {
  float scale = 1.f / std::sqrt(static_cast<float>(dim));
  std::vector<float> qq(q.begin(), q.end()), kk(k.begin(), k.end());
  if (l2norm_qk) {
    std::vector<float> tq(qq.size()), tk(kk.size());
    l2_normalize(qq, batch * time * heads, dim, 1e-6f, tq);
    l2_normalize(kk, batch * time * heads, dim, 1e-6f, tk);
    qq.swap(tq);
    kk.swap(tk);
  }
  for (float& x : qq) x *= scale;
  std::vector<float> s(batch * heads * dim * dim, 0.f);
  for (std::size_t t = 0; t < time; ++t) {
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t h = 0; h < heads; ++h) {
        std::size_t q_base = ((b * time + t) * heads + h) * dim;
        std::size_t s_base = (b * heads + h) * dim * dim;
        float bval = beta[(b * time + t) * heads + h];
        for (std::size_t kd = 0; kd < dim; ++kd) {
          float eg = std::exp(g[q_base + kd]);
          for (std::size_t vd = 0; vd < dim; ++vd) s[s_base + kd * dim + vd] *= eg;
        }
        std::vector<float> ks(dim), delta(dim);
        for (std::size_t vd = 0; vd < dim; ++vd) {
          float acc = 0.f;
          for (std::size_t kd = 0; kd < dim; ++kd) acc += kk[q_base + kd] * s[s_base + kd * dim + vd];
          ks[vd] = acc;
          delta[vd] = v[q_base + vd] - ks[vd];
        }
        for (std::size_t kd = 0; kd < dim; ++kd) {
          float bk = bval * kk[q_base + kd];
          for (std::size_t vd = 0; vd < dim; ++vd) s[s_base + kd * dim + vd] += bk * delta[vd];
        }
        for (std::size_t vd = 0; vd < dim; ++vd) {
          float acc = 0.f;
          for (std::size_t kd = 0; kd < dim; ++kd) acc += qq[q_base + kd] * s[s_base + kd * dim + vd];
          out[q_base + vd] = acc;
        }
      }
    }
  }
  std::copy(s.begin(), s.end(), final_state.begin());
}

void mla_eager_attention(std::span<const float> query, std::span<const float> key,
                         std::span<const float> value, std::size_t batch, std::size_t heads,
                         std::size_t time, std::size_t dim, float scaling, bool causal,
                         std::span<float> out) {
  std::vector<float> scores(batch * heads * time * time), probs(scores.size()), tmp(query.size());
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t qi = 0; qi < time; ++qi) {
        for (std::size_t kj = 0; kj < time; ++kj) {
          float s = 0.f;
          std::size_t qb = ((b * heads + h) * time + qi) * dim;
          std::size_t kb = ((b * heads + h) * time + kj) * dim;
          for (std::size_t d = 0; d < dim; ++d) s += query[qb + d] * key[kb + d];
          s *= scaling;
          if (causal && kj > qi) s = -1e9f;
          scores[(((b * heads + h) * time) + qi) * time + kj] = s;
        }
      }
    }
  }
  stable_softmax_rows(scores.data(), batch * heads * time, time, probs.data());
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t qi = 0; qi < time; ++qi) {
        for (std::size_t d = 0; d < dim; ++d) {
          float acc = 0.f;
          for (std::size_t kj = 0; kj < time; ++kj) {
            float p = probs[(((b * heads + h) * time) + qi) * time + kj];
            acc += p * value[((b * heads + h) * time + kj) * dim + d];
          }
          tmp[((b * heads + h) * time + qi) * dim + d] = acc;
        }
      }
    }
  }
  for (std::size_t b = 0; b < batch; ++b)
    for (std::size_t t = 0; t < time; ++t)
      for (std::size_t h = 0; h < heads; ++h)
        for (std::size_t d = 0; d < dim; ++d)
          out[((b * time + t) * heads + h) * dim + d] = tmp[((b * heads + h) * time + t) * dim + d];
}

void gated_mla_output(std::span<const float> attn, std::span<const float> gate_logits,
                      std::span<float> out) {
  for (std::size_t i = 0; i < attn.size(); ++i) out[i] = attn[i] * sigmoid(gate_logits[i]);
}

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
                        std::optional<float> linear_beta, std::span<float> out) {
  const std::size_t n = batch * seq;
  std::vector<long long> idx(n * top_k);
  std::vector<float> wt(n * top_k), latent_x(n * latent), y(n * latent), routed(n * hidden_dim),
      shared(n * hidden_dim);
  moe_gate(hidden, n, hidden_dim, router_w, num_experts, router_b, top_k, 1.f, true, idx, wt);
  linear(hidden, n, hidden_dim, routed_down, latent, latent_x);
  for (std::size_t t = 0; t < n; ++t) {
    std::vector<float> acc(latent, 0.f), tmp(latent);
    auto xt = std::span<const float>(latent_x.data() + t * latent, latent);
    for (std::size_t k = 0; k < top_k; ++k) {
      auto e = static_cast<std::size_t>(idx[t * top_k + k]);
      expert_ffn(xt, 1, latent, ew1[e], ew2[e], ew3[e], inter, beta, linear_beta, tmp);
      for (std::size_t j = 0; j < latent; ++j) acc[j] += wt[t * top_k + k] * tmp[j];
    }
    std::copy(acc.begin(), acc.end(), y.begin() + static_cast<std::ptrdiff_t>(t * latent));
  }
  if (!routed_norm.empty()) {
    std::vector<float> yn(y.size());
    rmsnorm(y, routed_norm, rms_eps, yn);
    y.swap(yn);
  }
  linear(y, n, latent, routed_up, hidden_dim, routed);
  expert_ffn(hidden, n, hidden_dim, shared_gate, shared_down, shared_up, inter, beta, linear_beta,
             shared);
  for (std::size_t i = 0; i < n * hidden_dim; ++i) out[i] = routed[i] + shared[i];
}

void decoder_block_forward(std::span<const float> hidden, std::size_t batch, std::size_t seq,
                           std::size_t hidden_dim, std::span<const float> block_residual,
                           std::size_t num_blocks, std::span<const float> input_norm,
                           std::span<const float> post_norm, std::span<const float> sa_proj,
                           std::span<const float> sa_norm, std::span<const float> mlp_proj,
                           std::span<const float> mlp_norm, std::size_t layer_idx,
                           std::size_t block_size, float attn_scale, float mlp_scale, float rms_eps,
                           std::span<float> prefix_out, std::vector<float>& br_out,
                           std::size_t& nb_out) {
  const std::size_t n = batch * seq;
  std::vector<float> hidden_buf(hidden.begin(), hidden.end());
  std::optional<std::vector<float>> prefix = std::vector<float>(hidden.begin(), hidden.end());
  std::vector<float> br(block_residual.begin(), block_residual.end());
  std::size_t nb = num_blocks;
  if (nb > 0) {
    std::vector<float> tmp(n * hidden_dim);
    apply_attn_res(*prefix, n, hidden_dim, br, nb, sa_proj, sa_norm, rms_eps, tmp);
    hidden_buf.swap(tmp);
  }
  if (layer_idx % block_size == 0) {
    br.insert(br.end(), prefix->begin(), prefix->end());
    ++nb;
    prefix.reset();
  }
  std::vector<float> normed(n * hidden_dim);
  rmsnorm(hidden_buf, input_norm, rms_eps, normed);
  for (float& x : normed) x *= attn_scale;
  std::vector<float> prefix_sum;
  if (prefix) {
    prefix_sum = *prefix;
    for (std::size_t i = 0; i < n * hidden_dim; ++i) prefix_sum[i] += normed[i];
  } else {
    prefix_sum = normed;
  }
  std::vector<float> hidden2(n * hidden_dim), post(n * hidden_dim);
  apply_attn_res(prefix_sum, n, hidden_dim, br, nb, mlp_proj, mlp_norm, rms_eps, hidden2);
  rmsnorm(hidden2, post_norm, rms_eps, post);
  for (float& x : post) x *= mlp_scale;
  for (std::size_t i = 0; i < n * hidden_dim; ++i) prefix_sum[i] += post[i];
  std::copy(prefix_sum.begin(), prefix_sum.end(), prefix_out.begin());
  br_out = std::move(br);
  nb_out = nb;
}

}  // namespace kimi::k3
