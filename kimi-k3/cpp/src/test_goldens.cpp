#include "kimi_k3_ref.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include "../../fixtures/goldens_embedded.h"

using namespace kimi::k3;

static int g_fails = 0;

static void assert_close(const float* a, const float* b, size_t n, float atol, float rtol,
                         const char* name) {
  for (size_t i = 0; i < n; ++i) {
    float tol = atol + rtol * std::fabs(b[i]);
    if (std::fabs(a[i] - b[i]) > tol) {
      std::fprintf(stderr, "FAIL %s at %zu: got %g want %g tol %g\n", name, i, a[i], b[i], tol);
      ++g_fails;
      return;
    }
  }
  std::printf("ok %s\n", name);
}

int main() {
  {
    std::vector<float> y(g_rmsnorm_x_len);
    rmsnorm(g_rmsnorm_x, g_rmsnorm_w, g_rmsnorm_eps, y);
    assert_close(y.data(), g_rmsnorm_y, g_rmsnorm_y_len, 1e-5f, 1e-5f, "rmsnorm");
  }
  {
    std::vector<float> y(g_situ_x_len / 2);
    situ_and_mul(g_situ_x, g_situ_shape[2], 4.f, 25.f, y);
    assert_close(y.data(), g_situ_y, g_situ_y_len, 1e-5f, 1e-5f, "situ");
  }
  {
    size_t n = g_moe_B * g_moe_S;
    std::vector<long long> idx(n * g_moe_K);
    std::vector<float> wt(n * g_moe_K);
    moe_gate(g_moe_hidden, n, g_moe_H, g_moe_weight, g_moe_E, g_moe_bias, g_moe_K, 1.f, true, idx,
             wt);
    for (size_t i = 0; i < g_moe_idx_len; ++i) {
      if (idx[i] != g_moe_idx[i]) {
        std::fprintf(stderr, "FAIL moe_gate idx\n");
        ++g_fails;
        break;
      }
    }
    assert_close(wt.data(), g_moe_wt, g_moe_wt_len, 1e-5f, 1e-5f, "moe_gate");
  }
  {
    std::vector<float> y(g_effn_rows * g_effn_h);
    expert_ffn(g_effn_x, g_effn_rows, g_effn_h, g_effn_w1, g_effn_w2, g_effn_w3, g_effn_inter, 4.f,
               25.f, y);
    assert_close(y.data(), g_effn_y, g_effn_y_len, 1e-4f, 1e-4f, "expert_ffn");
  }
  {
    std::vector<float> y(g_ar_n * g_ar_h);
    apply_attn_res(g_ar_prefix, g_ar_n, g_ar_h, g_ar_block, g_ar_nb, g_ar_proj, g_ar_norm, 1e-5f, y);
    assert_close(y.data(), g_ar_y, g_ar_y_len, 1e-5f, 1e-5f, "attn_res");
  }
  {
    std::vector<float> y(g_sc_x_len);
    short_conv1d_silu(g_sc_x, g_sc_b, g_sc_t, g_sc_c, g_sc_w, g_sc_k, y);
    assert_close(y.data(), g_sc_y, g_sc_y_len, 1e-5f, 1e-5f, "short_conv");
  }
  {
    std::vector<float> y(g_kg_g_len);
    kda_lowerbound_gate(g_kg_g, g_kg_B * g_kg_T, g_kg_H, g_kg_D, g_kg_alog, g_kg_dt, -5.f, y);
    assert_close(y.data(), g_kg_y, g_kg_y_len, 1e-5f, 1e-5f, "kda_gate");
  }
  {
    std::vector<float> g(g_kda_g_raw_len), o(g_kda_o_len), s(g_kda_final_state_len);
    kda_lowerbound_gate(g_kda_g_raw, g_kda_B * g_kda_T, g_kda_H, g_kda_D, g_kda_a_log, g_kda_dt_bias,
                        -5.f, g);
    kda_recurrent(g_kda_q, g_kda_k, g_kda_v, g, g_kda_beta, g_kda_B, g_kda_T, g_kda_H, g_kda_D, true,
                  o, s);
    assert_close(o.data(), g_kda_o, g_kda_o_len, 1e-4f, 1e-4f, "kda_recurrent_o");
    assert_close(s.data(), g_kda_final_state, g_kda_final_state_len, 1e-4f, 1e-4f, "kda_recurrent_s");
  }
  {
    std::vector<float> o(g_mla_attn_out_len), gated(g_mla_gated_len);
    mla_eager_attention(g_mla_q, g_mla_k, g_mla_v, g_mla_B, g_mla_H, g_mla_T, g_mla_D, g_mla_scale,
                        true, o);
    assert_close(o.data(), g_mla_attn_out, g_mla_attn_out_len, 1e-4f, 1e-4f, "mla_eager");
    gated_mla_output(o, g_mla_gate_logits, gated);
    assert_close(gated.data(), g_mla_gated, g_mla_gated_len, 1e-4f, 1e-4f, "mla_gated");
  }
  {
    std::vector<std::vector<float>> ew1(g_lm_E), ew2(g_lm_E), ew3(g_lm_E);
    const float* w1s[] = {g_lm_e0_w1, g_lm_e1_w1, g_lm_e2_w1, g_lm_e3_w1};
    const float* w2s[] = {g_lm_e0_w2, g_lm_e1_w2, g_lm_e2_w2, g_lm_e3_w2};
    const float* w3s[] = {g_lm_e0_w3, g_lm_e1_w3, g_lm_e2_w3, g_lm_e3_w3};
    size_t w1n = g_lm_I * g_lm_L;
    for (size_t e = 0; e < g_lm_E; ++e) {
      ew1[e].assign(w1s[e], w1s[e] + w1n);
      ew2[e].assign(w2s[e], w2s[e] + w1n);
      ew3[e].assign(w3s[e], w3s[e] + w1n);
    }
    std::vector<float> y(g_lm_y_len);
    latent_moe_forward(g_lm_hidden, g_lm_B, g_lm_S, g_lm_H, g_lm_rw, g_lm_rb, g_lm_E, g_lm_K, ew1,
                       ew2, ew3, g_lm_L, g_lm_I, g_lm_down, g_lm_up, g_lm_norm, g_lm_sg, g_lm_su,
                       g_lm_sd, 1e-5f, 4.f, 25.f, y);
    assert_close(y.data(), g_lm_y, g_lm_y_len, 1e-4f, 1e-4f, "latent_moe");
  }
  {
    std::vector<float> prefix(g_db_ps_len);
    std::vector<float> br;
    size_t nb = 0;
    // Use explicit lengths: empty block_residual arrays still occupy a C placeholder element.
    decoder_block_forward(std::span<const float>(g_db_hidden, g_db_hidden_len), g_db_B, g_db_S,
                          g_db_H, std::span<const float>(g_db_br, g_db_br_len), g_db_NB,
                          std::span<const float>(g_db_in, g_db_in_len),
                          std::span<const float>(g_db_post, g_db_post_len),
                          std::span<const float>(g_db_sap, g_db_sap_len),
                          std::span<const float>(g_db_san, g_db_san_len),
                          std::span<const float>(g_db_mrp, g_db_mrp_len),
                          std::span<const float>(g_db_mrn, g_db_mrn_len), g_db_layer, g_db_bs,
                          g_db_attn, g_db_mlp, 1e-5f, prefix, br, nb);
    assert_close(prefix.data(), g_db_ps, g_db_ps_len, 1e-4f, 1e-4f, "decoder_prefix");
    assert_close(br.data(), g_db_obr, g_db_obr_len, 1e-4f, 1e-4f, "decoder_br");
  }
  {
    size_t dim = g_l2_shape[2];
    size_t rows = g_l2_x_len / dim;
    std::vector<float> y(g_l2_x_len);
    l2_normalize(g_l2_x, rows, dim, 1e-6f, y);
    assert_close(y.data(), g_l2_y, g_l2_y_len, 1e-5f, 1e-5f, "l2_normalize");
  }

  if (g_fails) {
    std::fprintf(stderr, "%d failures\n", g_fails);
    return 1;
  }
  std::puts("all cpp goldens passed");
  return 0;
}
