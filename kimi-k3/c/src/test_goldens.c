#include "kimi_k3_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../fixtures/goldens_embedded.h"

static int g_fails;

static void assert_close(const float *a, const float *b, size_t n, float atol, float rtol,
                         const char *name) {
  for (size_t i = 0; i < n; ++i) {
    float tol = atol + rtol * fabsf(b[i]);
    if (fabsf(a[i] - b[i]) > tol) {
      fprintf(stderr, "FAIL %s at %zu: got %g want %g tol %g\n", name, i, a[i], b[i], tol);
      ++g_fails;
      return;
    }
  }
  printf("ok %s\n", name);
}

int main(void) {
  {
    float *y = malloc(g_rmsnorm_x_len * sizeof(float));
    size_t rows = g_rmsnorm_shape[0] * g_rmsnorm_shape[1];
    k3_rmsnorm(g_rmsnorm_x, g_rmsnorm_w, rows, g_rmsnorm_shape[2], g_rmsnorm_eps, y);
    assert_close(y, g_rmsnorm_y, g_rmsnorm_y_len, 1e-5f, 1e-5f, "rmsnorm");
    free(y);
  }
  {
    float *y = malloc((g_situ_x_len / 2) * sizeof(float));
    size_t rows = g_situ_shape[0] * g_situ_shape[1];
    k3_situ_and_mul(g_situ_x, rows, g_situ_shape[2], 4.f, 1, 25.f, y);
    assert_close(y, g_situ_y, g_situ_y_len, 1e-5f, 1e-5f, "situ");
    free(y);
  }
  {
    size_t n = g_moe_B * g_moe_S;
    long long *idx = malloc(n * g_moe_K * sizeof(long long));
    float *wt = malloc(n * g_moe_K * sizeof(float));
    k3_moe_gate(g_moe_hidden, n, g_moe_H, g_moe_weight, g_moe_E, g_moe_bias, g_moe_K, 1.f, 1, idx,
                wt);
    for (size_t i = 0; i < g_moe_idx_len; ++i) {
      if (idx[i] != g_moe_idx[i]) {
        fprintf(stderr, "FAIL moe idx\n");
        ++g_fails;
        break;
      }
    }
    assert_close(wt, g_moe_wt, g_moe_wt_len, 1e-5f, 1e-5f, "moe_gate");
    free(idx);
    free(wt);
  }
  {
    float *y = malloc(g_effn_rows * g_effn_h * sizeof(float));
    k3_expert_ffn(g_effn_x, g_effn_rows, g_effn_h, g_effn_w1, g_effn_w2, g_effn_w3, g_effn_inter, 4.f,
                  1, 25.f, y);
    assert_close(y, g_effn_y, g_effn_y_len, 1e-4f, 1e-4f, "expert_ffn");
    free(y);
  }
  {
    float *y = malloc(g_ar_n * g_ar_h * sizeof(float));
    k3_apply_attn_res(g_ar_prefix, g_ar_n, g_ar_h, g_ar_block, g_ar_nb, g_ar_proj, g_ar_norm, 1e-5f,
                      y);
    assert_close(y, g_ar_y, g_ar_y_len, 1e-5f, 1e-5f, "attn_res");
    free(y);
  }
  {
    float *y = malloc(g_sc_x_len * sizeof(float));
    k3_short_conv1d_silu(g_sc_x, g_sc_b, g_sc_t, g_sc_c, g_sc_w, g_sc_k, y);
    assert_close(y, g_sc_y, g_sc_y_len, 1e-5f, 1e-5f, "short_conv");
    free(y);
  }
  {
    float *y = malloc(g_kg_g_len * sizeof(float));
    k3_kda_lowerbound_gate(g_kg_g, g_kg_B * g_kg_T, g_kg_H, g_kg_D, g_kg_alog, g_kg_dt, -5.f, y);
    assert_close(y, g_kg_y, g_kg_y_len, 1e-5f, 1e-5f, "kda_gate");
    free(y);
  }
  {
    float *g = malloc(g_kda_g_raw_len * sizeof(float));
    float *o = malloc(g_kda_o_len * sizeof(float));
    float *s = malloc(g_kda_final_state_len * sizeof(float));
    k3_kda_lowerbound_gate(g_kda_g_raw, g_kda_B * g_kda_T, g_kda_H, g_kda_D, g_kda_a_log,
                           g_kda_dt_bias, -5.f, g);
    k3_kda_recurrent(g_kda_q, g_kda_k, g_kda_v, g, g_kda_beta, g_kda_B, g_kda_T, g_kda_H, g_kda_D, 1,
                     o, s);
    assert_close(o, g_kda_o, g_kda_o_len, 1e-4f, 1e-4f, "kda_recurrent_o");
    assert_close(s, g_kda_final_state, g_kda_final_state_len, 1e-4f, 1e-4f, "kda_recurrent_s");
    free(g);
    free(o);
    free(s);
  }
  {
    float *o = malloc(g_mla_attn_out_len * sizeof(float));
    float *gated = malloc(g_mla_gated_len * sizeof(float));
    k3_mla_eager_attention(g_mla_q, g_mla_k, g_mla_v, g_mla_B, g_mla_H, g_mla_T, g_mla_D,
                           g_mla_scale, 1, o);
    assert_close(o, g_mla_attn_out, g_mla_attn_out_len, 1e-4f, 1e-4f, "mla_eager");
    k3_gated_mla_output(o, g_mla_gate_logits, g_mla_gated_len, gated);
    assert_close(gated, g_mla_gated, g_mla_gated_len, 1e-4f, 1e-4f, "mla_gated");
    free(o);
    free(gated);
  }
  {
    const float *w1s[] = {g_lm_e0_w1, g_lm_e1_w1, g_lm_e2_w1, g_lm_e3_w1};
    const float *w2s[] = {g_lm_e0_w2, g_lm_e1_w2, g_lm_e2_w2, g_lm_e3_w2};
    const float *w3s[] = {g_lm_e0_w3, g_lm_e1_w3, g_lm_e2_w3, g_lm_e3_w3};
    float *y = malloc(g_lm_y_len * sizeof(float));
    k3_latent_moe_forward(g_lm_hidden, g_lm_B, g_lm_S, g_lm_H, g_lm_rw, g_lm_rb, g_lm_E, g_lm_K, w1s,
                          w2s, w3s, g_lm_L, g_lm_I, g_lm_down, g_lm_up, g_lm_norm, g_lm_sg, g_lm_su,
                          g_lm_sd, 1e-5f, 4.f, 1, 25.f, y);
    assert_close(y, g_lm_y, g_lm_y_len, 1e-4f, 1e-4f, "latent_moe");
    free(y);
  }
  {
    float *prefix = malloc(g_db_ps_len * sizeof(float));
    float *br = malloc((g_db_obr_len + g_db_B * g_db_S * g_db_H) * sizeof(float));
    size_t br_len = 0;
    k3_decoder_block_forward(g_db_hidden, g_db_B, g_db_S, g_db_H, g_db_br, g_db_NB, g_db_in,
                             g_db_post, g_db_sap, g_db_san, g_db_mrp, g_db_mrn, g_db_layer, g_db_bs,
                             g_db_attn, g_db_mlp, 1e-5f, prefix, br, &br_len);
    assert_close(prefix, g_db_ps, g_db_ps_len, 1e-4f, 1e-4f, "decoder_prefix");
    assert_close(br, g_db_obr, g_db_obr_len, 1e-4f, 1e-4f, "decoder_br");
    free(prefix);
    free(br);
  }
  {
    size_t dim = g_l2_shape[2];
    size_t rows = g_l2_x_len / dim;
    float *y = malloc(g_l2_x_len * sizeof(float));
    k3_l2_normalize(g_l2_x, rows, dim, 1e-6f, y);
    assert_close(y, g_l2_y, g_l2_y_len, 1e-5f, 1e-5f, "l2_normalize");
    free(y);
  }

  if (g_fails) {
    fprintf(stderr, "%d failures\n", g_fails);
    return 1;
  }
  puts("all c goldens passed");
  return 0;
}
