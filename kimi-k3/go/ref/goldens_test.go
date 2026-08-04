package ref_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/lajamerrmittesdine/salesexampleproject/kimi-k3/go/ref"
)

type root struct {
	Cases map[string]map[string]any `json:"cases"`
}

func load(t *testing.T) root {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	p := filepath.Join(filepath.Dir(file), "..", "..", "fixtures", "goldens.json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	var r root
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func f32s(v any) []float32 {
	arr := v.([]any)
	out := make([]float32, len(arr))
	for i, x := range arr {
		out[i] = float32(x.(float64))
	}
	return out
}

func i64s(v any) []int64 {
	arr := v.([]any)
	out := make([]int64, len(arr))
	for i, x := range arr {
		out[i] = int64(x.(float64))
	}
	return out
}

func ints(v any) []int {
	arr := v.([]any)
	out := make([]int, len(arr))
	for i, x := range arr {
		out[i] = int(x.(float64))
	}
	return out
}

func assertClose(t *testing.T, a, b []float32, atol, rtol float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("len %d != %d", len(a), len(b))
	}
	for i := range a {
		tol := atol + rtol*float32(math.Abs(float64(b[i])))
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			t.Fatalf("mismatch at %d: got %v want %v tol %v", i, a[i], b[i], tol)
		}
	}
}

func TestRMSNorm(t *testing.T) {
	c := load(t).Cases["rmsnorm"]
	y := ref.RMSNorm(f32s(c["x"]), f32s(c["weight"]), float32(c["eps"].(float64)))
	assertClose(t, y, f32s(c["y"]), 1e-5, 1e-5)
}

func TestSitu(t *testing.T) {
	c := load(t).Cases["situ"]
	shape := ints(c["shape"])
	lb := float32(25)
	y := ref.SituAndMul(f32s(c["x"]), shape[len(shape)-1], 4, &lb)
	assertClose(t, y, f32s(c["y"]), 1e-5, 1e-5)
}

func TestMoEGate(t *testing.T) {
	c := load(t).Cases["moe_gate"]
	hs := ints(c["hidden_shape"])
	n := hs[0] * hs[1]
	idx, wt := ref.MoEGate(f32s(c["hidden"]), n, hs[2], f32s(c["weight"]), int(c["num_experts"].(float64)), f32s(c["bias"]), int(c["top_k"].(float64)), 1, true)
	want := i64s(c["topk_idx"])
	for i := range idx {
		if idx[i] != want[i] {
			t.Fatalf("idx %v want %v", idx, want)
		}
	}
	assertClose(t, wt, f32s(c["topk_weight"]), 1e-5, 1e-5)
}

func TestExpertFFN(t *testing.T) {
	c := load(t).Cases["expert_ffn"]
	xs := ints(c["x_shape"])
	w1s := ints(c["w1_shape"])
	lb := float32(25)
	y := ref.ExpertFFN(f32s(c["x"]), xs[0], xs[1], f32s(c["w1"]), f32s(c["w2"]), f32s(c["w3"]), w1s[0], 4, &lb)
	assertClose(t, y, f32s(c["y"]), 1e-4, 1e-4)
}

func TestAttnRes(t *testing.T) {
	c := load(t).Cases["attn_res"]
	n := int(c["n"].(float64))
	h := int(c["hidden"].(float64))
	nb := int(c["num_blocks"].(float64))
	y := ref.ApplyAttnRes(f32s(c["prefix_sum"]), n, h, f32s(c["block_residual"]), nb, f32s(c["proj_weight"]), f32s(c["norm_weight"]), 1e-5)
	assertClose(t, y, f32s(c["y"]), 1e-5, 1e-5)
}

func TestShortConv(t *testing.T) {
	c := load(t).Cases["short_conv"]
	xs := ints(c["x_shape"])
	ws := ints(c["weight_shape"])
	y := ref.ShortConv1dSiLU(f32s(c["x"]), xs[0], xs[1], xs[2], f32s(c["weight"]), ws[1])
	assertClose(t, y, f32s(c["y"]), 1e-5, 1e-5)
}

func TestKDAGate(t *testing.T) {
	c := load(t).Cases["kda_gate"]
	gs := ints(c["g_shape"])
	y := ref.KDALowerBoundGate(f32s(c["g"]), gs[0]*gs[1], gs[2], gs[3], f32s(c["a_log"]), f32s(c["dt_bias"]), -5)
	assertClose(t, y, f32s(c["y"]), 1e-5, 1e-5)
}

func TestKDARecurrent(t *testing.T) {
	c := load(t).Cases["kda_recurrent"]
	qs := ints(c["q_shape"])
	b, tt, h, d := qs[0], qs[1], qs[2], qs[3]
	g := ref.KDALowerBoundGate(f32s(c["g_raw"]), b*tt, h, d, f32s(c["a_log"]), f32s(c["dt_bias"]), -5)
	o, s := ref.KDARecurrent(f32s(c["q"]), f32s(c["k"]), f32s(c["v"]), g, f32s(c["beta"]), b, tt, h, d, nil, true)
	assertClose(t, o, f32s(c["o"]), 1e-4, 1e-4)
	assertClose(t, s, f32s(c["final_state"]), 1e-4, 1e-4)
}

func TestMLAEager(t *testing.T) {
	c := load(t).Cases["mla_eager"]
	qs := ints(c["q_shape"])
	o := ref.MLAEagerAttention(f32s(c["q"]), f32s(c["k"]), f32s(c["v"]), qs[0], qs[1], qs[2], qs[3], float32(c["scaling"].(float64)), true)
	assertClose(t, o, f32s(c["attn_out"]), 1e-4, 1e-4)
	gated := ref.GatedMLAOutput(o, f32s(c["gate_logits"]))
	assertClose(t, gated, f32s(c["gated"]), 1e-4, 1e-4)
}

func TestLatentMoE(t *testing.T) {
	c := load(t).Cases["latent_moe"]
	hs := ints(c["hidden_shape"])
	ne := int(c["num_experts"].(float64))
	latent := int(c["latent"].(float64))
	inter := int(c["inter"].(float64))
	ew1 := make([][]float32, ne)
	ew2 := make([][]float32, ne)
	ew3 := make([][]float32, ne)
	for i := 0; i < ne; i++ {
		ew1[i] = f32s(c["experts_w1"].([]any)[i])
		ew2[i] = f32s(c["experts_w2"].([]any)[i])
		ew3[i] = f32s(c["experts_w3"].([]any)[i])
	}
	lb := float32(25)
	y := ref.LatentMoEForward(
		f32s(c["hidden"]), hs[0], hs[1], hs[2],
		f32s(c["router_weight"]), f32s(c["router_bias"]), ne, int(c["top_k"].(float64)),
		ew1, ew2, ew3, latent, inter,
		f32s(c["routed_down"]), f32s(c["routed_up"]), f32s(c["routed_norm"]),
		f32s(c["shared_gate"]), f32s(c["shared_up"]), f32s(c["shared_down"]),
		1e-5, 4, &lb,
	)
	assertClose(t, y, f32s(c["y"]), 1e-4, 1e-4)
}

func TestDecoderBlock(t *testing.T) {
	c := load(t).Cases["decoder_block"]
	hs := ints(c["hidden_shape"])
	brs := ints(c["block_residual_shape"])
	ps, br, _ := ref.DecoderBlockForward(
		f32s(c["hidden"]), hs[0], hs[1], hs[2],
		f32s(c["block_residual"]), brs[1],
		f32s(c["input_norm"]), f32s(c["post_norm"]),
		f32s(c["self_attn_res_proj"]), f32s(c["self_attn_res_norm"]),
		f32s(c["mlp_res_proj"]), f32s(c["mlp_res_norm"]),
		int(c["layer_idx"].(float64)), int(c["attn_res_block_size"].(float64)),
		float32(c["attn_scale"].(float64)), float32(c["mlp_scale"].(float64)), 1e-5,
	)
	assertClose(t, ps, f32s(c["prefix_sum"]), 1e-4, 1e-4)
	assertClose(t, br, f32s(c["out_block_residual"]), 1e-4, 1e-4)
}

func TestL2Normalize(t *testing.T) {
	c := load(t).Cases["l2_normalize"]
	xs := ints(c["x_shape"])
	dim := xs[len(xs)-1]
	rows := 1
	for _, v := range xs {
		rows *= v
	}
	rows /= dim
	y := ref.L2Normalize(f32s(c["x"]), rows, dim, 1e-6)
	assertClose(t, y, f32s(c["y"]), 1e-5, 1e-5)
}
