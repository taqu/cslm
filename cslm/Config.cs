using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace cslm
{
    public struct Config
    {
        public int dim_;           // transformer dimension
        public int hidden_dim_;    // for ffn layers
        public int head_dim_;      // for attention heads; usually dim / n_heads
        public int n_layers_;      // number of layers
        public int n_heads_;       // number of query heads
        public int n_kv_heads_;    // number of key/value heads (can be < query heads because of multiquery)
        public int vocab_size_;    // vocabulary size, usually 256 (byte-level)
        public int seq_len_;       // max sequence length
        public float rope_theta_;  // RoPE theta
        public int rotary_dim_;    // RoPE rotary dimension (elements after that don't get rotated)
        public int n_experts_;     // number of experts for MoE models
        public int n_experts_ac_;  // number of active experts for MoE models
        public float norm_eps_;    // epsilon for layer normalization
        public bool act_gelu_;     // use GELU activation function
        public bool norm_ln_;      // use full LN normalization
        public bool norm_par_;     // use parallel MLP/attention by omitting intermediate normalization
        public float qkv_clip_;    // clip qkv values to [-clip, clip]
    }

    public enum ForwardFlags
    {
        FF_UPDATE_KV_ONLY = 1 << 0, // only update kv cache and don't output logits
    };
}
