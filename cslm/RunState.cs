
namespace cslm
{
    public struct RunState
    {
        // current wave of activations
        public float[] x_;      // activation at current time stamp (dim,)
        public float[] xb_;     // same, but inside a residual branch (dim,)
        public float[] xb2_;    // an additional buffer just for convenience (dim,)
        public float[] hb_;     // buffer for hidden dimension in the ffn (hidden_dim,)
        public float[] hb2_;    // buffer for hidden dimension in the ffn (hidden_dim,)
        public float[] he_;     // buffer for hidden dimension in the ffn (n_experts_ac,hidden_dim,)
        public float[] q_;      // query (dim,)
        public float[] k_;      // key (dim,)
        public float[] v_;      // value (dim,)
        public float[] att_;    // buffer for scores/attention values (n_heads, seq_len)
        public float[] exp_;    // buffer for MoE computations (n_experts + n_experts_ac * 2)
        public float[] logits_; // output logits
                                // kv cache
        public int kvbits_;        // 8 for fp8, 16 for fp16_; determines type of void* below
        public byte[] key_cache_;   // (layer, seq_len, dim)
        public byte[] value_cache_; // (layer, seq_len, dim)

        public void Initialize(in Config config)
        {
            int q_dim = config.head_dim_ * config.n_heads_;
            int kv_dim = config.head_dim_ * config.n_kv_heads_;
            x_ = new float[q_dim];
            xb_ = new float[q_dim];
            xb2_ = new float[q_dim];

            hb_ = new float[config.hidden_dim_];
            hb2_ = new float[config.hidden_dim_];

            q_ = new float[q_dim];
            k_ = new float[kv_dim];
            v_ = new float[kv_dim];

            att_ = new float[config.n_heads_ * config.seq_len_];
            exp_ = new float[config.n_experts_ + (0 != config.n_experts_ac_ ? config.n_experts_ac_ : 1) * 2];
            logits_ = new float[config.vocab_size_];
            //assert(s->kvbits == sizeof(kvtype_t) * 8);

            long cache_size = (long)config.n_layers_ * config.seq_len_ * kv_dim * sizeof(short);
            key_cache_ = new byte[cache_size];
            value_cache_ = new byte[cache_size];
        }
    }
}
