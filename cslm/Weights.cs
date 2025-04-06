namespace cslm
{
    public struct Weights
    {
        public int dbits_; // 4 for gf4, 8 for fp8, 16 for fp16; determines type of byte[,] below

        // token embedding table
        public byte[] token_embedding_table_; // (vocab_size, dim)
                                              // weights for norms
        public float[,] rms_att_weight_; // (dim) rmsnorm weights
        public float[,] rms_ffn_weight_; // (dim)
                                         // weights for matmuls
        public byte[,] wq_; // (n_heads * head_dim, dim)
        public byte[,] wk_; // (n_kv_heads * head_dim, dim)
        public byte[,] wv_; // (n_kv_heads * head_dim, dim)
        public byte[,] wo_; // (dim, n_heads * head_dim)
                            // weights for ffn
        public byte[,] w1_; // (n_experts?, hidden_dim, dim)
        public byte[,] w2_; // (n_experts?, dim, hidden_dim)
        public byte[,] w3_; // (n_experts?, hidden_dim, dim)
                            // final norm
        public float[] rms_final_weight_; // (dim,)
                                          // classifier weights for the logits, on the last layer
        public byte[] wcls_;
        // biases for qkv (qwen)
        public float[,] bqkv_; // ((n_heads + n_kv_heads * 2) * head_dim)
                               // moe gate weights (mixtral)
        public byte[,] moegate_; // (n_experts, dim)
    };

}
