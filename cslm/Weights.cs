using System.Runtime.InteropServices;

namespace cslm
{
    public struct Weights
    {
        public struct Weight
        {
            public int offset_;
            public int size_;
        }

        public Span<T> AsSpan<T>(Weight weight) where T:struct
        {
            return MemoryMarshal.Cast<byte,T>(bytes_.AsSpan<byte>(weight.offset_, weight.size_));
        }

		public Span<T> AsSpan<T>(Weight weight, int offset) where T : struct
		{
			return MemoryMarshal.Cast<byte, T>(bytes_.AsSpan<byte>(weight.offset_+ offset, weight.size_));
		}

		public byte[] bytes_;

        public int dbits_; // 4 for gf4, 8 for fp8, 16 for fp16; determines type of byte[,] below

        // token embedding table
        public Weight token_embedding_table_; // (vocab_size, dim)
                                              // weights for norms
        public Weight[] rms_att_weight_; // (dim) rmsnorm weights
        public Weight[] rms_ffn_weight_; // (dim)
                                         // weights for matmuls
        public Weight[] wq_; // (n_heads * head_dim, dim)
        public Weight[] wk_; // (n_kv_heads * head_dim, dim)
        public Weight[] wv_; // (n_kv_heads * head_dim, dim)
        public Weight[] wo_; // (dim, n_heads * head_dim)
                            // weights for ffn
        public Weight[] w1_; // (n_experts?, hidden_dim, dim)
        public Weight[] w2_; // (n_experts?, dim, hidden_dim)
        public Weight[] w3_; // (n_experts?, hidden_dim, dim)
                            // final norm
        public Weight rms_final_weight_; // (dim,)
                                          // classifier weights for the logits, on the last layer
        public Weight wcls_;
        // biases for qkv (qwen)
        public Weight[] bqkv_; // ((n_heads + n_kv_heads * 2) * head_dim)
                               // moe gate weights (mixtral)
        public Weight[] moegate_; // (n_experts, dim)
	 };

}
