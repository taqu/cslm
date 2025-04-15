using System.Runtime.InteropServices;

namespace cslm
{
    public struct Weights
    {
        public Span<T> AsSpan<T>(Tensor tensor) where T:struct
        {
            return MemoryMarshal.Cast<byte,T>(bytes_.AsSpan<byte>((int)tensor.data_, (int)tensor.size_));
        }

		public Span<T> AsSpan<T>(Tensor tensor, int offset) where T : struct
		{
			return MemoryMarshal.Cast<byte, T>(bytes_.AsSpan<byte>((int)tensor.data_+ offset, (int)tensor.size_));
		}

		public byte[] bytes_;

        public int dbits_; // 4 for gf4, 8 for fp8, 16 for fp16; determines type of byte[,] below

        // token embedding table
        public Tensor token_embedding_table_; // (vocab_size, dim)
                                              // weights for norms
        public Tensor[] rms_att_weight_; // (dim) rmsnorm weights
        public Tensor[] rms_ffn_weight_; // (dim)
                                         // weights for matmuls
        public Tensor[] wq_; // (n_heads * head_dim, dim)
        public Tensor[] wk_; // (n_kv_heads * head_dim, dim)
        public Tensor[] wv_; // (n_kv_heads * head_dim, dim)
        public Tensor[] wo_; // (dim, n_heads * head_dim)
                            // weights for ffn
        public Tensor[] w1_; // (n_experts?, hidden_dim, dim)
        public Tensor[] w2_; // (n_experts?, dim, hidden_dim)
        public Tensor[] w3_; // (n_experts?, hidden_dim, dim)
                            // final norm
        public Tensor rms_final_weight_; // (dim,)
                                          // classifier weights for the logits, on the last layer
        public Tensor wcls_;
        // biases for qkv (qwen)
        public Tensor[] bqkv_; // ((n_heads + n_kv_heads * 2) * head_dim)
                               // moe gate weights (mixtral)
        public Tensor[] moegate_; // (n_experts, dim)
	 };

}
