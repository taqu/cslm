namespace cslm
{
    public struct Vocab
    {
		public long offset_;
		public long length_;
	}

	public struct TokenIndex
    {
        public int id_;
    }

	public enum TokenizerFlags
	{
		TF_ENCODE_BOS = 1 << 0,
		TF_ENCODE_EOS = 1 << 1,
	}

	public class Tokenizer
    {
        private byte[] vocab_;
        private float[] scores_;
        private TokenIndex sorted_vocab_;
        private int vocab_size_;
        private int bos_id_;
        private int eos_id_;
        private int eot_id_;
        private int byte_fallbacks_;
        private byte[,] byte_pieces;

        public static Tokenizer initialize(
            ReadOnlySpan<byte> tokens,
            ReadOnlySpan<float> scores,
            int bos_id,
            int eos_id,
            int vocab_size,
            int total_length)
        {
			Tokenizer tokenizer = new Tokenizer();
			tokenizer.vocab_size_ = vocab_size;
			tokenizer.bos_id_ = bos_id;
			tokenizer.eos_id_ = eos_id;
			tokenizer.eot_id_ = -1;
            return tokenizer;

	//		tokenizer.vocab_ = (char**)malloc(vocab_size * sizeof(char*));
	//		tokenizer.sorted_vocab_ = (struct TokenIndex*)malloc(vocab_size* sizeof(struct TokenIndex));
	//tokenizer.scores_ = scores;
		}
}
}

