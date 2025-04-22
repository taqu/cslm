using Silk.NET.OpenCL;

namespace cslm
{
	public struct Context
	{
		public string model_ = string.Empty;
		public float temperature_ = 0.0f;
		public float pvalue_ = 0.1f;
		public ulong seed_ = 42;
		public int steps_ = 256;
		public int sequences_ = 1;
		public int context_length_ = -1;
		public int threads_ = 1;
		public string input_ = string.Empty;
		public string perplexity_ = string.Empty;
		public string system_prompt_ = string.Empty;

		public Context()
		{ }
	}
}
