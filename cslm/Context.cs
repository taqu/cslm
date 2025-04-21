using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Silk.NET.OpenCL;

namespace cslm
{
	public struct Context
	{
		public string model_;
		public float temperature_;
		public float pvalue_;
		public ulong seed_;
		public int steps_;
		public int sequences_;
		public int context_length_;
		public string input_;
		public string perplexity_;
		public string system_prompt_;


		public void initialize()
		{
			CL.GetApi();
        }
	}
}
