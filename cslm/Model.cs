using CommunityToolkit.HighPerformance;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace cslm
{
	public struct Model
	{
		public Tensors tensors_;
		public Transformer transformer_;
		public Tokenizer tokenizer_;
		public Sampler sampler_;

		public bool initialize(string model, int context_size)
		{
			tensors_ = Tensors.OpenAsync(model).Result;
			if (null == tensors_)
			{
				return false;
			}
			transformer_ = new Transformer();
			get_config(context_size);
			get_tokenizer();
			get_weights();
			if (!tokenizer_.check_vocab())
			{
				return false;
			}

			ulong dummy;
			transformer_.n_bytes_ = count_bytes(tensors_, "model.", null, out transformer_.n_params_);
			transformer_.n_bandwidth_ = transformer_.n_bytes_ - count_bytes(tensors_, "model.embed.", null, out dummy);

			int index;
			if (tensors_.find("model.output.weight", 0, out index))
			{
				transformer_.n_bandwidth_ += (ulong)tensors_.get_tensor(index).size_;
			}
			if (0 < transformer_.config_.n_experts_)
			{
				ulong mlp = count_bytes(tensors_, "model.layers.", ".mlp.w", out dummy);
				transformer_.n_bandwidth_ -= mlp;
				transformer_.n_bandwidth_ += mlp / (ulong)(transformer_.config_.n_experts_ * transformer_.config_.n_experts_ac_);
			}

			transformer_.state_.kvbits_ = 16;
			transformer_.state_.Initialize(transformer_.config_);
			transformer_.forward_ = Inference.forward;
			transformer_.forward_(transformer_, 0, 0, 0);
			return true;
		}

        private ulong kvcache_bandwidth(int kvbits, int pos) {

    int kv_dim = transformer_.config_.head_dim_ * transformer_.config_.n_kv_heads_;
        int kv_len = transformer_.config_.seq_len_<=pos ? transformer_.config_.seq_len_ : pos + 1;
	return (ulong)(2L * (kvbits / 8)* transformer_.config_.n_layers_* kv_dim * kv_len);
}

		public void run(Context context)
		{
			Debug.Assert(null != transformer_.forward_);
			sampler_.vocab_size_ = transformer_.config_.vocab_size_;
			sampler_.temperature_ = context.temperature_;
			sampler_.minp_ = context.pvalue_;
			sampler_.random_.seed(context.seed_);

			Stopwatch sw = new Stopwatch();
			int steps = context.steps_ == 0 ? transformer_.config_.seq_len_ : context.steps_;
			int pos_offset = 0;
            List<ushort> tokens = new List<ushort>();
			List<byte> pieces = new List<byte>();
			for (int i=0; i<context.sequences_; ++i)
			{
				sw.Start();
				tokens.Clear();

				byte[] bytes = Encoding.UTF8.GetBytes(context.input_);
				tokenizer_.encode(tokens, bytes, TokenizerFlags.TF_ENCODE_BOS);
				if(tokens.Count <= 0){
					return;
			}
				int token = tokens[0];
				int pos = 0;
				int next;
				ulong read_bytes = 0;
				float[]? logits_last = null;
				while (pos < steps || steps < 0)
                {
					// forward the transformer to get logits for the next token
					ForwardFlags flags = pos < tokens.Count - 1 ? ForwardFlags.FF_UPDATE_KV_ONLY : ForwardFlags.FF_NONE;
					float[] logits = transformer_.forward_(transformer_, token, pos + pos_offset, (uint)flags);

                    read_bytes += transformer_.n_bandwidth_;

                    read_bytes += kvcache_bandwidth(transformer_.state_.kvbits_, pos + pos_offset);
                    logits_last = logits;

                    // advance the state machine
                    if (pos < tokens.Count - 1)
                    {
                        // if we are still processing the input prompt, force the next prompt token
                        next = tokens[pos + 1];
                    }
                    else
                    {
						// otherwise sample the next token from the logits
						next = Sampler.sample(sampler_, logits);
                        Debug.Assert(0<=next);

                        // data-dependent terminating condition: the BOS token delimits sequences, EOS token ends the sequence, EOT token ends the turn
                        if (next == tokenizer_.BOS_ID || next == tokenizer_.EOS_ID || next == tokenizer_.EOT_ID)
                        {
                            break;
                        }
                    }
                    pos++;

					// print the token as string, decode it with the Tokenizer object
					ReadOnlySpan<byte> piece = tokenizer_.decode(token, next);
					pieces.AddRange(piece);
                    token = next;
                }
				sw.Stop();
				Console.WriteLine(Encoding.UTF8.GetString(pieces.AsSpan<byte>()));
                // fold last token's logits into a hash for validation
                uint logits_hash = 0;
                if (null != logits_last)
                {
					ReadOnlySpan<byte> logits = MemoryMarshal.Cast<float, byte>(logits_last);
					logits_hash = System.IO.Hashing.XxHash32.HashToUInt32(logits);
                }
				string str = string.Format("# {0} tokens: throughput: {1} tok/s; latency: {2} ms/tok; bandwidth: {3} GB/s; total {4} sec; #{5:X}",
					pos,
					pos / sw.Elapsed.TotalSeconds,
					sw.Elapsed.TotalSeconds / pos,
					((double)read_bytes / 1e9) / sw.Elapsed.TotalSeconds,
					sw.Elapsed.TotalSeconds,
					logits_hash);
				Console.WriteLine(str);
            }
        }

		private static ulong count_bytes(Tensors tensors, string prefix, string? filter, out ulong out_params)
		{
			ulong bytes = 0;
			ulong nparams = 0;
			var fn = (Tensor tensor, int elts) =>
			{
				if (0 != tensor.shape0_)
				{
					elts *= tensor.shape0_;
				}
				if (0 != tensor.shape1_)
				{
					elts *= tensor.shape1_;
				}
				if (0 != tensor.shape2_)
				{
					elts *= tensor.shape2_;
				}
				if (0 != tensor.shape3_)
				{
					elts *= tensor.shape3_;
				}
				return elts;
			};
			for (int i = 0; i < tensors.num_tensors(); ++i)
			{
				Tensor tensor = tensors[i];
				if (!tensor.name_.StartsWith(prefix))
				{
					continue;
				}
				if (!string.IsNullOrEmpty(filter) && tensor.name_.Contains(filter))
				{
					continue;
				}

				int elts = tensor.dtype_ == DType.dt_i32 ? 8 : 1; // gsize hack for gf4
				nparams += (ulong)fn(tensor, elts);
				bytes += (ulong)tensor.size_;
			}
			out_params = nparams;
			return bytes;
		}

		public void get_config(int context)
		{
			transformer_.config_.dim_ = tensors_.get_metadata_int("dim", 0);
			transformer_.config_.hidden_dim_ = tensors_.get_metadata_int("hidden_dim", 0);
			transformer_.config_.n_layers_ = tensors_.get_metadata_int("n_layers", 0);
			transformer_.config_.n_heads_ = tensors_.get_metadata_int("n_heads", 0);
			transformer_.config_.n_kv_heads_ = tensors_.get_metadata_int("n_kv_heads", 0);
			transformer_.config_.vocab_size_ = tensors_.get_metadata_int("vocab_size", 0);
			transformer_.config_.head_dim_ = tensors_.get_metadata_int("head_dim", 0);
			transformer_.config_.seq_len_ = tensors_.get_metadata_int("head_dim", 4096);
			if (0 < context)
			{
				transformer_.config_.seq_len_ = context;
			}
			transformer_.config_.rope_theta_ = tensors_.get_metadata_float("rope_theta", 0.0f);
			transformer_.config_.head_dim_ = tensors_.get_metadata_int("rotary_dim", 0);

			int index = tensors_.find_metadata("n_exports");
			if (0 <= index)
			{
				transformer_.config_.n_experts_ = tensors_.get_metadata_int("n_exports", 0);
				transformer_.config_.n_experts_ac_ = tensors_.get_metadata_int("n_experts_ac", 0);
			}
			transformer_.config_.norm_eps_ = tensors_.get_metadata_float("norm_eps", 1.0e-5f);

			string act_type = tensors_.get_metadata_value(tensors_.find_metadata("act_type"));
			transformer_.config_.act_gelu_ = act_type == "gelu";

			string norm_type = tensors_.get_metadata_value(tensors_.find_metadata("norm_type"));
			transformer_.config_.norm_ln_ = norm_type.StartsWith("layernorm");
			transformer_.config_.norm_par_ = norm_type == "layernorm_par";

			transformer_.config_.qkv_clip_ = tensors_.get_metadata_float("qkv_clip", float.MaxValue);
		}

		public void get_weights()
		{
			transformer_.weights_.bytes_ = tensors_.Bytes;
			DType wtype;
			int gsize;
			{
				string type = tensors_.get_metadata_str("dtype", "gf16");
				switch (type)
				{
					case "gf4":
						wtype = DType.dt_i32;
						gsize = 8;
						transformer_.weights_.dbits_ = 4;
						break;
					case "fp8":
						wtype = DType.dt_f8e5m2;
						gsize = 1;
						transformer_.weights_.dbits_ = 8;
						break;
					default:
						wtype = DType.dt_f16;
						gsize = 1;
						transformer_.weights_.dbits_ = 16;
						break;

				}
			}
			transformer_.weights_.token_embedding_table_ = tensors_.get_tensor("model.embed.weight", 0, wtype, transformer_.config_.vocab_size_, transformer_.config_.dim_ / gsize, 0, 0);

			transformer_.weights_.rms_att_weight_ = new Tensor[transformer_.config_.n_layers_];
			if (!transformer_.config_.norm_par_)
			{
				transformer_.weights_.rms_ffn_weight_ = new Tensor[transformer_.config_.n_layers_];
			}
			transformer_.weights_.wq_ = new Tensor[transformer_.config_.n_layers_];
			transformer_.weights_.wk_ = new Tensor[transformer_.config_.n_layers_];
			transformer_.weights_.wv_ = new Tensor[transformer_.config_.n_layers_];
			transformer_.weights_.wo_ = new Tensor[transformer_.config_.n_layers_];
			transformer_.weights_.w1_ = new Tensor[transformer_.config_.n_layers_];
			transformer_.weights_.w2_ = new Tensor[transformer_.config_.n_layers_];
			transformer_.weights_.w3_ = new Tensor[transformer_.config_.n_layers_];
			if (0 < transformer_.config_.n_experts_)
			{
				transformer_.weights_.moegate_ = new Tensor[transformer_.config_.n_layers_];
			}

			int layer_dim0 = transformer_.config_.n_heads_ * transformer_.config_.head_dim_;
			int layer_dim1 = transformer_.config_.dim_ / gsize;
			for (int l = 0; l < transformer_.config_.n_layers_; ++l)
			{
				transformer_.weights_.rms_att_weight_[l] = tensors_.get_tensor("model.layers.{0}.attn.norm.weight", l, DType.dt_f32, transformer_.config_.dim_, 0, 0, 0);

				if (!transformer_.config_.norm_par_)
				{
					transformer_.weights_.rms_ffn_weight_[l] = tensors_.get_tensor("model.layers.{0}.mlp.norm.weight", l, DType.dt_f32, transformer_.config_.dim_, 0, 0, 0);
				}
				transformer_.weights_.wq_[l] = tensors_.get_tensor("model.layers.{0}.attn.wq.weight", l, wtype, transformer_.config_.n_heads_ * transformer_.config_.head_dim_, layer_dim1, 0, 0);
				transformer_.weights_.wk_[l] = tensors_.get_tensor("model.layers.{0}.attn.wk.weight", l, wtype, transformer_.config_.n_kv_heads_ * transformer_.config_.head_dim_, layer_dim1, 0, 0);
				transformer_.weights_.wv_[l] = tensors_.get_tensor("model.layers.{0}.attn.wv.weight", l, wtype, transformer_.config_.n_kv_heads_ * transformer_.config_.head_dim_, layer_dim1, 0, 0);
				transformer_.weights_.wo_[l] = tensors_.get_tensor("model.layers.{0}.attn.wo.weight", l, wtype, transformer_.config_.dim_, layer_dim1, 0, 0);
				if (0 <= tensors_.find("model.layers.{0}.attn.wqkv.bias", l))
				{
					if (null == transformer_.weights_.bqkv_)
					{
						transformer_.weights_.bqkv_ = new Tensor[transformer_.config_.n_layers_];
					}
					transformer_.weights_.bqkv_[l] = tensors_.get_tensor("model.layers.{0}.attn.wqkv.bias", l, DType.dt_f32, (transformer_.config_.n_heads_ + transformer_.config_.n_kv_heads_ * 2) * transformer_.config_.head_dim_, 0, 0, 0);
				}

				if (0 < transformer_.config_.n_experts_)
				{
					transformer_.weights_.moegate_[l] = tensors_.get_tensor("model.layers.{0}.moegate.weight", l, wtype, transformer_.config_.n_experts_, transformer_.config_.dim_ / gsize, 0, 0);

					transformer_.weights_.w1_[l] = tensors_.get_tensor("model.layers.{0}.mlp.w1.weight", l, wtype, transformer_.config_.n_experts_, transformer_.config_.hidden_dim_, transformer_.config_.dim_ / gsize, 0);
					transformer_.weights_.w2_[l] = tensors_.get_tensor("model.layers.{0}.mlp.w2.weight", l, wtype, transformer_.config_.n_experts_, transformer_.config_.dim_, transformer_.config_.hidden_dim_ / gsize, 0);
					transformer_.weights_.w3_[l] = tensors_.get_tensor("model.layers.{0}.mlp.w3.weight", l, wtype, transformer_.config_.n_experts_, transformer_.config_.hidden_dim_, transformer_.config_.dim_ / gsize, 0);
				}
				else
				{
					transformer_.weights_.w1_[l] = tensors_.get_tensor("model.layers.{0}.mlp.w1.weight", l, wtype, transformer_.config_.hidden_dim_, transformer_.config_.dim_ / gsize, 0, 0);
					transformer_.weights_.w2_[l] = tensors_.get_tensor("model.layers.{0}.mlp.w2.weight", l, wtype, transformer_.config_.dim_, transformer_.config_.hidden_dim_ / gsize, 0, 0);
					transformer_.weights_.w3_[l] = tensors_.get_tensor("model.layers.{0}.mlp.w3.weight", l, wtype, transformer_.config_.hidden_dim_, transformer_.config_.dim_ / gsize, 0, 0);
				}
			}

			transformer_.weights_.rms_final_weight_ = tensors_.get_tensor("model.norm.weight", 0, DType.dt_f32, transformer_.config_.dim_, 0, 0, 0);

			if (0 < tensors_.find("model.output.weight", 0))
			{
				transformer_.weights_.wcls_ = transformer_.weights_.token_embedding_table_; // tied weights
			}
			else
			{
				transformer_.weights_.wcls_ = tensors_.get_tensor("model.output.weight", 0, wtype, transformer_.config_.vocab_size_, transformer_.config_.dim_ / gsize, 0, 0);
			}
		}

		public void get_tokenizer()
		{
			Tensor tensor = tensors_.get_tensor("tokenizer.tokens");
			ReadOnlySpan<byte> tokens = tensors_.as_span<byte>("tokenizer.tokens");
			ReadOnlySpan<float> scores = tensors_.as_span<float>("tokenizer.scores");

			int bos_id = tensors_.get_metadata_int("bos_token_id", 0);
			int eos_id = tensors_.get_metadata_int("eos_token_id", 0);

			tokenizer_ = Tokenizer.initialize(tokens, scores, bos_id, eos_id, transformer_.config_.vocab_size_);
		}
	}
}
