
using cslm;
using static cslm.Weights;
using static System.Formats.Asn1.AsnWriter;

namespace cslm
{
	public struct Model
	{
		public Tensors tensors_;
		public Transformer transformer_;
		public Tokenizer tokenizer_;

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
#if false
			const char* dtype = tensors_metadata(tensors, "dtype");

		enum DType wtype = strcmp(dtype, "gf4") == 0 ? dt_i32 : (strcmp(dtype, "fp8") == 0 ? dt_f8e5m2 : dt_f16);
	int gsize = strcmp(dtype, "gf4") == 0 ? 8 : 1;

		weights->dbits = strcmp(dtype, "gf4") == 0 ? 4 : (strcmp(dtype, "fp8") == 0 ? 8 : 16);

	weights->token_embedding_table = tensors_get(tensors, "model.embed.weight", 0, wtype, (int[]){ config->vocab_size, config->dim / gsize, 0, 0});

	for (int l = 0; l<config->n_layers; ++l) {
		weights->rms_att_weight[l] = (float*) tensors_get(tensors, "model.layers.%d.attn.norm.weight", l, dt_f32, (int[]){ config->dim, 0, 0, 0});

		if (!config->norm_par) {
			weights->rms_ffn_weight[l] = (float*) tensors_get(tensors, "model.layers.%d.mlp.norm.weight", l, dt_f32, (int[]){ config->dim, 0, 0, 0});
		}

	weights->wq[l] = tensors_get(tensors, "model.layers.%d.attn.wq.weight", l, wtype, (int[]){ config->n_heads * config->head_dim, config->dim / gsize, 0, 0});
		weights->wk[l] = tensors_get(tensors, "model.layers.%d.attn.wk.weight", l, wtype, (int[]){ config->n_kv_heads * config->head_dim, config->dim / gsize, 0, 0});
		weights->wv[l] = tensors_get(tensors, "model.layers.%d.attn.wv.weight", l, wtype, (int[]){ config->n_kv_heads * config->head_dim, config->dim / gsize, 0, 0});
		weights->wo[l] = tensors_get(tensors, "model.layers.%d.attn.wo.weight", l, wtype, (int[]){ config->dim, config->n_heads * config->head_dim / gsize, 0, 0});

		if (tensors_find(tensors, "model.layers.%d.attn.wqkv.bias", l)) {
			weights->bqkv[l] = (float*) tensors_get(tensors, "model.layers.%d.attn.wqkv.bias", l, dt_f32, (int[]){ (config->n_heads + config->n_kv_heads * 2) * config->head_dim, 0, 0, 0});
		}

if (config->n_experts)
{
	weights->moegate[l] = tensors_get(tensors, "model.layers.%d.moegate.weight", l, wtype, (int[]){ config->n_experts, config->dim / gsize, 0, 0});

	weights->w1[l] = tensors_get(tensors, "model.layers.%d.mlp.w1.weight", l, wtype, (int[]){ config->n_experts, config->hidden_dim, config->dim / gsize, 0});
	weights->w2[l] = tensors_get(tensors, "model.layers.%d.mlp.w2.weight", l, wtype, (int[]){ config->n_experts, config->dim, config->hidden_dim / gsize, 0});
	weights->w3[l] = tensors_get(tensors, "model.layers.%d.mlp.w3.weight", l, wtype, (int[]){ config->n_experts, config->hidden_dim, config->dim / gsize, 0});
}
else
{
	weights->w1[l] = tensors_get(tensors, "model.layers.%d.mlp.w1.weight", l, wtype, (int[]){ config->hidden_dim, config->dim / gsize, 0, 0});
	weights->w2[l] = tensors_get(tensors, "model.layers.%d.mlp.w2.weight", l, wtype, (int[]){ config->dim, config->hidden_dim / gsize, 0, 0});
	weights->w3[l] = tensors_get(tensors, "model.layers.%d.mlp.w3.weight", l, wtype, (int[]){ config->hidden_dim, config->dim / gsize, 0, 0});
}
	}

	weights->rms_final_weight = (float*)tensors_get(tensors, "model.norm.weight", 0, dt_f32, (int[]){ config->dim, 0, 0, 0});

if (tensors_find(tensors, "model.output.weight", 0) == NULL)
{
	weights->wcls = weights->token_embedding_table; // tied weights
}
else
{
	weights->wcls = tensors_get(tensors, "model.output.weight", 0, wtype, (int[]){ config->vocab_size, config->dim / gsize, 0, 0});
}
#endif
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
