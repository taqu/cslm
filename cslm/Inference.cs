using CommunityToolkit.HighPerformance;
using System.Runtime.InteropServices;

namespace cslm
{
    using kvtype_t = Half;

    public static class Inference
    {
        public static kvtype_t fp82half(byte v)
        {
            ushort u = v;
            u <<= 8;
            return BitConverter.UInt16BitsToHalf(u);
        }

        public static float gf4_ff(uint v, int k)
        {
            float s = (float)fp82half((byte)(v & 0xFFU)) / -4.0f; // we expect compiler to reuse this across multiple calls
            return ((int)((v >> (8 + k * 3)) & 7) - 4) * s;
        }

        public static float dotprod_fp16(Span<byte> w, int n, int i, float[] x)
        {
            Span<Half> r = MemoryMarshal.Cast<byte, Half>(w).Slice(i + n);
            float val = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                val += (float)r[j] * x[j];
            }
            return val;
        }

        public static float dotprod_fp8(Span<byte> w, int n, int i, float[] x)
        {
            Span<byte> r = w.Slice(i * n);
            float val = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                val += (float)fp82half((byte)r[j]) * x[j];
            }
            return val;
        }

        public static float dotprod_gf4(Span<byte> w, int n, int i, float[] x)
        {
            Span<uint> r = MemoryMarshal.Cast<byte, uint>(w).Slice(i * n / 8);
            float val = 0.0f;
            for (int j = 0; j < n; j += 8)
            {
                uint wg = r[j / 8];
                for (int k = 0; k < 8; ++k)
                {
                    val += gf4_ff(wg, k) * x[j + k];
                }
            }
            return val;
        }

        public static void prepare(Transformer transformer)
        {
            transformer.state_.Initialize(transformer.config_);
        }

        public static void rmsnorm(float[] o, float[] x, Span<float> weight, int size, float eps, bool ln)
        {
            // calculate mean
            float mean = 0.0f;

            if (ln)
            {
                for (int i = 0; i < size; ++i)
                {
                    mean += x[i];
                }
                mean /= size;
            }

            // calculate sum of squared deltas
            float ss = 0.0f;
            for (int i = 0; i < size; ++i)
            {
                ss += (x[i] - mean) * (x[i] - mean);
            }

            float var = ss / size;

            // normalize and scale
            float scale = 1.0f / MathF.Sqrt(var + eps);
            for (int i = 0; i < size; ++i)
            {
                o[i] = (x[i] - mean) * scale * weight[i];
            }
        }

        public delegate float Dotprod(Span<byte> w, int n, int i, float[] x);

        public static void matmul(float[] xout, float[] x, Span<byte> w, Span<float> b, int n, int d, Dotprod dotprod)
        {
            // W (d,n) @ x (n,) -> xout (d,)
            // by far the most amount of time is spent inside this little function
            for (int i = 0; i < d; ++i)
            {
                float val = dotprod(w, n, i, x);
                if (0 < b.Length)
                {
                    val += b[i];
                }
                xout[i] = val;
            }
        }

        public static void rope(float[] vec, int d, int head_dim, int pos, float theta, int rotary_dim)
        {
            for (int i = 0; i < d; i += 2)
            {
                int j_head = i % head_dim;
                float freq = rotary_dim <= j_head ? 0.0f : 1.0f / MathF.Pow(theta, (float)j_head / (float)rotary_dim);
                float val = pos * freq;
                float fcr = MathF.Cos(val);
                float fci = MathF.Sin(val);

                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        public static void attn(Span<float> xout, Span<float> atth, Span<float> qh, Span<kvtype_t> kh, Span<kvtype_t> vh, int head_dim, int kv_dim, int kv_len)
        {
            float score_max = float.MinValue;
            float sqrt_head_dim = MathF.Sqrt(head_dim);

            // calculate attention scores as dot products of q and k; also track score max for this head
            for (int t = 0; t < kv_len; ++t)
            {
                float score = 0.0f;
                for (int j = 0; j < head_dim; ++j)
                {
                    score += qh[j] * (float)kh[t * kv_dim + j];
                }
                score /= sqrt_head_dim;
                score_max = (score_max < score) ? score : score_max;
                atth[t] = score;
            }

            // softmax the scores to get attention weights over [0..kv_len)
            float score_sum = 0.0f;
            for (int t = 0; t < kv_len; ++t)
            {
                atth[t] = MathF.Exp(atth[t] - score_max);
                score_sum += atth[t];
            }

            // mix values with attention weights
            for (int j = 0; j < head_dim; ++j)
            {
                float res = 0.0f;
                for (int t = 0; t < kv_len; ++t)
                {
                    res += (atth[t] / score_sum) * (float)vh[t * kv_dim + j];
                }
                xout[j] = res;
            }
        }

        public static float gelu(float x)
        {
            return 0.5f * x * (1.0f + MathF.Tanh(0.797885f * (x + 0.044715f * x * x * x)));
        }

        public static float silu(float x)
        {
            return x / (1.0f + MathF.Exp(-x));
        }

        public static void moe_gate(Span<float> moe_weights, Span<int> moe_experts, Span<float> x, int d, int active)
        {
            // softmax across experts
            float max_val = float.MinValue;
            for (int j = 0; j < d; ++j)
            {
                max_val = (max_val < x[j]) ? x[j] : max_val;
            }

            // top k
            ulong mask = 0;
            float wsum = 0.0f;
            for (int k = 0; k < active; ++k)
            {
                int best = -1;
                for (int j = 0; j < d; ++j)
                {
                    if ((mask & (1UL << j)) == 0 && (best == -1 || x[j] > x[best]))
                    {
                        best = j;
                    }
                }

                moe_experts[k] = best;
                wsum += MathF.Exp(x[moe_experts[k]] - max_val);
                mask |= 1UL << best;
            }

            // top k weights, normalized
            for (int k = 0; k < active; ++k)
            {
                moe_weights[k] = MathF.Exp(x[moe_experts[k]] - max_val) / wsum;
            }
        }

        public static float clip(float x, float v)
        {
            return x < -v ? -v : (v < x ? x : v);
        }

        public static float[] forward(Transformer transformer, int token, int pos, uint flags)
        {
            Dotprod dotprod;
            switch (transformer.weights_.dbits_)
            {
                case 4:
                    dotprod = dotprod_gf4;
                    break;
                case 8:
                    dotprod = dotprod_fp8;
                    break;
                case 16:
                    dotprod = dotprod_fp16;
                    break;
                default:
                    System.Diagnostics.Debug.Assert(false, "Unsupported dbits: must be 8 or 16 for CPU");
                    return null;
            }

            float[] x = transformer.state_.x_;
            int dim = transformer.config_.dim_;
            int hidden_dim = transformer.config_.hidden_dim_;
            int q_dim = transformer.config_.head_dim_ * transformer.config_.n_heads_;
            int kv_dim = transformer.config_.head_dim_ * transformer.config_.n_kv_heads_;
            int kv_mul = transformer.config_.n_heads_ / transformer.config_.n_kv_heads_; // integer multiplier of the kv sharing in multiquery

            // following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
            int kv_sink = transformer.config_.seq_len_ <= pos ? Consts.KV_SINKS : 0;
            int kv_pos = kv_sink + (pos - kv_sink) % (transformer.config_.seq_len_ - kv_sink);
            int kv_len = transformer.config_.seq_len_ <= pos ? transformer.config_.seq_len_ : pos + 1;

            int dbits = transformer.weights_.dbits_;

            // copy the token embedding into x
            Span<byte> content_row = transformer.weights_.token_embedding_table_.AsSpan<byte>(token * dim * (dbits / 8));
            if (4 == dbits)
            {
                Span<uint> content_row_u32 = MemoryMarshal.Cast<byte, uint>(content_row);
                for (int i = 0; i < dim; i += 8)
                {
                    uint wg = content_row_u32[i / 8];
                    for (int k = 0; k < 8; ++k)
                    {
                        x[i + k] = gf4_ff(wg, k);
                    }
                }
            }
            else if (8 == dbits)
            {
                for (int i = 0; i < dim; ++i)
                {
                    x[i] = (float)fp82half(content_row[i]);
                }
            }
            else
            {
                Span<short> content_row_s16 = MemoryMarshal.Cast<byte, short>(content_row);
                for (int i = 0; i < dim; ++i)
                {
                    Half h = (Half)content_row_s16[i];
                    x[i] = (float)h;
                }
            }

            Span<float> empty_span = new Span<float>();

            // forward all the layers
            for (int l = 0; l < transformer.config_.n_layers_; ++l)
            {
                // attention rmsnorm
                Span2D<float> rms_att_weight = transformer.weights_.rms_att_weight_;
                rms_att_weight.GetRowSpan(l);

                rmsnorm(transformer.state_.xb_, x, rms_att_weight.GetRowSpan(l), dim, transformer.config_.norm_eps_, transformer.config_.norm_ln_);

                // qkv matmuls for this position
                Span<byte> wq = transformer.weights_.wq_.GetRowSpan(l);
                Span<byte> wk = transformer.weights_.wk_.GetRowSpan(l);
                Span<byte> wv = transformer.weights_.wv_.GetRowSpan(l);
                int bqkv_len = transformer.weights_.bqkv_.Length;
                Span<float> bqkv = 0 < bqkv_len ? transformer.weights_.bqkv_.GetRowSpan(l) : new Span<float>();
                matmul(transformer.state_.q_, transformer.state_.xb_, wq, bqkv, dim, q_dim, dotprod);
                bqkv = 0 < bqkv_len ? bqkv.Slice(q_dim) : empty_span;
                matmul(transformer.state_.k_, transformer.state_.xb_, wk, bqkv, dim, kv_dim, dotprod);
                bqkv = 0 < bqkv_len ? bqkv.Slice(kv_dim) : empty_span;
                matmul(transformer.state_.v_, transformer.state_.xb_, wv, bqkv, dim, kv_dim, dotprod);

                // some models require clipping qkv values
                for (int i = 0; i < q_dim; ++i)
                {
                    transformer.state_.q_[i] = clip(transformer.state_.q_[i], transformer.config_.qkv_clip_);
                }
                for (int i = 0; i < kv_dim; ++i)
                {
                    transformer.state_.k_[i] = clip(transformer.state_.k_[i], transformer.config_.qkv_clip_);
                    transformer.state_.v_[i] = clip(transformer.state_.v_[i], transformer.config_.qkv_clip_);
                }

                // RoPE relative positional encoding: complex-valued rotate q and k in each head
                rope(transformer.state_.q_, q_dim, transformer.config_.head_dim_, pos, transformer.config_.rope_theta_, transformer.config_.rotary_dim_);
                rope(transformer.state_.k_, kv_dim, transformer.config_.head_dim_, pos, transformer.config_.rope_theta_, transformer.config_.rotary_dim_);

                // key and value point to the kv cache
                int loff = l * transformer.config_.seq_len_ * kv_dim; // kv cache layer offset for convenience
                Span<kvtype_t> kb = MemoryMarshal.Cast<byte, kvtype_t>(transformer.state_.key_cache_.AsSpan()).Slice(loff);
                Span<kvtype_t> vb = MemoryMarshal.Cast<byte, kvtype_t>(transformer.state_.value_cache_.AsSpan()).Slice(loff);

                // update kv cache
                for (int i = 0; i < kv_dim; ++i)
                {
                    kb[kv_pos * kv_dim + i] = (kvtype_t)transformer.state_.k_[i];
                    vb[kv_pos * kv_dim + i] = (kvtype_t)transformer.state_.v_[i];
                }

                // rotate sink tokens forward to keep pace with non-sink tokens
                for (int r = 0; r < kv_sink; ++r)
                {
                    for (int i = 0; i < kv_dim; ++i)
                    {
                        transformer.state_.k_[i] = (float)kb[r * kv_dim + i];
                    }

                    rope(transformer.state_.k_, kv_dim, transformer.config_.head_dim_, 1, transformer.config_.rope_theta_, transformer.config_.rotary_dim_);

                    for (int i = 0; i < kv_dim; ++i)
                    {
                        kb[r * kv_dim + i] = (kvtype_t)transformer.state_.k_[i];
                    }
                }

                // multihead attention. iterate over all heads
                for (int h = 0; h < transformer.config_.n_heads_; ++h)
                {
                    Span<float> qh = transformer.state_.q_.AsSpan(h * transformer.config_.head_dim_);
                    Span<float> atth = transformer.state_.att_.AsSpan(h * transformer.config_.seq_len_);
                    Span<kvtype_t> kh = kb.Slice((h / kv_mul) * transformer.config_.head_dim_);
                    Span<kvtype_t> vh = vb.Slice((h / kv_mul) * transformer.config_.head_dim_);

                    attn(transformer.state_.xb2_.AsSpan(h * transformer.config_.head_dim_), atth, qh, kh, vh, transformer.config_.head_dim_, kv_dim, kv_len);
                }

                // final matmul to get the output of the attention
                // TODO: we're using hb as a temporary storage, hacky
                matmul(transformer.state_.hb_, transformer.state_.xb2_, transformer.weights_.wo_.GetRowSpan(l), empty_span, q_dim, dim, dotprod);

                // residual connection back into x
                for (int i = 0; i < dim; ++i)
                {
                    x[i] += transformer.state_.hb_[i];
                }

                if (!transformer.config_.norm_par_)
                {
                    // ffn rmsnorm
                    rmsnorm(transformer.state_.xb_, x, transformer.weights_.rms_ffn_weight_.GetRowSpan(l), dim, transformer.config_.norm_eps_, transformer.config_.norm_ln_);
                }

                Span<float> moe_weights = transformer.state_.exp_.AsSpan(transformer.config_.n_experts_);
                Span<int> moe_experts = MemoryMarshal.Cast<float, int>(moe_weights).Slice(0 != transformer.config_.n_experts_ac_ ? transformer.config_.n_experts_ac_ : 1);

                if (0 != transformer.config_.n_experts_)
                {
                    // moe gate
                    matmul(transformer.state_.exp_, transformer.state_.xb_, transformer.weights_.moegate_.GetRowSpan(l), empty_span, dim, transformer.config_.n_experts_, dotprod);
                    moe_gate(moe_weights, moe_experts, transformer.state_.exp_, transformer.config_.n_experts_, transformer.config_.n_experts_ac_);
                }
                else
                {
                    moe_weights[0] = 1.0f;
                    moe_experts[0] = 0;
                }

                // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
                for (int e = 0; e < (0 != transformer.config_.n_experts_ac_ ? transformer.config_.n_experts_ac_ : 1); ++e)
                {
                    int esize = dim * hidden_dim * (transformer.weights_.dbits_ / 8);
                    matmul(transformer.state_.hb_, transformer.state_.xb_, transformer.weights_.w1_.GetRowSpan(l).Slice(moe_experts[e] * esize), empty_span, dim, hidden_dim, dotprod);
                    matmul(transformer.state_.hb2_, transformer.state_.xb_, transformer.weights_.w3_.GetRowSpan(l).Slice(moe_experts[e] * esize), empty_span, dim, hidden_dim, dotprod);

                    if (transformer.config_.act_gelu_)
                    {
                        // GEGLU non-linearity
                        for (int i = 0; i < hidden_dim; ++i)
                        {
                            transformer.state_.hb_[i] = gelu(transformer.state_.hb_[i]) * transformer.state_.hb2_[i];
                        }
                    }
                    else
                    {
                        // SwiGLU non-linearity
                        for (int i = 0; i < hidden_dim; ++i)
                        {
                            transformer.state_.hb_[i] = silu(transformer.state_.hb_[i]) * transformer.state_.hb2_[i];
                        }
                    }

                    matmul(transformer.state_.xb2_, transformer.state_.hb_, transformer.weights_.w2_.GetRowSpan(l).Slice(moe_experts[e] * esize), empty_span, hidden_dim, dim, dotprod);

                    for (int i = 0; i < dim; ++i)
                    {
                        x[i] += transformer.state_.xb2_[i] * moe_weights[e];
                    }
                }
            }
            if (0 != (flags & (uint)ForwardFlags.FF_UPDATE_KV_ONLY))
            {
                // only update kv cache and don't output logits
                return null;
            }


            // final rmsnorm
            rmsnorm(x, x, transformer.weights_.rms_final_weight_, dim, transformer.config_.norm_eps_, transformer.config_.norm_ln_);

            // classifier into logits
            matmul(transformer.state_.logits_, x, transformer.weights_.wcls_, empty_span, transformer.config_.dim_, transformer.config_.vocab_size_, dotprod);

            return transformer.state_.logits_;
        }
    }
}

