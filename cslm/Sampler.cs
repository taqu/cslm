namespace cslm
{
    public struct Sampler
    {
        public int vocab_size_;
        public ulong rng_state_;
        public float temperature_;
        public float minp_;

        public static float sample_prob(int idx, float[] logits, int size)
        {
            // find max value (for numerical stability)
            float max_val = float.MinValue;
            for (int i = 0; i < size; ++i)
            {
                max_val = max_val<logits[i] ? logits[i] : max_val;
            }
            // exp and sum
            float sum = 0.0f;
            for (int i = 0; i < size; ++i)
            {
                sum += MathF.Exp(logits[i] - max_val);
            }
            // return probability of the given index
            return MathF.Exp(logits[idx] - max_val) / sum;
        }

        public static int sample_argmax(float[] logits, int n)
        {
            int max_i = -1;
            float max_p = float.MinValue;
            for (int i = 0; i < n; ++i)
            {
                max_i = logits[i] > max_p ? i : max_i;
                max_p = logits[i] > max_p ? logits[i] : max_p;
            }
            return max_i;
        }

        public static int sample_minp(float[] logits, int n, float minp, float temperature, float coin)
        {
            // find max logit; we will use this to derive minp cutoff (in log space), since minp is scale-invariant (wrt softmax)
            float max_logit = float.MinValue;
            for (int i = 0; i < n; ++i)
            {
                max_logit = logits[i] > max_logit ? logits[i] : max_logit;
            }

            // exp(logit / temp) <= exp(max_logit / temp) * minp -> logit <= max_logit + log(minp) * temp
            float logit_cutoff = max_logit + MathF.Log(minp) * temperature;

            // convert from logits to probabilities in-place while simultaneously doing (unscaled) softmax; we'll rescale later
            float[] probs = logits;
            int fallback = 0;
            float cumulative_prob = 0.0f;
            for (int i = 0; i < n; i++)
            {
                if (logit_cutoff<=logits[i])
                {
                    probs[i] = MathF.Exp((logits[i] - max_logit) / temperature);
                    cumulative_prob += probs[i];
                    fallback = i; // for fallback due to rounding errors
                }
                else
                {
                    probs[i] = 0.0f;
                }
            }

            // sample from the truncated list
            float r = coin * cumulative_prob;
            float cdf = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                cdf += probs[i];
                if (r < cdf)
                {
                    return i;
                }
            }
            return fallback; // in case of rounding errors
        }

        public static int sample(ref Sampler sampler, float[] logits)
        {
            if (sampler.temperature_ == 0.0f || 1.0f<= sampler.minp_)
            {
                // greedy argmax sampling: take the token with the highest probability
                return sample_argmax(logits, sampler.vocab_size_);
            }
            else
            {
                float coin = random_f32(ref sampler.rng_state_);
                // min-p (cutoff) sampling, clamping the least likely tokens to zero
                return sample_minp(logits, sampler.vocab_size_, sampler.minp_, sampler.temperature_, coin);
            }
        }

        public static uint rotr32(uint x, uint r)
        {
            int rot = (int)r;
            return (x >> rot) | x << (-rot & 31);
        }
        public static uint random_u32(ref ulong state)
        {
            ulong x = state;
            uint count = (uint)(x >> 59);

            state = x * 6364136223846793005UL + 1442695040888963407UL;
            x ^= x >> 18;
            return rotr32((uint)(x >> 27), count);
        }

        public static float random_f32(ref ulong state)
        {
            uint x = random_u32(ref state);
            return (x>>8) / 16777216.0f;
        }
    }
}

