namespace cslm
{
	public struct Sampler
	{
		public int vocab_size_;
		public float temperature_;
		public float minp_;
		public Random random_;
		public Sampler()
		{
			vocab_size_ = 0;
			temperature_ = 1.0f;
			minp_ = 0.1f;
            random_ = new Random(0);
        }

        public static float sample_prob(int idx, float[] logits, int size)
		{
			// find max value (for numerical stability)
			float max_val = float.MinValue;
			for (int i = 0; i < size; ++i)
			{
				max_val = max_val < logits[i] ? logits[i] : max_val;
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
				if (logit_cutoff <= logits[i])
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

		public static int sample(in Sampler sampler, float[] logits)
		{
			if (sampler.temperature_ < 1.0e-6f || 1.0f <= sampler.minp_)
			{
				// greedy argmax sampling: take the token with the highest probability
				return sample_argmax(logits, sampler.vocab_size_);
			}
			else
			{
				float coin = sampler.random_.frand();
				// min-p (cutoff) sampling, clamping the least likely tokens to zero
				return sample_minp(logits, sampler.vocab_size_, sampler.minp_, sampler.temperature_, coin);
			}
		}

	}
}

