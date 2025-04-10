using CommunityToolkit.HighPerformance;
using cslm;
using System;
using System.Diagnostics;
using System.Reflection;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

namespace cslm
{
	public struct Vocab
	{
		public int offset_;
		public int length_;
	}

	public struct TokenIndex
	{
		public ushort id_;
	}

	public enum TokenizerFlags
	{
		TF_ENCODE_NONE = 0,
		TF_ENCODE_BOS = 1 << 0,
		TF_ENCODE_EOS = 1 << 1,
	}

	public class Tokenizer
	{
		private const int MAX_TOKEN_LENGTH = 512;
		private byte[] tokens_;
		private Vocab[] vocab_;
		private float[] scores_;
		private TokenIndex[] sorted_vocab_;
		private ushort vocab_size_;
		private ushort bos_id_;
		private ushort eos_id_;
		private ushort eot_id_;
		private int byte_fallbacks_;
		private byte[][] byte_pieces_;
		private const int BufferSize = 128;
		private byte[] buffer_ = new byte[BufferSize];

		private static int length(byte[] tokens, int offset)
		{
			for (int i = offset; i < tokens.Length; ++i)
			{
				if (0 == tokens[i])
				{
					return i - offset;
				}
			}
			return 0;
		}

		private static int strcmp(ReadOnlySpan<byte> x0, ReadOnlySpan<byte> x1)
		{
			int len = Math.Min(x0.Length, x1.Length);
			for (int i = 0; i < len; ++i)
			{
				if (x0[i] != x1[i])
				{
					return x0[i] - x1[i];
				}
			}
			return x0.Length - x1.Length;
		}

		private int lower_bound(ReadOnlySpan<byte> x)
		{
			int count = sorted_vocab_.Length;
			int first = 0;
			while (0 < count)
			{
				int d = count / 2;
				int mid = first + d;
				Vocab vocab = vocab_[sorted_vocab_[mid].id_];
				ReadOnlySpan<byte> m = tokens_.AsSpan(vocab.offset_, vocab.length_);
				if (strcmp(m, x) < 0)
				{
					first = mid + 1;
					count -= d + 1;
				}
				else
				{
					count = d;
				}
			}
			return first;
		}

		private ReadOnlySpan<byte> get_token(int id)
		{
			return tokens_.AsSpan(vocab_[id].offset_, vocab_[id].length_);
		}

		private int str_lookup(ReadOnlySpan<byte> x)
		{
			int i = lower_bound(x);
			if (i != sorted_vocab_.Length && !(strcmp(x, get_token(sorted_vocab_[i].id_)) < 0))
			{
				return sorted_vocab_[i].id_;
			}
			else
			{
				return -1;
			}
		}

		public int str_lookup(ReadOnlySpan<char> x)
		{
			int len = Encoding.UTF8.GetBytes(x, buffer_);
			return 0 < len ? str_lookup(buffer_.AsSpan(0, len)) : -1;
		}

		private TokenIndex get_sorted_vocab(int index)
		{
			return sorted_vocab_[index];
		}

		private ReadOnlySpan<byte> get_sorted_token(int index)
		{
			return get_token(sorted_vocab_[index].id_);
		}

		public static Tokenizer initialize(
					ReadOnlySpan<byte> tokens,
					ReadOnlySpan<float> scores,
					int bos_id,
					int eos_id,
					int vocab_size)
		{
			Debug.Assert(vocab_size < ushort.MaxValue);
			Tokenizer tokenizer = new Tokenizer();
			tokenizer.vocab_size_ = (ushort)vocab_size;
			tokenizer.bos_id_ = (0<= bos_id)? (ushort)bos_id : ushort.MaxValue;
			tokenizer.eos_id_ = (0 <= eos_id) ? (ushort)eos_id : ushort.MaxValue;
			tokenizer.eot_id_ = ushort.MaxValue;

			tokenizer.tokens_ = new byte[tokens.Length];
			tokenizer.vocab_ = new Vocab[vocab_size];
			tokenizer.sorted_vocab_ = new TokenIndex[vocab_size];
			tokenizer.scores_ = new float[vocab_size];

			tokens.CopyTo(tokenizer.tokens_);
			scores.CopyTo(tokenizer.scores_);

			int token_offset = 0;
			for (int i = 0; i < vocab_size; ++i)
			{
				tokenizer.vocab_[i].offset_ = token_offset;
				tokenizer.vocab_[i].length_ = length(tokenizer.tokens_, token_offset);
				tokenizer.sorted_vocab_[i].id_ = (ushort)i;
				token_offset += tokenizer.vocab_[i].length_ + 1;
				Debug.Assert(token_offset <= tokens.Length);
			}
			Array.Sort(tokenizer.sorted_vocab_, (a, b) =>
			{
				ReadOnlySpan<byte> x0 = tokenizer.tokens_.AsSpan(tokenizer.vocab_[a.id_].offset_, tokenizer.vocab_[a.id_].length_);
				ReadOnlySpan<byte> x1 = tokenizer.tokens_.AsSpan(tokenizer.vocab_[b.id_].offset_, tokenizer.vocab_[b.id_].length_);
				return strcmp(x0, x1);
			});
			tokenizer.byte_fallbacks_ = tokenizer.str_lookup("<0x00>");

			tokenizer.byte_pieces_ = new byte[256][];
			if (0 < tokenizer.byte_fallbacks_)
			{
				for (int i = 0; i < 256; ++i)
				{
					tokenizer.byte_pieces_[i] = new byte[1];
					tokenizer.byte_pieces_[i][0] = (byte)i;
				}
			}

			if (ushort.MaxValue<=tokenizer.eot_id_)
			{
				int id = tokenizer.str_lookup("<|eot_id|>");
				tokenizer.eot_id_ = 0<=id? (ushort)id : ushort.MaxValue;
			}
			if (ushort.MaxValue <= tokenizer.eot_id_)
			{
				int id = tokenizer.str_lookup("<|end|>");
					tokenizer.eot_id_ = 0 <= id ? (ushort)id : ushort.MaxValue;
			}
			if (ushort.MaxValue <= tokenizer.eot_id_)
			{
				int id = tokenizer.str_lookup("<|im_end|>");
					tokenizer.eot_id_ = 0 <= id ? (ushort)id : ushort.MaxValue;
			}
			return tokenizer;
		}

		private static int bound(int bytes)
		{
			return bytes + 3; // +3 for prefix space, ?BOS, ?EOS
		}

		public ReadOnlySpan<byte> decode(int prev_token, int token)
		{
			ReadOnlySpan<byte> piece = tokens_.AsSpan<byte>(vocab_[token].offset_, vocab_[token].length_);

			// following BOS token, sentencepiece decoder strips any leading whitespace (see PR #89)
			if (prev_token == bos_id_ && piece[0] == ' ')
			{
				piece = piece.Slice(1);
			}
			// return byte piece for byte fallback tokens (<0x00>, <0x01>, etc.)
			if (0 <= byte_fallbacks_ && (token - byte_fallbacks_) < 256)
			{
				piece = byte_pieces_[token - byte_fallbacks_];
			}
			return piece;
		}

		public string decode(List<ushort> tokens)
		{
			if (tokens.Count <= 0)
			{
				return string.Empty;
			}
			List<byte> bytes = new List<byte>(1024);
			bytes.AddRange(decode(tokens[0], tokens[0]));
			for(int i=1; i< tokens.Count; ++i)
			{
				bytes.AddRange(decode(tokens[i - 1], tokens[i]));
			}
			return Encoding.UTF8.GetString(bytes.ToArray());
		}

#if false
		private struct Merge
		{
			public int lpos_;
			public int lid_;
			public int rpos_;
			public int rid_;
			public int resid_;
			public float score_;
		}
#else
		private struct Merge
		{
			public int lpos_;
			public int rpos_;
			public ushort lid_;
			public ushort rid_;
			public ushort resid_;
			public float score_;
		}
#endif

		private static void heap_swap(Merge[] heap, int i, int j)
		{
			Merge tmp = heap[i];
			heap[i] = heap[j];
			heap[j] = tmp;
		}

		private static void heap_insert(Merge[] heap, int n_heap, Merge merge)
		{
			// insert a new element at the end (breaks heap invariant)
			heap[n_heap] = merge;
			n_heap++;

			// bubble up the new element to its correct position
			int i = n_heap - 1;
			while (i > 0 && heap[(i - 1) / 2].score_ < heap[i].score_)
			{
				heap_swap(heap, i, (i - 1) / 2);
				i = (i - 1) / 2;
			}
		}

		private static void heap_poptop(Merge[] heap, int n_heap)
		{
			// move the last element to the top (breaks heap invariant)
			n_heap--;
			heap[0] = heap[n_heap];

			// bubble down the new top element to its correct position
			int i = 0;
			while (i * 2 + 1 < n_heap)
			{
				// find the largest child
				int j = i * 2 + 1;
				if (j + 1 < n_heap && heap[j].score_ < heap[j + 1].score_)
				{
					j++;
				}
				// if the largest child is smaller than the parent, we're done
				if (heap[j].score_ <= heap[i].score_)
				{
					break;
				}
				// otherwise, swap the parent and child
				heap_swap(heap, i, j);
				i = j;
			}
		}

		private int merge_tokens_tryadd(Merge[] heap, int n_heap, int lpos, int lid, int rpos, int rid)
		{
			ReadOnlySpan<byte> l = tokens_.AsSpan<byte>(vocab_[lid].offset_, vocab_[lid].length_);
			ReadOnlySpan<byte> r = tokens_.AsSpan<byte>(vocab_[rid].offset_, vocab_[rid].length_);
			l.CopyTo(buffer_);
			r.CopyTo(buffer_.AsSpan<byte>(l.Length));
			int id = str_lookup(buffer_.AsSpan<byte>(0, l.Length + r.Length));

			if (0 <= id)
			{
				Merge merge = new Merge();
				merge.lpos_ = lpos;
				merge.lid_ = (ushort)lid;
				merge.rpos_ = rpos;
				merge.rid_ = (ushort)rid;
				merge.resid_ = (ushort)id;
				merge.score_ = scores_[id];
				heap_insert(heap, n_heap++, merge);
			}
			return n_heap;
		}

		private void merge_tokens(List<ushort> tokens)
		{
			// create heap for all token merge pairs
			Merge[] heap = new Merge[2 * tokens.Count];
			int n_heap = 0;

			// insert all initial pairs
			for (int i = 0; i < tokens.Count - 1; ++i)
			{
				n_heap = merge_tokens_tryadd(heap, n_heap, i, tokens[i], i + 1, tokens[i + 1]);
			}

			// merge all pairs
			while (0 < n_heap)
			{
				Merge merge = heap[0];
				heap_poptop(heap, n_heap--);

				if (tokens[merge.lpos_] != merge.lid_ || tokens[merge.rpos_] != merge.rid_)
				{
					continue; // this pair was already merged, skip it
				}

				// merge
				tokens[merge.lpos_] = (ushort)merge.resid_;
				tokens[merge.rpos_] = ushort.MaxValue;

				// we might have new pairs to merge
				for (int i = merge.lpos_ - 1; 0 <= i; --i)
				{
					if (tokens[i] != uint.MaxValue)
					{
						n_heap = merge_tokens_tryadd(heap, n_heap, i, tokens[i], merge.lpos_, merge.resid_);
						break;
					}
				}

				for (int i = merge.rpos_ + 1; i < tokens.Count; ++i)
				{
					if (tokens[i] != uint.MaxValue)
					{
						n_heap = merge_tokens_tryadd(heap, n_heap, merge.lpos_, merge.resid_, i, tokens[i]);
						break;
					}
				}
			}

			// compact tokens
			int num_tokens = 0;
			for (int i = 0; i < tokens.Count; ++i)
			{
				if (tokens[i] != ushort.MaxValue)
				{
					tokens[num_tokens++] = tokens[i];
				}
			}
			tokens.RemoveRange(num_tokens, tokens.Count- num_tokens);
		}

		public int encode(List<ushort> tokens, ReadOnlySpan<byte> text, TokenizerFlags flags = TokenizerFlags.TF_ENCODE_NONE)
		{
			int n_tokens = 0;

			// add optional BOS token, if desired
			if (0 != (flags & TokenizerFlags.TF_ENCODE_BOS) && ushort.MaxValue != bos_id_)
			{
				tokens.Add(bos_id_);
			}

			// process the raw (UTF-8) byte sequence of the input string
			byte[] codepoint = new byte[5];
			for (int i = 0; i < text.Length;)
			{
				Array.Fill<byte>(codepoint, 0);

				codepoint[0] = text[i++];

				if (i < text.Length && codepoint[0] == '<' && text[i] == '|')
				{
					// special token, skip until '|>'
					int e = i + 1;
					while (e < (text.Length - 1) && !(text[e] == '|' && text[e + 1] == '>'))
					{
						e++;
					}
					if (text[e] == '|' && text[e + 1] == '>' && (e - i + 3) <= MAX_TOKEN_LENGTH)
					{
						// we found the end of the special token, try to encode it as is
						ReadOnlySpan<byte> special = text.Slice(i - 1, e - i + 3);

						int sid = str_lookup(special);
						if (sid != -1)
						{
							// we found special codepoint in vocab, add it as a token
							tokens.Add((ushort)sid);
							i = e + 2;
							continue;
						}
					}
				}

				// this byte is a leading byte (11...), so it's a multi-byte UTF8 codepoint
				int code_length = 1;
				if ((codepoint[0] & 0xC0U) == 0xC0U)
				{
					for (;(i+code_length) < text.Length && code_length < 4 && (text[i + code_length] & 0xC0) == 0x80; ++code_length)
					{
						codepoint[code_length] = text[i + code_length];
					}
				}

				Span<byte> codepoint_span = codepoint.AsSpan(0, code_length);
				int id = str_lookup(codepoint_span);
				if (0 <= id)
				{
					// we found this codepoint in vocab, add it as a token
					tokens.Add((ushort)id);
				}
				else if (0 <= byte_fallbacks_)
				{
					// byte_fallback encoding: just encode each byte as a token
					foreach (byte b in codepoint_span)
					{
						tokens.Add((ushort)(b + byte_fallbacks_));
					}
				}
			}

			// optimized heap-based merge
			merge_tokens(tokens);

			// add optional EOS token, if desired
			if (0 != (flags & TokenizerFlags.TF_ENCODE_EOS))
			{
				tokens.Add(eos_id_);
			}

			Debug.Assert(tokens.Count <= bound(text.Length));
			return tokens.Count;
		}

#if DEBUG
		public bool check_vocab()
		{
			ReadOnlySpan<byte> prev = new ReadOnlySpan<byte>();
			for (int i = 0; i < sorted_vocab_.Length; ++i)
			{
				ReadOnlySpan<byte> bytes = get_sorted_token(i);
				string str = Encoding.UTF8.GetString(bytes);
				Debug.WriteLine(str);
				if (0 < prev.Length)
				{
					if (strcmp(prev, bytes) > 0)
					{
						return false;
					}
				}
				prev = bytes;
			}
			int id = str_lookup("<s>");
			if (id != 1)
			{
				return false;
			}
			id = str_lookup("<unk>");
			if (id != 0)
			{
				return false;
			}
			return true;
		}
#endif
	}
}

