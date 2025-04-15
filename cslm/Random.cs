
using System;
using static cslm.Random;

namespace cslm
{
	public struct Random
	{
		public struct W128
		{
			public uint x0_;
			public uint x1_;
			public uint x2_;
			public uint x3_;
		}

		public const int SFMT_MEXP = 607;
		public const int SFMT_N = (SFMT_MEXP / 128 + 1);
		public const int SFMT_N32 = SFMT_N * 4;
		public const int SFMT_POS1 = 2;
		public const int SFMT_SL1 = 15;
		public const int SFMT_SL2 = 3;
		public const int SFMT_SR1 = 13;
		public const int SFMT_SR2 = 3;
		public const uint SFMT_MSK1 = 0xfdff37ffU;
		public const uint SFMT_MSK2 = 0xef7f3f7dU;
		public const uint SFMT_MSK3 = 0xff777b7dU;
		public const uint SFMT_MSK4 = 0x7ff7fb2fU;
		public const uint SFMT_PARITY1 = 0x00000001U;
		public const uint SFMT_PARITY2 = 0x00000000U;
		public const uint SFMT_PARITY3 = 0x00000000U;
		public const uint SFMT_PARITY4 = 0x5986f054U;

		private uint index_;
		private W128[] state_;

		public Random()
		{
			index_ = 0;
			state_ = new W128[SFMT_N];
		}

		private static void lshift128(out W128 r, in W128 i, int shift)
		{

			ulong th = ((ulong)i.x3_ << 32) | ((ulong)i.x2_);
			ulong tl = ((ulong)i.x1_ << 32) | ((ulong)i.x0_);
			ulong oh = th << (shift * 8);
			ulong ol = tl << (shift * 8);
			oh |= tl >> (64 - shift * 8);
			r.x1_ = (uint)(ol >> 32);
			r.x0_ = (uint)ol;
			r.x3_ = (uint)(oh >> 32);
			r.x2_ = (uint)oh;
		}

		private static void rshift128(out W128 r, in W128 i, int shift)
		{
			ulong th = ((ulong)i.x3_ << 32) | ((ulong)i.x2_);
			ulong tl = ((ulong)i.x1_ << 32) | ((ulong)i.x0_);
			ulong oh = th >> (shift * 8);
			ulong ol = tl >> (shift * 8);
			ol |= th << (64 - shift * 8);
			r.x1_ = (uint)(ol >> 32);
			r.x0_ = (uint)ol;
			r.x3_ = (uint)(oh >> 32);
			r.x2_ = (uint)oh;
		}

		private static void do_recursion(ref W128 r, W128 a, W128 b, W128 c, W128 d)
		{
			W128 x;
			W128 y;
			lshift128(out x, a, SFMT_SL2);
			rshift128(out y, c, SFMT_SR2);
			r.x0_ = a.x0_ ^ x.x0_ ^ ((b.x0_ >> SFMT_SR1) & SFMT_MSK1) ^ y.x0_ ^ (d.x0_ << SFMT_SL1);
			r.x1_ = a.x1_ ^ x.x1_ ^ ((b.x1_ >> SFMT_SR1) & SFMT_MSK2) ^ y.x1_ ^ (d.x1_ << SFMT_SL1);
			r.x2_ = a.x2_ ^ x.x2_ ^ ((b.x2_ >> SFMT_SR1) & SFMT_MSK3) ^ y.x2_ ^ (d.x2_ << SFMT_SL1);
			r.x3_ = a.x3_ ^ x.x3_ ^ ((b.x3_ >> SFMT_SR1) & SFMT_MSK4) ^ y.x3_ ^ (d.x3_ << SFMT_SL1);
		}

		private void generate()
		{
			W128 r1 = state_[SFMT_N - 2];
			W128 r2 = state_[SFMT_N - 1];
			int i;
			for (i = 0; i < SFMT_N - SFMT_POS1; ++i)
			{
				do_recursion(ref state_[i], state_[i], state_[i + SFMT_POS1], r1, r2);
				r1 = r2;
				r2 = state_[i];
			}
			for (; i < SFMT_N; ++i)
			{
				do_recursion(ref state_[i], state_[i], state_[i + SFMT_POS1 - SFMT_N], r1, r2);
				r1 = r2;
				r2 = state_[i];
			}
		}

		public uint rand()
		{
			if (SFMT_N32 <= index_)
			{
				generate();
				index_ = 0;
			}
			uint i = index_ >> 2;
			uint r = index_ & 3U;
			switch (r)
			{
				case 0:
					return state_[i].x0_;
				case 1:
					return state_[i].x1_;
				case 2:
					return state_[i].x2_;
				default:
					return state_[i].x3_;
			}
		}
	}
}
