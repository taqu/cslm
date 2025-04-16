
using System;
using static cslm.Random;

namespace cslm
{
	public struct Random
	{
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
		private uint[] state_;

		public Random(uint s=0)
		{
			index_ = 0;
			state_ = new uint[SFMT_N32];
            seed(s);
		}

		private void check_modification(int i, uint parity)
		{
            uint work = 1;
            for (int j = 0; j < 32; ++j)
            {
                if ((work & parity) != 0)
                {
                    state_[i] ^= work;
                    return;
                }
                work = work << 1;
            }

        }

        private void period_certification()
        {
            uint inner = 0;
			{
                inner ^= state_[0] & SFMT_PARITY1;
                inner ^= state_[1] & SFMT_PARITY2;
                inner ^= state_[2] & SFMT_PARITY3;
                inner ^= state_[3] & SFMT_PARITY4;
            }
            for (int i = 16; 0<i; i >>= 1)
            {
                inner ^= inner >> i;
            }
            inner &= 1;
            if (inner == 1)
            {
                return;
            }
			check_modification(0, SFMT_PARITY1);
            check_modification(1, SFMT_PARITY2);
            check_modification(2, SFMT_PARITY3);
            check_modification(3, SFMT_PARITY4);
        }

		private void generate()
		{
            const int SL2_x8 = SFMT_SL2 * 8;
            const int SR2_x8 = SFMT_SR2 * 8;
            const int SL2_ix8 = 64 - SFMT_SL2 * 8;
            const int SR2_ix8 = 64 - SFMT_SR2 * 8;

            int a = 0;
            int b = SFMT_POS1 * 4;
            int c = (SFMT_N - 2) * 4;
            int d = (SFMT_N - 1) * 4;
            do
            {
                ulong xh = ((ulong)state_[a + 3] << 32) | state_[a + 2];
                ulong xl = ((ulong)state_[a + 1] << 32) | state_[a + 0];
                ulong yh = xh << (SL2_x8) | xl >> (SL2_ix8);
                ulong yl = xl << (SL2_x8);
                xh = ((ulong)state_[c + 3] << 32) | state_[c + 2];
                xl = ((ulong)state_[c + 1] << 32) | state_[c + 0];
                yh ^= xh >> (SR2_x8);
                yl ^= xl >> (SR2_x8) | xh << (SR2_ix8);

                state_[a + 3] = state_[a + 3] ^ ((state_[b + 3] >> SFMT_SR1) & SFMT_MSK4) ^ (state_[d + 3] << SFMT_SL1) ^ ((uint)(yh >> 32));
                state_[a + 2] = state_[a + 2] ^ ((state_[b + 2] >> SFMT_SR1) & SFMT_MSK3) ^ (state_[d + 2] << SFMT_SL1) ^ ((uint)yh);
                state_[a + 1] = state_[a + 1] ^ ((state_[b + 1] >> SFMT_SR1) & SFMT_MSK2) ^ (state_[d + 1] << SFMT_SL1) ^ ((uint)(yl >> 32));
                state_[a + 0] = state_[a + 0] ^ ((state_[b + 0] >> SFMT_SR1) & SFMT_MSK1) ^ (state_[d + 0] << SFMT_SL1) ^ ((uint)yl);

                c = d;
                d = a;
                a += 4;
                b += 4;
                if (SFMT_N32<=b)
                {
                    b = 0;
                }
            } while (a < SFMT_N32);
        }

		public void seed(uint s)
		{
			state_[0] = s;
			for(int i=1; i<SFMT_N32; ++i)
			{
                state_[i] = (uint)(1812433253U * (state_[i - 1] ^ (state_[i - 1] >> 30)) + i);
            }
			index_ = SFMT_N32;
                period_certification();
        }

        public uint rand()
		{
            if (SFMT_N32 <= index_)
            {
                generate();
                index_ = 1;
                return state_[0];
            }
            else
            {
                return state_[index_++];
            }
        }

        public float frand()
        {
            uint u = rand();
            return (u >> 8) / 16777216.0f;
        }
	}
}
