using System.Runtime.InteropServices;

namespace cslm
{
	public struct Blob
	{
		public byte[] data_;
		public long offset_;
		public long length_;
	}

	public static class Util
	{
		public static uint CalcHash(ReadOnlySpan<uint> data)
		{
            uint hash = 0;
			for (int i = 0; i < data.Length; ++i)
			{
				hash = hash * 5 + data[i];
            }
			return hash;
        }

        public static uint CalcHash(ReadOnlySpan<float> data)
        {
			return CalcHash(MemoryMarshal.Cast<float, uint>(data));
        }
    }
}

