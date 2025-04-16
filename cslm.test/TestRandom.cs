
using System.Text;

namespace cslm.test
{
    public class TestRandom
    {
		[SetUp]
		public void Setup()
		{
		}

		[Test]
		public void Test1()
        {
			Random random = new Random(1234);
			Assert.That(random.rand(), Is.EqualTo(1196421539));
			random.seed(1234UL);

			System.Security.Cryptography.RandomNumberGenerator sec_random = new System.Security.Cryptography.RNGCryptoServiceProvider();
			byte[] s = new byte[Random.SFMT_N8];
			sec_random.GetBytes(s);
			random.seed(s);
			for (int i = 0; i < 100000; ++i)
			{
				float f = random.frand();
				Assert.True(0.0f <= f && f < 1.0f);
			}
		}
	}
}

