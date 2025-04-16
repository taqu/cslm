
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
        }
    }
}

