using NuGet.Frameworks;
using System.Diagnostics;

namespace cslm.test
{
    public class TestLoad
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test1()
        {
			string path = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), "..\\..\\..\\..\\Qwen2-0.5B-Instruct.calm"));
            Tensors tensors = Tensors.OpenAsync(path).Result;
			Assert.Pass();
        }
    }
}