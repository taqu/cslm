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
			string path = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), "..\\..\\..\\..\\tinyllm.calm"));
            Tensors tensors = Tensors.OpenAsync(path).Result;
            Assert.IsNotNull(tensors);
            for(int i=0; i < tensors.num_metadata(); ++i)
            {
                Debug.WriteLine(string.Format("{0}: {1}", tensors.get_metadata_key(i), tensors.get_metadata_value(i)));
            }
			for (int i = 0; i < tensors.num_tensors(); ++i)
			{
                Tensor tensor = tensors.get_tensor(i);
				Debug.WriteLine(string.Format("{0}({1}) {2}x{3}x{4}x{5} {6}:{7}:{8}", tensor.name_, tensor.dtype_, tensor.shape0_, tensor.shape1_, tensor.shape2_, tensor.shape3_, tensor.data_, tensor.size_, tensor.data_+tensor.size_));
			}
		}
	}
}