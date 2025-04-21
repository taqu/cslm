using System.CommandLine;
using System.CommandLine.Parsing;

namespace cslm.run
{
	internal class Program
	{
		private static readonly DateTime UNIX_EPOCH = new DateTime(1970, 1, 1, 0, 0, 0, 0);
		private static ulong GetUnixTime(DateTime targetTime)
		{
			targetTime = targetTime.ToUniversalTime();
			TimeSpan elapsedTime = targetTime - UNIX_EPOCH;
			return (ulong)elapsedTime.TotalMilliseconds;
		}

		static void Main(string[] args)
		{
			cslm.Device device = new cslm.Device();
			device.Initialize();
			cslm.Context context = new cslm.Context();
			{
				RootCommand rootCommand = new RootCommand
				{
					Description = "Run a trained model"
				};
				Argument<string> checkpoint = new Argument<string>("checkpoint");
				rootCommand.AddArgument(checkpoint);
				Option<float> temperature = new Option<float>(aliases: new string[] { "-t" }, description: "temperature in [0,inf], default 1.0", getDefaultValue: () => 1.0f);
				Option<float> pvalue = new Option<float>(aliases: new string[] { "-p" }, description: "p value in min-p (cutoff) sampling in [0,1] default 0.1", getDefaultValue: () => 0.1f);
				Option<ulong> seed = new Option<ulong>(aliases: new string[] { "-s" }, description: "random seed, default UtcNow", getDefaultValue: () => GetUnixTime(System.DateTime.UtcNow));
				Option<int> steps = new Option<int>(aliases: new string[] { "-n" }, description: "number of steps to run for, default 256. 0 = max_seq_len, -1 = infinite", getDefaultValue: () => 256);
				Option<int> sequences = new Option<int>(aliases: new string[] { "-r" }, description: "number of sequences to decode, default 1", getDefaultValue: () => 1);
				Option<int> context_length = new Option<int>(aliases: new string[] { "-c" }, description: "context length, default to model-specific maximum", getDefaultValue: () => -1);
				Option<string> input = new Option<string>(aliases: new string[] { "-i" }, description: "input prompt (- to read from stdin)", getDefaultValue: () => string.Empty);
				Option<string> perplexity = new Option<string>(aliases: new string[] { "-x" }, description: "compute perplexity for text file", getDefaultValue: () => string.Empty);
				Option<string> system_prompt = new Option<string>(aliases: new string[] { "-y" }, description: "chat mode with a system prompt", getDefaultValue: () => string.Empty);
				rootCommand.AddOption(temperature);
				rootCommand.AddOption(pvalue);
				rootCommand.AddOption(seed);
				rootCommand.AddOption(steps);
				rootCommand.AddOption(sequences);
				rootCommand.AddOption(context_length);
				rootCommand.AddOption(input);
				rootCommand.AddOption(perplexity);
				rootCommand.AddOption(system_prompt);

				ParseResult result = rootCommand.Parse(args);

				context.model_ = result.GetValueForArgument<string>(checkpoint);
				context.temperature_ = result.GetValueForOption<float>(temperature);
				context.pvalue_ = result.GetValueForOption<float>(pvalue);
				context.seed_ = result.GetValueForOption<ulong>(seed);
				context.steps_ = result.GetValueForOption<int>(steps);
				context.sequences_ = result.GetValueForOption<int>(sequences);
				context.context_length_ = result.GetValueForOption<int>(context_length);
				context.input_ = result.GetValueForOption<string>(input);
				context.perplexity_ = result.GetValueForOption<string>(perplexity);
				context.system_prompt_ = result.GetValueForOption<string>(system_prompt);

				if (0 < result.Errors.Count)
				{
					foreach (ParseError error in result.Errors)
					{
						Console.WriteLine(error.Message);
					}
					return;
				}
			}


			Model model = new Model();
			if (!model.initialize(context.model_, context.context_length_))
			{
				return;
			}
			model.run(context);
        }
	}
}
