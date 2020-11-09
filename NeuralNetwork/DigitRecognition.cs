using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
	public class DigitRecognition
	{
		private class TrainingSample
		{
			public int Digit;
			public int[] Values;
		}

		public void Run()
		{
			// Input image is 28 x 28 pixels, so 784 inputs.
			// 10 digits, so 10 outputs.
			var network = new Network(28 * 28, 100, 10, 0.3f);
			
			string appPath = AppDomain.CurrentDomain.BaseDirectory;
			string trainingSetPath = Path.Combine(appPath, "Data", "mnist_train.csv");
			string testingSetPath = Path.Combine(appPath, "Data", "mnist_test.csv");

			Console.WriteLine("Parsing training data...");
			var trainingSamples = ParseTrainingTable(trainingSetPath);
			var stopwatch = new Stopwatch();

			Console.WriteLine($"Training on {trainingSamples.Count} samples...");
			stopwatch.Start();

			foreach(TrainingSample sample in trainingSamples)
			{
				var normalizedValues = sample.Values.Select(channel => NormalizeColorChannel(channel)).ToArray();
				var targets = Enumerable.Repeat(0.01f, 10).ToArray();
				targets[sample.Digit] = 0.99f;

				network.Train(normalizedValues, targets);
			}

			stopwatch.Stop();
			Console.WriteLine($"Trained for {stopwatch.Elapsed.TotalSeconds} seconds");

			Console.WriteLine("Parsing testing data...");
			var testingSamples = ParseTrainingTable(testingSetPath);
			int correctAnswerCount = 0;
			Console.WriteLine("Testing...");

			foreach(TrainingSample sample in testingSamples)
			{
				var normalizedValues = sample.Values.Select(channel => NormalizeColorChannel(channel)).ToArray();
				float[] outputs = network.Query(normalizedValues);
				int indexOfMaxValue = -1;

				for(int i = 0; i < outputs.Length; ++i)
				{
					if(indexOfMaxValue == -1 || outputs[indexOfMaxValue] < outputs[i])
					{
						indexOfMaxValue = i;
					}
				}

				if(indexOfMaxValue == sample.Digit)
				{
					++correctAnswerCount;
				}
			}

			Console.WriteLine($"Tested {testingSamples.Count} with {correctAnswerCount} correct recognitions");
			Console.WriteLine($"Network perfomance is {(double)correctAnswerCount / testingSamples.Count}");
		}

		private float NormalizeColorChannel(int channel)
		{
			return (float)channel / 255 * 0.99f + 0.01f;
		}

		private List<TrainingSample> ParseTrainingTable(string path)
		{
			var table = ParseCsv(path);
			var samples = new List<TrainingSample>();

			foreach(var row in table)
			{
				int digit = int.Parse(row[0]);
				var values = row.Skip(1).Select(x => int.Parse(x)).ToArray();
				var sample = new TrainingSample
				{
					Digit = digit,
					Values = values
				};
				samples.Add(sample);
			}

			return samples;
		}

		private string[][] ParseCsv(string path)
		{
			var lines = File.ReadAllLines(path);
			return lines.Select(x => x.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)).ToArray();
		}
	}
}
