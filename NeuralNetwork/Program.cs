using System;

namespace NeuralNetwork
{
	class Program
	{
		static void Main(string[] args)
		{
			Matrix x = Matrix.ConvertAsColumn(new[] { 1.0f, 2.0f });
			Matrix y = Matrix.ConvertAsRow(new[] { 3.0f, 4.0f });
			var z = x * y;

			var recognition = new DigitRecognition();
			recognition.Run();
		}
	}
}
