using System;

namespace NeuralNetwork
{
	public class Network
	{
		private readonly float learningRate;
		private Matrix inputWeights;
		private Matrix outputWeights;

		public Network(int inputNodeCount, int hiddenNodeCount, int outputNodeCount, float learningRate)
		{
			this.learningRate = learningRate;

			inputWeights = new Matrix(hiddenNodeCount, inputNodeCount);
			outputWeights = new Matrix(outputNodeCount, hiddenNodeCount);

			var random = new Random();
			RandomizeWeights(inputWeights, random);
			RandomizeWeights(outputWeights, random);
		}

		public void Train(float[] inputs, float[] targets)
		{
			Matrix inputsMatrix = Matrix.ConvertAsColumn(inputs);
			Matrix targetsMatrix = Matrix.ConvertAsColumn(targets);

			Matrix hiddenLayerOutputs = MathUtility.Sigmoid(inputWeights * inputsMatrix);
			Matrix endOutputs = MathUtility.Sigmoid(outputWeights * hiddenLayerOutputs);

			Matrix endErrors = targetsMatrix - endOutputs;
			Matrix hiddenLayerErrors = Matrix.Transpose(outputWeights) * endErrors;

			outputWeights += learningRate * endErrors * endOutputs * (1.0f - endOutputs) * Matrix.Transpose(hiddenLayerOutputs);
			inputWeights += learningRate * hiddenLayerErrors * hiddenLayerOutputs * (1.0f - hiddenLayerOutputs) * Matrix.Transpose(inputsMatrix);
		}

		public float[] Query(float[] inputs)
		{
			Matrix inputsMatrix = Matrix.ConvertAsColumn(inputs);

			Matrix hiddenLayerOutputs = MathUtility.Sigmoid(inputWeights * inputsMatrix);
			Matrix endOutputs = MathUtility.Sigmoid(outputWeights * hiddenLayerOutputs);

			return Matrix.Flatten(endOutputs);
		}

		private void RandomizeWeights(Matrix weights, Random random)
		{
			for(int x = 0; x < weights.RowCount; ++x)
			{
				for(int y = 0; y < weights.ColumnCount; ++y)
				{
					weights[x, y] = (float)random.NextDouble() - 0.5f;
				}
			}
		}
	}
}
