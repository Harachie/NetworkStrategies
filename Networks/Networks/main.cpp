#include <cstring>
#include <cstdlib>

void rowCalculateNeuron(const float *inputSets, const float *weights, const size_t index, const size_t setSize, const size_t predictors, const size_t nextLayerNeuronsCount, float *results)
{
	size_t inputsPerSet = predictors + 1;
	size_t inputsPerNextLayer = nextLayerNeuronsCount + 1;
	float sum;

	for (size_t setIndex = 0; setIndex < setSize; setIndex++)
	{
		sum = 0.0f;

		for (size_t predictorIndex = 0; predictorIndex < inputsPerSet; predictorIndex++)
		{
			sum += inputSets[setIndex * inputsPerSet + predictorIndex] * weights[predictorIndex];
		}

		results[setIndex * inputsPerNextLayer + index] = sum;
	}

}

void rowCalculateLayer(const float *inputSets, const float *weights, const size_t setSize, const size_t predictors, const size_t nextLayerNeuronsCount, float *results)
{
	size_t weightsPerNeuron = predictors + 1;

	for (size_t neuronIndex = 0; neuronIndex < nextLayerNeuronsCount; neuronIndex++)
	{
		rowCalculateNeuron(inputSets, weights + neuronIndex * weightsPerNeuron, neuronIndex, setSize, predictors, nextLayerNeuronsCount, results);
	}
}

void rowSetWeights(float *weights, float *singleNeuronWeights, const size_t weightsCount, const size_t index)
{
	memcpy(weights + index * weightsCount, singleNeuronWeights, weightsCount * sizeof(float));
}

void rowSetInputs(float *inputs, float *values, const size_t predictors, const size_t index)
{
	memcpy(inputs + index * (predictors + 1), values, predictors * sizeof(float));
}

void rowFillBiasValues(float *results, const size_t setSize, const size_t predictors) //for hidden layers to the next layer, the predictors are the hidden neurons count
{
	for (size_t i = 0; i < setSize; i++)
	{
		results[i * (predictors + 1) + predictors] = 1.0f; //bias is always 1, for row wise memory layout its the last column (row width = predictors + 1
	}
}



// column wise memory layout (better for GPU)


void columnFillBiasValues(float *results, const size_t setSize, const size_t predictors) //for hidden layers to the next layer, the predictors are the hidden neurons count
{
	size_t startIndex;

	startIndex = setSize * predictors;

	for (size_t i = 0; i < setSize; i++)
	{
		results[startIndex + i] = 1.0f; //bias is always 1, in column layout it's the last row
	}
}

void columnSetInputs(float *inputs, float *values, const size_t predictors, const size_t setSize, const size_t index)
{
	for (size_t i = 0; i < predictors; i++)
	{
		inputs[(i * setSize) + index] = values[i];
	}
}

void testRow1LayerNetwork()
{
	size_t setSize = 5;
	size_t inputPredictors = 2;
	size_t firstLayerPredictors = 3;
	size_t secondLayerPredictors = 4;

	float *input1 = new float[inputPredictors];
	float *input2 = new float[inputPredictors];
	float *input3 = new float[inputPredictors];
	float *input4 = new float[inputPredictors];
	float *input5 = new float[inputPredictors];

	float *inputs = new float[setSize * (inputPredictors + 1)]; //einmal bias wert = 1

	float *inputToFirstLayerWeights = new float[firstLayerPredictors * (inputPredictors + 1)];
	float *inputToFirstLayerResults = new float[setSize * (firstLayerPredictors + 1)]; //ein bias

	float *inputNeuron1Weights = new float[inputPredictors + 1]; //ein bias
	float *inputNeuron2Weights = new float[inputPredictors + 1]; //ein bias
	float *inputNeuron3Weights = new float[inputPredictors + 1]; //ein bias


	input1[0] = 1.0f;
	input1[1] = 2.0f;

	input2[0] = 3.0f;
	input2[1] = 4.0f;

	input3[0] = 5.0f;
	input3[1] = 6.0f;

	input4[0] = 7.0f;
	input4[1] = 8.0f;

	input5[0] = 9.0f;
	input5[1] = 10.0f;

	inputNeuron1Weights[0] = 1.0f;
	inputNeuron1Weights[1] = 2.0f;
	inputNeuron1Weights[2] = 3.0f; //bias

	inputNeuron2Weights[0] = 4.0f;
	inputNeuron2Weights[1] = 5.0f;
	inputNeuron2Weights[2] = 6.0f; //bias

	inputNeuron3Weights[0] = 7.0f;
	inputNeuron3Weights[1] = 8.0f;
	inputNeuron3Weights[2] = 9.0f; //bias

	rowFillBiasValues(inputs, setSize, inputPredictors);
	rowFillBiasValues(inputToFirstLayerResults, setSize, firstLayerPredictors);

	rowSetInputs(inputs, input1, inputPredictors, 0);
	rowSetInputs(inputs, input2, inputPredictors, 1);
	rowSetInputs(inputs, input3, inputPredictors, 2);
	rowSetInputs(inputs, input4, inputPredictors, 3);
	rowSetInputs(inputs, input5, inputPredictors, 4);

	rowSetWeights(inputToFirstLayerWeights, inputNeuron1Weights, inputPredictors + 1, 0);
	rowSetWeights(inputToFirstLayerWeights, inputNeuron2Weights, inputPredictors + 1, 1);
	rowSetWeights(inputToFirstLayerWeights, inputNeuron3Weights, inputPredictors + 1, 2);

	rowCalculateLayer(inputs, inputToFirstLayerWeights, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);

	rowCalculateNeuron(inputs, inputNeuron1Weights, 0, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputNeuron2Weights, 1, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputNeuron3Weights, 2, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);

	rowCalculateNeuron(inputs, inputToFirstLayerWeights + 0 * (inputPredictors + 1), 0, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputToFirstLayerWeights + 1 * (inputPredictors + 1), 1, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputToFirstLayerWeights + 2 * (inputPredictors + 1), 2, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);


}

void testColumn1LayerNetwork()
{
	size_t setSize = 5;
	size_t inputPredictors = 2;
	size_t firstLayerPredictors = 3;
	size_t secondLayerPredictors = 4;

	float *input1 = new float[inputPredictors];
	float *input2 = new float[inputPredictors];
	float *input3 = new float[inputPredictors];
	float *input4 = new float[inputPredictors];
	float *input5 = new float[inputPredictors];

	float *inputs = new float[setSize * (inputPredictors + 1)]; //einmal bias wert = 1

	float *inputToFirstLayerWeights = new float[firstLayerPredictors * (inputPredictors + 1)];
	float *inputToFirstLayerResults = new float[setSize * (firstLayerPredictors + 1)]; //ein bias

	float *inputNeuron1Weights = new float[inputPredictors + 1]; //ein bias
	float *inputNeuron2Weights = new float[inputPredictors + 1]; //ein bias
	float *inputNeuron3Weights = new float[inputPredictors + 1]; //ein bias


	input1[0] = 1.0f;
	input1[1] = 2.0f;

	input2[0] = 3.0f;
	input2[1] = 4.0f;

	input3[0] = 5.0f;
	input3[1] = 6.0f;

	input4[0] = 7.0f;
	input4[1] = 8.0f;

	input5[0] = 9.0f;
	input5[1] = 10.0f;

	inputNeuron1Weights[0] = 1.0f;
	inputNeuron1Weights[1] = 2.0f;
	inputNeuron1Weights[2] = 3.0f; //bias

	inputNeuron2Weights[0] = 4.0f;
	inputNeuron2Weights[1] = 5.0f;
	inputNeuron2Weights[2] = 6.0f; //bias

	inputNeuron3Weights[0] = 7.0f;
	inputNeuron3Weights[1] = 8.0f;
	inputNeuron3Weights[2] = 9.0f; //bias

	columnFillBiasValues(inputs, setSize, inputPredictors);
	columnFillBiasValues(inputToFirstLayerResults, setSize, firstLayerPredictors);

	columnSetInputs(inputs, input1, inputPredictors, setSize, 0);
	columnSetInputs(inputs, input2, inputPredictors, setSize, 1);
	columnSetInputs(inputs, input3, inputPredictors, setSize, 2);
	columnSetInputs(inputs, input4, inputPredictors, setSize, 3);
	columnSetInputs(inputs, input5, inputPredictors, setSize, 4);

	rowSetWeights(inputToFirstLayerWeights, inputNeuron1Weights, inputPredictors + 1, 0);
	rowSetWeights(inputToFirstLayerWeights, inputNeuron2Weights, inputPredictors + 1, 1);
	rowSetWeights(inputToFirstLayerWeights, inputNeuron3Weights, inputPredictors + 1, 2);

	rowCalculateLayer(inputs, inputToFirstLayerWeights, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);

	rowCalculateNeuron(inputs, inputNeuron1Weights, 0, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputNeuron2Weights, 1, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputNeuron3Weights, 2, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);

	rowCalculateNeuron(inputs, inputToFirstLayerWeights + 0 * (inputPredictors + 1), 0, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputToFirstLayerWeights + 1 * (inputPredictors + 1), 1, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateNeuron(inputs, inputToFirstLayerWeights + 2 * (inputPredictors + 1), 2, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);


}

void testRow2LayerNetwork()
{
	size_t setSize = 5;
	size_t inputPredictors = 2;
	size_t firstLayerPredictors = 3;
	size_t secondLayerPredictors = 4;

	float *input1 = new float[inputPredictors];
	float *input2 = new float[inputPredictors];
	float *input3 = new float[inputPredictors];
	float *input4 = new float[inputPredictors];
	float *input5 = new float[inputPredictors];

	float *inputs = new float[setSize * (inputPredictors + 1)]; //einmal bias wert = 1

	float *inputToFirstLayerWeights = new float[firstLayerPredictors * (inputPredictors + 1)];
	float *firstLayerToSecondLayerWeights = new float[secondLayerPredictors * (firstLayerPredictors + 1)];

	float *inputToFirstLayerResults = new float[setSize * (firstLayerPredictors + 1)]; //ein bias
	float *firstLayerToSecondLayerResults = new float[setSize * (secondLayerPredictors + 1)]; //ein bias

	float *inputNeuron1Weights = new float[inputPredictors + 1]; //ein bias
	float *inputNeuron2Weights = new float[inputPredictors + 1]; //ein bias
	float *inputNeuron3Weights = new float[inputPredictors + 1]; //ein bias

	float *hiddenNeuron1Weights = new float[firstLayerPredictors + 1]; //ein bias
	float *hiddenNeuron2Weights = new float[firstLayerPredictors + 1]; //ein bias
	float *hiddenNeuron3Weights = new float[firstLayerPredictors + 1]; //ein bias
	float *hiddenNeuron4Weights = new float[firstLayerPredictors + 1]; //ein bias


	input1[0] = 1.0f;
	input1[1] = 2.0f;

	input2[0] = 3.0f;
	input2[1] = 4.0f;

	input3[0] = 5.0f;
	input3[1] = 6.0f;

	input4[0] = 7.0f;
	input4[1] = 8.0f;

	input5[0] = 9.0f;
	input5[1] = 10.0f;

	inputNeuron1Weights[0] = 1.0f;
	inputNeuron1Weights[1] = 2.0f;
	inputNeuron1Weights[2] = 3.0f; //bias

	inputNeuron2Weights[0] = 4.0f;
	inputNeuron2Weights[1] = 5.0f;
	inputNeuron2Weights[2] = 6.0f; //bias

	inputNeuron3Weights[0] = 7.0f;
	inputNeuron3Weights[1] = 8.0f;
	inputNeuron3Weights[2] = 9.0f; //bias

	hiddenNeuron1Weights[0] = 1.0f;
	hiddenNeuron1Weights[1] = 2.0f;
	hiddenNeuron1Weights[2] = 3.0f;
	hiddenNeuron1Weights[3] = 4.0f; //bias

	hiddenNeuron2Weights[0] = 5.0f;
	hiddenNeuron2Weights[1] = 6.0f;
	hiddenNeuron2Weights[2] = 7.0f;
	hiddenNeuron2Weights[3] = 8.0f; //bias

	hiddenNeuron3Weights[0] = 9.0f;
	hiddenNeuron3Weights[1] = 10.0f;
	hiddenNeuron3Weights[2] = 11.0f;
	hiddenNeuron3Weights[3] = 12.0f; //bias

	hiddenNeuron4Weights[0] = 13.0f;
	hiddenNeuron4Weights[1] = 14.0f;
	hiddenNeuron4Weights[2] = 15.0f;
	hiddenNeuron4Weights[3] = 16.0f; //bias

	rowFillBiasValues(inputs, setSize, inputPredictors);
	rowFillBiasValues(inputToFirstLayerResults, setSize, firstLayerPredictors);
	rowFillBiasValues(firstLayerToSecondLayerResults, setSize, secondLayerPredictors);

	rowSetInputs(inputs, input1, inputPredictors, 0);
	rowSetInputs(inputs, input2, inputPredictors, 1);
	rowSetInputs(inputs, input3, inputPredictors, 2);
	rowSetInputs(inputs, input4, inputPredictors, 3);
	rowSetInputs(inputs, input5, inputPredictors, 4);

	rowSetWeights(inputToFirstLayerWeights, inputNeuron1Weights, inputPredictors + 1, 0);
	rowSetWeights(inputToFirstLayerWeights, inputNeuron2Weights, inputPredictors + 1, 1);
	rowSetWeights(inputToFirstLayerWeights, inputNeuron3Weights, inputPredictors + 1, 2);

	rowSetWeights(firstLayerToSecondLayerWeights, hiddenNeuron1Weights, firstLayerPredictors + 1, 0);
	rowSetWeights(firstLayerToSecondLayerWeights, hiddenNeuron2Weights, firstLayerPredictors + 1, 1);
	rowSetWeights(firstLayerToSecondLayerWeights, hiddenNeuron3Weights, firstLayerPredictors + 1, 2);
	rowSetWeights(firstLayerToSecondLayerWeights, hiddenNeuron4Weights, firstLayerPredictors + 1, 3);

	rowCalculateLayer(inputs, inputToFirstLayerWeights, setSize, inputPredictors, firstLayerPredictors, inputToFirstLayerResults);
	rowCalculateLayer(inputToFirstLayerResults, firstLayerToSecondLayerWeights, setSize, firstLayerPredictors, secondLayerPredictors, firstLayerToSecondLayerResults);


}

int main()
{
	testRow1LayerNetwork();
	testRow2LayerNetwork();

	testColumn1LayerNetwork();

	return 0;
}