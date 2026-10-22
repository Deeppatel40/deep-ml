import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here

	probabilities = []

	for fx in features:
		z= sum(w*x for w,x in zip(weights,fx)) + bias
		prob = 1/(1+math.exp(-z))
		probabilities.append(round(prob,4))

	mse = sum((p-y)**2 for p,y in zip(probabilities,labels)) / len(labels)
	mse = round(mse, 4)

	return probabilities, mse