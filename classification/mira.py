# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util

PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys()  # this could be useful for your code later...

        if (self.automaticTuning):
            caps = [0.002, 0.004, 0.008]
        else:
            caps = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, caps)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, caps):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        weightsPerCap = dict()
        for cap in caps:
            print "Starting training for c = ", str(cap)
            for iteration in range(self.max_iterations):
                self.doIteration(cap, iteration, trainingData, trainingLabels)
            weightsPerCap[cap] = self.weights
            self.initializeWeightsToZero()

        accuracies = util.Counter()
        for cap, weights in weightsPerCap.iteritems():
            self.weights = weights
            accuracy = self.getAccuracy(validationData, validationLabels)
            print "Accuracy on validation set for c = ", str(cap), ": ", str(accuracy)
            accuracies[cap] = accuracy

        self.weights = weightsPerCap[accuracies.argMax()]

    def getAccuracy(self, validationData, validationLabels):
        numberCorrect = 0
        for i in range(len(validationData)):
            guessedLabel = self.guessLabel(validationData[i])
            label = validationLabels[i]
            if guessedLabel == label:
                numberCorrect += 1

        return float(numberCorrect) / float(len(validationData))

    def doIteration(self, ceiling, iteration, trainingData, trainingLabels):
        print "Starting iteration ", iteration
        for i in range(len(trainingData)):
            self.processInstance(trainingData[i], trainingLabels[i], ceiling)

    def processInstance(self, features, label, ceiling):
        guessedLabel = self.guessLabel(features)
        if guessedLabel != label:
            scaling = min(ceiling, self.calculateScaling(features, label, guessedLabel))
            scaledFeatures = self.calculateScaledFeatures(features, scaling)
            self.weights[label] += scaledFeatures
            self.weights[guessedLabel] -= scaledFeatures

    @staticmethod
    def calculateScaledFeatures(features, scaling):
        scaledFeatures = util.Counter()
        for key in features:
            scaledFeatures[key] = scaling * features[key]
        return scaledFeatures

    def guessLabel(self, features):
        scores = util.Counter()
        for label in self.legalLabels:
            scores[label] = features * self.weights[label]

        return scores.argMax()

    def calculateScaling(self, features, actualLabel, guessedLabel):
        guessedWeights = self.weights[guessedLabel]
        actualWeights = self.weights[actualLabel]
        dividend = (guessedWeights - actualWeights) * features + 1.0
        divisor = 2 * (features * features)

        return dividend / divisor

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """

        return self.weights[label].sortedKeys()[:100]
