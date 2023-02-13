package moa.classifiers.functions;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import weka.core.Utils;

public class MultinomialLogisticRegression extends AbstractClassifier  implements MultiClassClassifier {

    private static final long serialVersionUID = 221L;

    private int nbInputs;
    private int nbClasses;
    private double regularizationFactor;
    private double learningRate;

    private double[][] weights;

    private boolean shouldReset = true;

    @Override
    public String getPurposeString() {
        return "Logistic regression: A simple logistic regression based classifier.";
    }

    @Override
    public void resetLearningImpl() {
        shouldReset = true;        
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if(shouldReset) {
            shouldReset = false;

            nbInputs = inst.numAttributes();
            nbClasses = inst.numClasses();

            regularizationFactor = 0.5;
            learningRate = 0.005;

            weights = new double[inst.numClasses()][nbInputs];
        }

        double[] inputValues = new double[inst.numInputAttributes()];
        for (int i = 0; i < inst.numInputAttributes(); ++i)
            inputValues[i] = inst.valueInputAttribute(i);
        
        int groundTruth = (int) inst.classValue();

        double[] probabilities = new double[inst.numClasses()];
        for (int i = 0; i < inst.numClasses() - 1; ++i)
            probabilities[i] = dotProduct(inputValues, weights[i]);
        
        softmax(probabilities);

        for (int i = 0; i < inst.numClasses() - 1; ++i) {
            double[] classWeights = weights[i];

            boolean isGroundTruthClass = i == groundTruth;
            double error = isGroundTruthClass ? 1.0 - probabilities[i] : -probabilities[i];

            classWeights[nbInputs - 1] += learningRate * error;
            for (int j = 0; j < nbInputs - 1; ++j)
                classWeights[j] += learningRate * error * inputValues[j];
            
            if (regularizationFactor > 0.0)
                for (int j = 0; j < nbInputs - 1; ++j)
                    classWeights[j] -= learningRate * regularizationFactor * classWeights[j];
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (shouldReset) return new double[inst.numClasses()];

        double[] inputValues = new double[inst.numInputAttributes()];
        for (int i = 0; i < inst.numInputAttributes(); ++i)
            inputValues[i] = inst.valueInputAttribute(i);

        double[] votes = classify(inputValues);

        try {
            Utils.normalize(votes);
        } catch(Exception e) {}

        return votes;
    }

    private double[] classify(double[] input) {
        double[] posteriori = new double[nbClasses];

        posteriori[nbClasses - 1] = 0.0;

        for (int i = 0; i < nbClasses - 1; ++i)
            posteriori[i] = dotProduct(input, weights[i]);
            
        return softmax(posteriori);
    }

    private double dotProduct(double[] input, double[] weights) {
        double product = 0;

        for (int i = 0; i < input.length; ++i)
            product += input[i] * weights[i];
        
        return product;
    }

    private double[] softmax(double[] posteriori) {
        double maximum = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < posteriori.length; ++i) {
            if (posteriori[i] > maximum) {
                maximum = posteriori[i];
            }
        }

        double normalizer = 0.0;
        for (int i = 0; i < posteriori.length; ++i) {
            double exp = Math.exp(posteriori[i] - maximum);
            posteriori[i] = exp;
            normalizer += exp;
        }

        for (int i = 0; i < posteriori.length; ++i)
            posteriori[i] /= normalizer;
        
        return posteriori;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }
    
    @Override
    public void getModelDescription(StringBuilder out, int indent) {}

    @Override
    public boolean isRandomizable() {
        return false;
    }
}