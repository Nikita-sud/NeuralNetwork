import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    private int numLayers;
    @SuppressWarnings("unused")
    private int[] sizes;
    private double[][][] weights;
    private double[][] biases;

    static class backPropResult {
        double[][][] delta_nabla_w;
        double[][] delta_nabla_b;
        public backPropResult(double[][][] delta_nabla_w, double[][] delta_nabla_b) {
            this.delta_nabla_w = delta_nabla_w;
            this.delta_nabla_b = delta_nabla_b;
        }
    }

    public NeuralNetwork(int[] sizes) {
        this.numLayers = sizes.length;
        this.sizes = sizes;
        this.biases = new double[sizes.length - 1][];
        this.weights = new double[sizes.length - 1][][];
        Random rand = new Random();

        for (int i = 1; i < sizes.length; i++) {
            biases[i - 1] = new double[sizes[i]];
            weights[i - 1] = new double[sizes[i]][sizes[i - 1]];

            for (int j = 0; j < sizes[i]; j++) {
                biases[i - 1][j] = rand.nextGaussian();
                for (int k = 0; k < sizes[i - 1]; k++) {
                    weights[i - 1][j][k] = rand.nextGaussian();
                }
            }
        }
    }

    public void SGD(List<ImageData> training_data, int epochs, int mini_batch_size, double eta, List<ImageData> test_data) {
        int n_test = test_data.size();
        @SuppressWarnings("unused")
        int n = training_data.size();

        for (int i = 0; i < epochs; i++) {
            Collections.shuffle(training_data);
            List<List<ImageData>> miniBatches = createMiniBatches(training_data, mini_batch_size);

            miniBatches.parallelStream().forEach(miniBatch -> updateMiniBatch(miniBatch, eta));

            if (n_test > 0) {
                System.out.println("Epoch " + (i + 1) + ": " + evaluate(test_data) + " / " + n_test);
            }
        }
    }

    public void updateMiniBatch(List<ImageData> miniBatch, double eta) {
        double[][] nabla_b = new double[this.biases.length][];
        for (int i = 0; i < this.biases.length; i++) {
            nabla_b[i] = new double[this.biases[i].length];
            Arrays.fill(nabla_b[i], 0.0); // Fill with zeros
        }
        double[][][] nabla_w = new double[this.weights.length][][];
        for (int i = 0; i < this.weights.length; i++) {
            nabla_w[i] = new double[this.weights[i].length][];
            for (int j = 0; j < this.weights[i].length; j++) {
                nabla_w[i][j] = new double[this.weights[i][j].length];
                Arrays.fill(nabla_w[i][j], 0.0); // Fill with zeros
            }
        }

        for (ImageData imageData : miniBatch) {
            backPropResult backPropRes = backprop(imageData.imageData, imageData.label);
            double[][][] delta_nabla_w = backPropRes.delta_nabla_w;
            double[][] delta_nabla_b = backPropRes.delta_nabla_b;

            for (int j = 0; j < nabla_w.length; j++) {
                nabla_w[j] = vectorAddition(nabla_w[j], delta_nabla_w[j]);
            }

            for (int j = 0; j < nabla_b.length; j++) {
                for (int m = 0; m < nabla_b[j].length; m++) {
                    nabla_b[j][m] += delta_nabla_b[j][m];
                }
            }
        }

        for (int j = 0; j < weights.length; j++) {
            for (int m = 0; m < weights[j].length; m++) {
                for (int n = 0; n < weights[j][m].length; n++) {
                    this.weights[j][m][n] -= (eta / miniBatch.size()) * nabla_w[j][m][n];
                }
            }
        }

        for (int j = 0; j < biases.length; j++) {
            for (int m = 0; m < biases[j].length; m++) {
                this.biases[j][m] -= (eta / miniBatch.size()) * nabla_b[j][m];
            }
        }
    }

    public backPropResult backprop(double[][] x, double[][] y) {
        // Initialize gradients for biases and weights
        double[][] nabla_b = new double[this.biases.length][];
        for (int i = 0; i < this.biases.length; i++) {
            nabla_b[i] = new double[this.biases[i].length];
            Arrays.fill(nabla_b[i], 0.0); // Fill with zeros
        }
        double[][][] nabla_w = new double[this.weights.length][][];
        for (int i = 0; i < this.weights.length; i++) {
            nabla_w[i] = new double[this.weights[i].length][];
            for (int j = 0; j < this.weights[i].length; j++) {
                nabla_w[i][j] = new double[this.weights[i][j].length];
                Arrays.fill(nabla_w[i][j], 0.0); // Fill with zeros
            }
        }
        
        // Feedforward
        double[][] activation = x;
        List<double[][]> activations = new ArrayList<>();
        activations.add(x);
        List<double[][]> zs = new ArrayList<>();
        
        for (int i = 0; i < numLayers - 2; i++) {
            double[][] z = multiplyMatrices(weights[i], activation);
            z = addBias(z, biases[i]);
            zs.add(z);
            activation = ReLUVector(z);
            activations.add(activation);
        }
        
        // Последний слой с softmax
        double[][] z = multiplyMatrices(weights[numLayers - 2], activation);
        z = addBias(z, biases[numLayers - 2]);
        zs.add(z);
        activation = softmaxVector(z);
        activations.add(activation);
        
        // Backward pass
        double[][] delta = crossEntropyDelta(activations.get(activations.size() - 1), y);
        nabla_b[nabla_b.length - 1] = getOneDimensionalVector(delta);
        nabla_w[nabla_w.length - 1] = multiplyMatrices(delta, transpose(activations.get(activations.size() - 2)));
        
        for (int i = 2; i < numLayers; i++) {
            z = zs.get(zs.size() - i);
            double[][] sp = primeReLUVector(z);
            delta = multiplyMatrices(transpose(weights[weights.length - i + 1]), delta);
            delta = hadamardProduct(delta, sp);
            nabla_b[nabla_b.length - i] = getOneDimensionalVector(delta);
            nabla_w[nabla_w.length - i] = multiplyMatrices(delta, transpose(activations.get(activations.size() - i - 1)));
        }
        
        return new backPropResult(nabla_w, nabla_b);
    }

    private double[][] crossEntropyDelta(double[][] outputActivations, double[][] y) {
        return vectorSubstraction(outputActivations, y);
    }

    public int predict(double[][] input) {
        double[][] activation = input;
        
        for (int i = 0; i < numLayers - 2; i++) { // Все слои, кроме последнего
            activation = multiplyMatrices(weights[i], activation);
            activation = addBias(activation, biases[i]);
            activation = ReLUVector(activation);
        }
        
        // Последний слой с softmax
        activation = multiplyMatrices(weights[numLayers - 2], activation);
        activation = addBias(activation, biases[numLayers - 2]);
        activation = softmaxVector(activation);
        
        return argMax(activation);
    }

    public int evaluate(List<ImageData> test_data) {
        int correct = 0;
        for (ImageData data : test_data) {
            int predicted = predict(data.imageData);
            int actual = argMax(data.label);
            if (predicted == actual) {
                correct++;
            }
        }
        return correct;
    }

    public static int argMax(double[][] array) {
        int maxIndex = 0;
        double max = array[0][0];
        for (int i = 1; i < array.length; i++) {
            if (array[i][0] > max) {
                max = array[i][0];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static double[] getOneDimensionalVector(double[][] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = x[i][0];
        }
        return result;
    }
    
    public static double[][] get2DimentionalVector(double[] x) {
        double[][] result = new double[x.length][1];
        for (int i = 0; i < x.length; i++) {
            result[i][0] = x[i];
        }
        return result;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double[][] sigmoidVector(double[][] x) {
        for (int i = 0; i < x.length; i++) {
            x[i][0] = sigmoid(x[i][0]);
        }
        return x;
    }

    public static double primeSigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static double[][] primeSigmoidVector(double[][] x) {
        for (int i = 0; i < x.length; i++) {
            x[i][0] = primeSigmoid(x[i][0]);
        }
        return x;
    }

    public static double ReLU(double x) {
        return Math.max(0, x);
    }

    public static double[][] ReLUVector(double[][] x) {
        for (int i = 0; i < x.length; i++) {
            x[i][0] = ReLU(x[i][0]);
        }
        return x;
    }

    public static double primeReLU(double x) {
        return x > 0 ? 1 : 0;
    }

    public static double[][] primeReLUVector(double[][] x) {
        for (int i = 0; i < x.length; i++) {
            x[i][0] = primeReLU(x[i][0]);
        }
        return x;
    }

    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposedMatrix = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }

        return transposedMatrix;
    }

    public static double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {
        if (firstMatrix[0].length != secondMatrix.length) {
            System.out.println("The matrices' sizes don't match");
            return null;
        }
        double[][] result = new double[firstMatrix.length][secondMatrix[0].length];
        double sum;
        for (int i = 0; i < firstMatrix.length; i++) {
            for (int j = 0; j < secondMatrix[0].length; j++) {
                sum = 0;
                for (int k = 0; k < firstMatrix[0].length; k++) {
                    sum += firstMatrix[i][k] * secondMatrix[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    public static double[][] vectorSubstraction(double[][] firstMatrix, double[][] secondMatrix) {
        if (firstMatrix.length != secondMatrix.length && firstMatrix[0].length != secondMatrix[0].length) {
            System.out.println("The matrices' sizes don't match");
            return null;
        }

        double[][] result = new double[firstMatrix.length][firstMatrix[0].length];
        for (int i = 0; i < firstMatrix.length; i++) {
            for (int j = 0; j < firstMatrix[0].length; j++) {
                result[i][j] = firstMatrix[i][j] - secondMatrix[i][j];
            }
        }
        return result;
    }

    public static double[][] vectorAddition(double[][] firstMatrix, double[][] secondMatrix) {
        if (firstMatrix.length != secondMatrix.length && firstMatrix[0].length != secondMatrix[0].length) {
            System.out.println("The matrices' sizes don't match");
            return null;
        }

        double[][] result = new double[firstMatrix.length][firstMatrix[0].length];
        for (int i = 0; i < firstMatrix.length; i++) {
            for (int j = 0; j < firstMatrix[0].length; j++) {
                result[i][j] = firstMatrix[i][j] + secondMatrix[i][j];
            }
        }
        return result;
    }

    public static double[][] hadamardProduct(double[][] firstMatrix, double[][] secondMatrix) {
        if (firstMatrix.length != secondMatrix.length && firstMatrix[0].length != secondMatrix[0].length) {
            System.out.println("The matrices' sizes don't match");
            return null;
        }

        double[][] result = new double[firstMatrix.length][firstMatrix[0].length];
        for (int i = 0; i < firstMatrix.length; i++) {
            for (int j = 0; j < firstMatrix[0].length; j++) {
                result[i][j] = firstMatrix[i][j] * secondMatrix[i][j];
            }
        }
        return result;
    }

    public static <T> List<List<T>> createMiniBatches(List<T> data, int miniBatchSize) {
        List<List<T>> miniBatches = new ArrayList<>();
        for (int start = 0; start < data.size(); start += miniBatchSize) {
            int end = Math.min(start + miniBatchSize, data.size());
            List<T> batch = new ArrayList<>(data.subList(start, end));
            miniBatches.add(batch);
        }
        return miniBatches;
    }

    public static double[][] addBias(double[][] matrix, double[] bias) {
        if (matrix.length != bias.length) {
            System.out.println("The matrices' sizes don't match");
            return null;
        }

        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = matrix[i][j] + bias[i];
            }
        }
        return result;
    }

    public static double[] softmax(double[] z) {
        double max = Arrays.stream(z).max().getAsDouble(); // Для предотвращения переполнения
        double sum = 0.0;
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = Math.exp(z[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < z.length; i++) {
            result[i] /= sum;
        }
        return result;
    }
    
    public static double[][] softmaxVector(double[][] z) {
        double[][] result = new double[z.length][1];
        double[] flatZ = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            flatZ[i] = z[i][0];
        }
        double[] flatResult = softmax(flatZ);
        for (int i = 0; i < result.length; i++) {
            result[i][0] = flatResult[i];
        }
        return result;
    }

    public double[][][] getWeights() {
        return weights;
    }

    public double[][] getBiases() {
        return biases;
    }

    public static void main(String[] args) throws IOException {
        NeuralNetwork network = new NeuralNetwork(new int[]{784, 60, 10});

        String trainingImagesFile = "/Users/nichitabulgaru/Documents/NN/NN/data/train-images.idx3-ubyte";
        String trainingLabelsFile = "/Users/nichitabulgaru/Documents/NN/NN/data/train-labels.idx1-ubyte";

        String testImagesFile = "/Users/nichitabulgaru/Documents/NN/NN/data/t10k-images.idx3-ubyte";
        String testingLabelsFile = "/Users/nichitabulgaru/Documents/NN/NN/data/t10k-labels.idx1-ubyte";

        List<ImageData> trainingDataset = MNISTReader.readMNISTData(trainingImagesFile, trainingLabelsFile);
        List<ImageData> testDataset = MNISTReader.readMNISTData(testImagesFile, testingLabelsFile);

        network.SGD(trainingDataset, 40, 100, 0.1, testDataset);
        for (int i = 0; i < 10; i++) {
            System.out.println("Predicted: " + network.predict(testDataset.get(i).imageData));
            System.out.println("Actual: " + argMax(testDataset.get(i).label));
        }
    }
}
