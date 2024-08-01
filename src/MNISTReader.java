import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MNISTReader {

    public static void main(String[] args) throws IOException {
        String imagesFile = "/Users/nichitabulgaru/Documents/NN/NN/data/train-images.idx3-ubyte";
        String labelsFile = "/Users/nichitabulgaru/Documents/NN/NN/data/train-labels.idx1-ubyte";
        List<ImageData> dataset = readMNISTData(imagesFile, labelsFile);
        Collections.shuffle(dataset);
        System.out.println(Arrays.deepToString(dataset.get(0).imageData));
        System.out.println(Arrays.deepToString(dataset.get(0).label));
        // Use the dataset as needed
    }

    public static List<ImageData> readMNISTData(String imagesFile, String labelsFile) throws IOException {
        try (DataInputStream images = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesFile)));
             DataInputStream labels = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsFile)))) {
            
            @SuppressWarnings("unused")
            int magicNumberImages = images.readInt();
            int numberOfImages = images.readInt();
            int rows = images.readInt();
            int cols = images.readInt();

            @SuppressWarnings("unused")
            int magicNumberLabels = labels.readInt();
            @SuppressWarnings("unused")
            int numberOfLabels = labels.readInt();

            List<ImageData> dataset = new ArrayList<>();
            for (int i = 0; i < numberOfImages; i++) {
                // Read and normalize image data
                double[][] imageData = new double[rows][cols];
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        imageData[r][c] = (images.readUnsignedByte() & 0xff) / 255.0;
                    }
                }

                // Read label
                int label = labels.readUnsignedByte();
                double[][] arrayLabel = new double[10][1];
                for (int j = 0; j < 10; j++) {
                    arrayLabel[j][0] = (j == label) ? 1.0 : 0.0;
                }

                double[][] preparedImageData = new double[784][1];
                int index = 0;
                for (int m = 0; m < 28; m++) {
                    for (int j = 0; j < 28; j++) {
                        preparedImageData[index][0] = imageData[m][j];
                        index++;
                    }
                }
                // Store the image data and label in the dataset
                dataset.add(new ImageData(preparedImageData, arrayLabel));
            }

            return dataset;
        }
    }
}

class ImageData {
    public double[][] imageData;
    public double[][] label;

    public ImageData(double[][] imageData, double[][] label) {
        this.imageData = imageData;
        this.label = label;
    }
}
