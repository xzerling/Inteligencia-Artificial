package iris;

/**
 *
 * @author Zerling
 */
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
 
public class Iris
{
    public static BufferedReader readDataFile(String filename)
    {
        BufferedReader inputReader = null;

        try
        {
            inputReader = new BufferedReader(new FileReader(filename));
        }
        catch (FileNotFoundException ex)
        {
            System.err.println("File not found: " + filename);
        }
        return inputReader;
    }
 
    public static Evaluation classify(Classifier model,Instances trainingSet, Instances testingSet) throws Exception
    {
        Evaluation evaluation = new Evaluation(trainingSet);
        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);
        return evaluation;
    }

    public static double calculateAccuracy(FastVector predictions)
    {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++)
        {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual())
            {
                correct++;
            }
        }

        return 100 * correct / predictions.size();
    }

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds)
    {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++)
        {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }
        return split;
    }

    public static void main(String[] args) throws Exception
    {

        if(args.length == 0)
        {
            System.out.println("error en cargar el archivo");
            System.out.println("La ejecucion es java -jar iris.jar data.txt");

            System.exit(0);
        }
        BufferedReader datafile = readDataFile(args[0]);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);

        // Do 10-split cross validation
        Instances[][] split = crossValidationSplit(data, 10);

        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];

        // Usando 
        Classifier[] models = {new J48()};

        // Run for each model
        for (int j = 0; j < models.length; j++)
        {
            FastVector predictions = new FastVector();

            // For each training-testing split pair, train and test the classifier
            for (int i = 0; i < trainingSplits.length; i++) 
            {
                Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
                predictions.appendElements(validation.predictions());
                System.out.println(models[j].toString());
            }
            // Calculate overall accuracy of current classifier on all splits
            double accuracy = calculateAccuracy(predictions);
            System.out.println("Certeza " + models[j].getClass().getSimpleName() + ": "
                            + String.format("%.2f%%", accuracy)
                            + "\n---------------------------------");
        }
    }
}