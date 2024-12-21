package com.example;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.regex.Pattern;

import com.example.learning.HybridLearningAlgorithm;
import com.example.models.Iris;
import com.example.models.TSK;
import com.example.models.resources.IrisFields;
import com.example.models.resources.SpeciesOfIris;


public class Main {
    public static List<Iris> dataset;
    public static String filename = "./fuzzy/resources/IRIS.csv";
    public static Map<SpeciesOfIris, EnumMap<IrisFields, Double>> averagesMedian;
    public static Map<SpeciesOfIris, EnumMap<IrisFields, Double>> means;
    public static void main(String[] args) throws FileNotFoundException {
        loadData();
        shuffle(dataset);
        TSK modelTSK = new TSK(4, 4, 3);
        HybridLearningAlgorithm.learningTSKBatchFirstStep(modelTSK, dataset, 16, 0);

        double[][] predict = modelTSK.predict(dataset, 16);

        SpeciesOfIris[] speciesOfIrises = SpeciesOfIris.values();
        System.out.println("--- PREDICT ---");
        for (int i = 0; i < predict.length; i++) {
            System.out.println(speciesOfIrises[imax(predict[i])] + " " + dataset.get(i).getSpecies());
        }
        
        // Map<SpeciesOfIris, List<Double>> answer = new EnumMap<>(SpeciesOfIris.class);
        // SpeciesOfIris speciesOfIris[] = SpeciesOfIris.values();

        // StringBuilder builder = new StringBuilder();
        // for (int i = 0; i < dataset.size(); i++) {
        //     double w[] = layer2.get(layer1.get(dataset.get(i).getValues()));
        //     int index = imax(w);
        //     builder.append(speciesOfIris[index].toString()).append("\t").append(Arrays.toString(w)).append("\n");
        // }
        // System.out.println(builder.toString());
    }

    public static int imax(double[] array) {

        if(array.length <= 0) throw new RuntimeException("Array is empty");
        int imax = 0;

        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[imax]) {
                imax = i;  // Обновляем максимальный элемент
            }
        }
        return imax;
    }

    public static <T> void shuffle(List<T> list) {
        Random random = new Random();
        for (int i = list.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            T temp = list.get(i);
            list.set(i, list.get(j));
            list.set(j, temp);
        }
    }
    public static double getMean(double[] args) {
        double sum = 0;
        for (double v : args) {
            sum += v;
        }
        return sum / args.length;
    }

    public static double getAverageMedian(double[] args) {
        return args[args.length/2];
    }

    public static void loadData() throws FileNotFoundException {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));
        FileInputStream fileInputStream = new FileInputStream(filename);
        List<Iris> irises = new ArrayList<>();
        try (Scanner scanner = new Scanner(fileInputStream)) {
            scanner.nextLine();
            scanner.useDelimiter(Pattern.compile("[,\\n]"));
            scanner.useLocale(Locale.ENGLISH);
            while(scanner.hasNext()) {
                irises.add(new Iris(scanner.nextDouble(), scanner.nextDouble(),
                    scanner.nextDouble(), scanner.nextDouble(), scanner.next()));
            }
            double[][][] values = new double[3][4][50];
            int counter = 0;
            for(int k = 0; k < 3; k++) {
                for (int i = 0; i < 50; i++, counter++) {
                    double[] irisValues = irises.get(counter).getValues();
                    for (int j = 0; j < 4; j++) {
                        values[k][j][i] = irisValues[j];
                    }
                    
                }
            }

            averagesMedian = new EnumMap<>(SpeciesOfIris.class);
            means = new EnumMap<>(SpeciesOfIris.class);
            SpeciesOfIris speciesOfIris[] = SpeciesOfIris.values();
            IrisFields irisFields[] = IrisFields.values();
            for (int i = 0; i < 3; i++) {
                means.put(speciesOfIris[i], new EnumMap<>(IrisFields.class));
                averagesMedian.put(speciesOfIris[i], new EnumMap<>(IrisFields.class));
            }
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    means.get(speciesOfIris[i]).put(irisFields[j], getMean(values[i][j]));
                    averagesMedian.get(speciesOfIris[i]).put(irisFields[j], getAverageMedian(values[i][j]));
                }
            }
            dataset = irises;
        }
    }
}