package com.example.model.resources;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;
import java.util.regex.Pattern;

public class DatasetIrises extends Dataset {
    public enum OutValue {
        SETOSA, VERSICOLOR, VIRGINICA;
    }

    public enum Field {
        SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH;
    }
    
    public DatasetIrises() throws FileNotFoundException {
        datasetLength = 150;
        classesLength = 3;
        xLength = 4;
        String filename = "./fuzzy/resources/IRIS.csv";
        FileInputStream fileInputStream = new FileInputStream(filename);
        values = new double[datasetLength][xLength];
        d = new double[datasetLength];
        try (Scanner scanner = new Scanner(fileInputStream)) {
            scanner.nextLine();
            scanner.useDelimiter(Pattern.compile("[,\\n]"));
            scanner.useLocale(Locale.ENGLISH);
            int i = 0;
            while(scanner.hasNext()) {
                values[i] = new double[]{scanner.nextDouble(), scanner.nextDouble(),
                    scanner.nextDouble(), scanner.nextDouble()};
                d[i++] = (double)(OutValue.valueOf(scanner.next().toUpperCase().replace("IRIS-", "")).ordinal()) / (classesLength-1);
            }
        }
    }
}
