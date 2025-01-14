package com.example.model.resources;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;
import java.util.regex.Pattern;

public class DatasetDiabet extends Dataset {
    public enum Field {
        GLUCOSE, BLOODPRESSURE;
    }
    
    public DatasetDiabet() throws FileNotFoundException {
        datasetLength = 995;
        classesLength = 2;
        xLength = 2;
        String filename = "./fuzzy/resources/diabet.csv";
        FileInputStream fileInputStream = new FileInputStream(filename);
        values = new double[datasetLength][xLength];
        d = new double[datasetLength];
        try (Scanner scanner = new Scanner(fileInputStream)) {
            scanner.nextLine();
            scanner.useDelimiter(Pattern.compile("[,\\n]"));
            scanner.useLocale(Locale.ENGLISH);
            int i = 0;
            while(scanner.hasNext()) {
                values[i] = new double[]{scanner.nextDouble(), scanner.nextDouble()};
                d[i++] = scanner.nextInt();
            }
        }
    }
}
