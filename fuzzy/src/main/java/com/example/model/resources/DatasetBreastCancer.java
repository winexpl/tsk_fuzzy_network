package com.example.model.resources;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

public class DatasetBreastCancer extends Dataset {
    public enum OutValue {
        M, B;
    }

    public enum Field {
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
        concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
        perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
        concave_points_worst, symmetry_worst, fractal_dimension_worst;
    }
    
    public DatasetBreastCancer() throws FileNotFoundException {
        datasetLength = 569;
        classesLength = 2;
        xLength = 30;
        String filename = "./fuzzy/resources/breast-cancer.csv";
        FileInputStream fileInputStream = new FileInputStream(filename);
        values = new double[datasetLength][xLength];
        d = new double[datasetLength];
        try (Scanner scanner = new Scanner(fileInputStream)) {
            scanner.nextLine();
            scanner.useDelimiter(Pattern.compile("[,\\n]"));
            scanner.useLocale(Locale.ENGLISH);
            int i = 0;
            while(scanner.hasNext() && i < datasetLength) {
                scanner.nextInt();
                d[i] = (double)(OutValue.valueOf(scanner.next()).ordinal()) / (classesLength-1);
                values[i++] = IntStream.range(0, 30)
                                .mapToDouble(j -> scanner.nextDouble())
                                .toArray();
            }
        }
    }
}
