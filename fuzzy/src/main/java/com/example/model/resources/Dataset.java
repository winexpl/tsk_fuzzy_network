package com.example.model.resources;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import lombok.Data;

@Data
public abstract class Dataset {
    protected int datasetLength;
    protected int xLength;
    protected int classesLength;
    
    protected double[][] values;
    protected double[] d;

    public enum OutValue {}
    public enum Field {}

    public double getMeans(OutValue outValue, Field field) {
            return Arrays.stream(values)
                .mapToDouble(row -> row[field.ordinal()])
                .average().getAsDouble();
        }
    public double getAverageMedian(OutValue outValue, Field field) {
            double[] valuesOfField = Arrays.stream(values)
                .sorted(Comparator.comparingDouble(a -> a[field.ordinal()]))
                .mapToDouble(a -> a[field.ordinal()])
                .toArray();
            return valuesOfField[valuesOfField.length / 2];
        }

    public void shuffle() {
        List<Integer> indices = IntStream.range(0, values.length).boxed().collect(Collectors.toList());
        Collections.shuffle(indices);
        double[][] shuffledValues = new double[values.length][];
        double[] shuffledD = new double[d.length];
        for (int i = 0; i < indices.size(); i++) {
            int index = indices.get(i);
            shuffledValues[i] = values[index];
            shuffledD[i] = d[index];
        }
        System.arraycopy(shuffledValues, 0, values, 0, values.length);
        System.arraycopy(shuffledD, 0, d, 0, d.length);
    }

    
}
