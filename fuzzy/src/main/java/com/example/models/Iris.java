package com.example.models;

import com.example.models.resources.SpeciesOfIris;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Iris {
    private double sepalLength;
    private double sepalWidth;
    private double petalLength;
    private double petalWidth;
    private SpeciesOfIris species;

    public void setSpecies(String speciesString) {
        species = SpeciesOfIris.valueOf(speciesString.toUpperCase().replace("IRIS-", ""));
    }

    public Iris(double sepalLength, double sepalWidth, double petalLength, double petalWidth, String species) {
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
        setSpecies(species);
    }

    public double[] getValues() {
        return new double[] { sepalLength, sepalWidth, petalLength, petalWidth };
    }

    public double[] getD() {
        double[] d = new double[3];
        d[species.getI()] = 1;
        return d;
    }
}
