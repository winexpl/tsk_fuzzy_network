package com.example.layers;

import java.util.Random;

import com.example.layers.resources.FuzzyFunction;
import com.example.layers.resources.QFunction;

import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;


@Data
@NoArgsConstructor
public final class FuzzyLayer implements Layer {
    // Первый слой
    private int dimInput;
    private int dimOutput;
    private int M;
    private double[] sigma;
    private double[] c;
    private double[] b;
    @Getter
    private QFunction<Double> fuzzyFunction;

    public void setFuzzyFunction(FuzzyFunction f) {
        fuzzyFunction = switch (f) {
            case GENERAL_GAUSSIAN -> FuzzyLayer::generalGaussianMembersFunction;
            default -> FuzzyLayer::generalGaussianMembersFunction;
        };
        
    }

    public static double generalGaussianMembersFunction(double x, double sigma, double c, double b) {
        double res = 1/(1+Math.pow( ( x-c )/ sigma , 2*b));
        return res;
    }

    public FuzzyLayer(int dimInput, int M) {
        setFuzzyFunction(FuzzyFunction.GENERAL_GAUSSIAN);
        this.dimInput = dimInput;
        this.dimOutput = dimInput * M;
        this.M = M;
        sigma = new double[dimOutput];
        c = new double[dimOutput];
        b = new double[dimOutput];
        Random rand = new Random();
        for (int i = 0; i < dimOutput; i++) {
            sigma[i] = rand.nextDouble(0.1,1);
            c[i] = rand.nextDouble(0.1,1);
            b[i] = rand.nextInt(1,5);
        }
    }

    
    public double[] get(double[] x) {
        if(x.length != dimInput) throw new RuntimeException("The input data does not match the size of the input layer");
        double[] y = new double[dimOutput];
        for (int i = 0; i < dimInput; i++) {
            for (int j = 0; j < M; j++) {
                y[i*M+j] = getFuzzyFunction().apply(x[i], sigma[i*M+j], c[i*M+j], b[i*M+j]);
            }
        }
        return y;
    }
}