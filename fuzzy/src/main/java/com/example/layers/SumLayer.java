package com.example.layers;

import java.util.Arrays;

import lombok.Data;

@Data
public class SumLayer implements Layer {
    private int dimInput; // число выходов * число правил
    private int dimOutput; // число выходов
    private int M; // число правил
    
    public SumLayer(int dimInput, int dimOutput) {
        if(((double)dimInput)/dimOutput != dimInput/dimOutput)
            throw new RuntimeException("The input dimension must be completely divided by the output dimension");
        this.dimInput = dimInput;
        this.dimOutput = dimOutput;
        this.M = dimInput/dimOutput;
    }

    public double get(double[] x, double v[]) {
        double y = Arrays.stream(x).sum();
        double sum = Arrays.stream(v).sum();
        
        return y/sum;
    }
}
