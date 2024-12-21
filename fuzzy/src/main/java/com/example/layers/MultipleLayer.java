package com.example.layers;

import java.util.Random;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class MultipleLayer implements Layer {
    // Третий слой
    private int dimInput; // число правил
    private int dimOutput; // число правил * число выходов
    private int outSize; // число выходов
    private int N; // число параметров входного вектора x
    private double[][] p; // wij, i - 11, 12, .., 21, 22, .., 31, 32... и т.д.
                          // j - вес для xj, где xj - jый параметра вектора x
    

    public MultipleLayer(int dimInput, int dimOutput, int N) {
        if(((double)dimOutput)/dimInput != dimOutput/dimInput)
            throw new RuntimeException("The input dimension must be completely divided by the output dimension");
        this.dimInput = dimInput;
        this.dimOutput = dimOutput;
        this.outSize = dimOutput / dimInput;
        this.N = N;
        p = new double[dimOutput][];
        Random rand = new Random();
        for (int i = 0; i < dimOutput; i++) {
            p[i] = new double[N+1];
            for (int j = 0; j < N+1; j++) {
                p[i][j] = rand.nextDouble(0,1);
            }
        }
    }

    public double[] get(double[] v, double[] x) {
        if(x.length != N) throw new RuntimeException("The input data does not match the size of the input layer");

        double[] y = new double[dimOutput];
        
        for (int i = 0; i < dimInput /* w.length */ ; i++) {
            for (int j = 0; j < outSize; j++) {
                y[i*outSize+j] += p[i*outSize+j][0];
                for (int k = 1; k < x.length + 1 /* count */; k++) {
                    y[i*outSize+j] += p[i*outSize+j][k] * x[k-1]; // w[i] = pi0 pi1 pi2 pi3 pi4, i = 11,12,21,22...
                }
                y[i*outSize+j] *= v[i];
            }
        }
        return y;
    }
    
}
