package com.example.layers;

import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class RoleMultipleLayer implements Layer {
    // Второй слой
    // число правил = число множеств
    private int dimInput; // число правил * размерность вектора x
    private int dimOutput; // число правил
    private int N; // размерность вектора x


    public RoleMultipleLayer(int dimInput, int dimOutput, int N) {
        this.dimInput = dimInput;
        this.N = N;
        this.dimOutput = dimOutput;
    }


    public double[] get(double[] x) {
        /*
         * x - выход слоя FuzzyLayer
         */
        if(x.length != dimInput) throw new RuntimeException("The input data does not match the size of the input layer");
        double[] y = new double[dimOutput];
        for(int i = 0; i < dimOutput; i++) {
            y[i] = 1;
            for (int j = 0; j < N ; j++) {
                y[i] *= x[i + j*dimOutput];
            }
        }

        return y;
    }
    
}
