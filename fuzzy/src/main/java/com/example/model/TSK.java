package com.example.model;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.commons.lang3.DoubleRange;

import com.example.layers.FuzzyLayer;
import com.example.layers.MultipleLayer;
import com.example.layers.RoleMultipleLayer;
import com.example.layers.SumLayer;

import lombok.Data;

@Data
public class TSK implements Serializable {
    private FuzzyLayer fuzzyLayer;
    private RoleMultipleLayer roleMultipleLayer;
    private MultipleLayer multipleLayer;
    private SumLayer sumLayer;

    private int M; // число правил
    private int N; // число параметров
    private int out; // число выходов

    public TSK(int N, int M, int out) { // 4 4 1
        this.M = M;
        this.N = N;
        this.out = out;

        FuzzyLayer layer1 = new FuzzyLayer(N, M);
        RoleMultipleLayer layer2 = new RoleMultipleLayer(M*N, M, N);
        MultipleLayer layer3 = new MultipleLayer(M, M*out, N);
        SumLayer layer4 = new SumLayer(M*out, out);
        
        fuzzyLayer = layer1;
        roleMultipleLayer = layer2;
        multipleLayer = layer3;
        sumLayer = layer4;
    }

    public void updateP(double[][] P) {
        
        double[][] oldP = multipleLayer.getP();
        double[][] newP = new double[oldP.length][oldP[0].length];

        for (int i = 0; i < P.length; i++) {
            newP[i/(oldP[0].length)][i%(oldP[0].length)] = P[i][0];
        }
        multipleLayer.setP(newP);
        // for (double[] pi : newP) {
        //     System.out.println(". " + Arrays.toString(pi));
        // }
    }

    public double predict1(double[] x) {

        double[] y1 = fuzzyLayer.get(x);
        double[] y2 = roleMultipleLayer.get(y1);
        double[] y3 = multipleLayer.get(y2, x);
        double y4 = sumLayer.get(y3, y2);
        if(Double.isNaN(y4)) throw new RuntimeException("y4 is NaN");
        return y4;
    }
    
    public double evaluate1(double[] x, double d) {

        double[] y1 = fuzzyLayer.get(x);
        double[] y2 = roleMultipleLayer.get(y1);
        double[] y3 = multipleLayer.get(y2, x);
        double y4 = sumLayer.get(y3, y2);
        if(Double.isNaN(y4)) throw new RuntimeException("y4 is NaN");
        // System.out.println("predict: " + y4 + " source: " + d);
        return y4;
    }

    public double[] evaluate(double[][] dataset, double[] d, int classesCount) {
        return evaluate(dataset, d, dataset.length, 0, classesCount);
    }

    public double[] evaluate(double[][] dataset, double [] d, int n, int k, int classesCount) {
        /*
         * Предсказать по части выборки
         * n - размер батча
         * k - индекс начального вектора из выборки
         */
        double accuracy = 0;
        double[] y = new double[n];
        for (int i = k; i < n+k; i++) {
            double[] x = dataset[i];
            y[i] = evaluate1(x, d[i]);
        }
        
        double[] minMax = new double[]{Double.MAX_VALUE, Double.MIN_VALUE};

        Arrays.stream(y).forEach(value -> {
            if (value < minMax[0]) {
                minMax[0] = value;
            }
            if (value > minMax[1]) {
                minMax[1] = value;
            }
        });
        double rangeReal = (minMax[1] - minMax[0]);
        double step = (double)1 / classesCount;
        for (int i = k; i < k + n; i++) {
            // System.out.print(DoubleRange.of( (d[i] - step) * rangeReal, (d[i] + step) * rangeReal) + " " + y[i]);
            // if(DoubleRange.of( (d[i] - step) * rangeReal, (d[i] + step) * rangeReal).contains(y[i])) {
            //     accuracy += 1;
            //     System.out.print(" add");
            // }
            // System.out.println();
            if(DoubleRange.of( (d[i] - step) , (d[i] + step) ).contains(y[i])) {
                accuracy += 1;
            }else {
                //System.out.println(DoubleRange.of( (d[i] - step), (d[i] + step) ) + " " + y[i]);
            }
        }
        accuracy /= n;
        System.out.println("accuracy " + accuracy);
        return y;
    }

    public double[] predict(double[][] dataset, int n, int k) {
        /*
         * Предсказать по части выборки
         * n - размер батча
         * k - индекс начального вектора из выборки
         */
        double[] y = new double[n];
        for (int i = k; i < n+k; i++) {
            double[] x = dataset[i];
            y[i] = predict1(x);
        }
        return y;
    }

    public double[] predict(double[][] dataset) {
        /**
         * Предсказать по всей выборке
         */

        return predict(dataset, dataset.length, 0);
    }


    public double[] getW(double[] x) {
        double[] w = fuzzyLayer.get(x);
        return roleMultipleLayer.get(w);
    }

    public double[][] getP() {
        return multipleLayer.getP();
    }

    public double[] getC() {
        return fuzzyLayer.getC();
    }

    public double[] getSigma() {
        return fuzzyLayer.getSigma();
    }

    public double[] getB() {
        return fuzzyLayer.getB();
    }

    public void setP(double[][] p) {
        multipleLayer.setP(p);
    }

    public void setC(double[] c) {
        fuzzyLayer.setC(c);
    }

    public void setSigma(double[] sigma) {
        fuzzyLayer.setSigma(sigma);
    }

    public void setB(double[] b) {
        fuzzyLayer.setB(b);
    }
}
