package com.example.models;

import java.util.List;

import com.example.layers.FuzzyLayer;
import com.example.layers.MultipleLayer;
import com.example.layers.RoleMultipleLayer;
import com.example.layers.SumLayer;

import lombok.Data;

@Data
public class TSK {
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
        // for(var i: newP) {
        //     //System.out.println("newP " + Arrays.toString(i));
        // }
        multipleLayer.setP(newP);
    }

    public double predict1(Iris x) {

        double[] y1 = fuzzyLayer.get(x.getValues());
        //System.out.println("y1: " + Arrays.toString(y1));
        double[] y2 = roleMultipleLayer.get(y1);
        //System.out.println("y2: " + Arrays.toString(y2));
        double[] y3 = multipleLayer.get(y2, x.getValues());
        //System.out.println("y3: " + Arrays.toString(y3));
        double y4 = sumLayer.get(y3, y2);
        System.out.println("predict: " + y4 + " source: " + x.getD());
        if(Double.isNaN(y4)) throw new RuntimeException("y4 is NaN");
        return y4;
    }
    
    public double[] predict(List<Iris> dataset) {
        /**
         * Предсказать по всей выборке
         */
        return predict(dataset, dataset.size(), 0);
    }

    public double[] predict(List<Iris> dataset, int n, int k) {
        /*
         * Предсказать по части выборки
         * n - размер батча
         * k - индекс начального вектора из выборки
         */

        double[] y = new double[n];
        for (int i = k; i < n+k; i++) {
            Iris x = dataset.get(i);
            y[i] = predict1(x);
        }
        return y;
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
