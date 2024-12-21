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
        
        for (int i = 0; i < newP.length /* M*out */; i++) {
            for (int j = 0; j < newP[0].length /* N+1 */; j++) {
                int newI = (j % (P.length/(newP.length/P[0].length)) + (i / 3) * 5);
                int newJ = (i % 3);
                newP[i][j] = P[newI][newJ];
            }
        }
        multipleLayer.setP(newP);
    }

    public double[] predict1(double[] x) {

        double[] y1 = fuzzyLayer.get(x);
        double[] y2 = roleMultipleLayer.get(y1);
        double[] y3 = multipleLayer.get(y2, x);
        double[] y4 = sumLayer.get(y3);

        return y4;
    }
    
    public double[][] predict(List<Iris> dataset) {
        /**
         * Предсказать по всей выборке
         */
        return predict(dataset, dataset.size(), 0);
    }
    public double[][] predict(List<Iris> dataset, int n, int k) {
        /*
         * Предсказать по части выборки
         * n - размер батча
         * k - индекс начального вектора из выборки
         */

        double[][] y = new double[n][];
        for (int i = k; i < n+k; i++) {
            double[] x = dataset.get(i).getValues();

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
