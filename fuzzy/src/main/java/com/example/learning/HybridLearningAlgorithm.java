package com.example.learning;

import java.util.List;

import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.example.layers.FuzzyLayer;
import com.example.models.Iris;
import com.example.models.TSK;

import lombok.AllArgsConstructor;
@AllArgsConstructor
public class HybridLearningAlgorithm {
    private TSK tsk;
    private List<Iris> irises;
    private int out = 1;
    public void learningTSKBatchFirstStep(int batchSize, int numBatch) {
        /*
         * batchSize - размер батча
         * numBatch - номер батча (НЕ ИНДЕКС ВЕКТОРА X)
         * Для получения индекса вектора x нужно уножить batchSize * numBatch
         *
         * w_k = MUL по j до N (nu^k(x_j)) / SUM по r до M(MUL по j до N (nu^k(x_j)))
         */

        RealMatrix D = MatrixUtils.createRealMatrix(getD(irises, numBatch*batchSize, batchSize));
        double[][] A_matrix = new double[batchSize][];
        double[] x;
        double[] w;
        
        int N = tsk.getN();
        int M = tsk.getM();
        for (int i = 0; i < batchSize; i++) {
            x = irises.get(i).getValues();
            w = tsk.getW(x);
            A_matrix[i] = new double[ (N + 1) * M ];
            for (int j = 0; j < M * (N + 1); j++) {
                A_matrix[i][j] = w[j / (N + 1)] * ((j % (N + 1) == 0) ?
                    1 : x[j % (N + 1) - 1]);
            }
        }

        RealMatrix A = MatrixUtils.createRealMatrix(A_matrix);
        RealMatrix A_inv = pseudoInverse(A);
        RealMatrix P = A_inv.multiply(D);
        tsk.updateP(P.getData());
    }

    public void learningTSKBatchSecondStep(int batchSize, int numBatch) {
        int N = tsk.getN();
        int M = tsk.getM();
        double[] oldC = tsk.getC();
        double[] oldSigma = tsk.getSigma();
        double[] oldB = tsk.getB();
        double[][] p = tsk.getP();
        for (int i = 0; i < batchSize; i++) { // по всему батчу
            Iris iris = irises.get(numBatch*batchSize + i);
            double y = tsk.predict1(iris);
            double d = iris.getD();

            double nu = 0.005;
            double e = y - d; // расчет ошибки
            // System.out.println("e = " + e + " y = " + y + " d = " + d);
            for (int k = 0; k < M*N; k++) { // по всем параметрам
                // c
                double newC = oldC[k];
                newC -= nu*dEdc(e, p, iris.getValues(), k);
                // System.out.println("newC " + newC);
                // sigma
                double newSigma = oldSigma[k];
                newSigma -= nu*dEdsigma(e, p, iris.getValues(), k);
                // System.out.println("newSigma " + newSigma);
                // b
                double newB = oldB[k];
                newB -= nu*dEdb(e, p, iris.getValues(), k);
                System.out.println("newC " + newC + " newSigma " + newSigma + " newB " + newB);
                // update
                oldC[k] = newC;
                oldSigma[k] = newSigma;
                oldB[k] = newB;
            }
            tsk.setC(oldC);
            tsk.setSigma(oldSigma);
            tsk.setB(oldB);
        }
        
    }

    private double dEdsigma(double e, double[][] p, double[] x, int ii) {
        /**
         * ii - номер изменяемого параметра в массиве, 
         *  т.е. ii%M = номер правила
         *          ii/M = номер входного параметра
         * e - ошибка
         * x - текущий вектор параметров ириса
         */
        double sum_p = 0;
        for (int i = 0; i < p.length; i++) {
            double temp = p[i][0];
            for (int r = 1; r < p[i].length; r++) {
                temp += p[i][r] * x[r-1];
            }
            temp *= dWdsigma(ii, i, x);
            sum_p += temp;
        }
        if(Double.isNaN(e*sum_p)) return 0;
        return e*sum_p;
    }

    private int dEdb(double e, double[][] p, double[] x, int ii) {
        /**
         * ii - номер изменяемого параметра в массиве, 
         *  т.е. ii%M = номер правила
         *          ii/M = номер входного параметра
         * e - ошибка
         * x - текущий вектор параметров ириса
         */
        double sum_p = 0;
        for (int i = 0; i < p.length; i+=1) {
            double temp = p[i][0];
            for (int r = 1; r < p[i].length; r++) {
                temp += p[i][r] * x[r-1];
            }
            temp *= dWdb(ii, i, x);
            sum_p += temp;
        }
        return (int) Math.ceil(e * sum_p);
    }

    private double dEdc(double e, double[][] p, double[] x, int ii) {
        /**
         * ii - номер изменяемого параметра в массиве,
         *  т.е. ii%M = номер правила
         *          ii/M = номер входного параметра
         * e - ошибка
         * x - текущий вектор параметров ириса
         */
        double sum_p = 0;
        for (int i = 0; i < p.length; i+=1) {
            double temp = p[i][0];
            for (int r = 1; r < p[i].length; r++) {
                temp += p[i][r] * x[r-1];
            }
            temp *= dWdC(ii, i, x);
            sum_p += temp;
        }
        return e*sum_p;
    }

    private double dWdb(int ii, int r, double[] x) {
        /**
         * ii - номер изменяемого параметра в массиве,
         *  т.е. ii%M = номер правила
         *          ii/M = номер входного параметра
         * x - текущий вектор параметров ириса
         */
        int k = ii % tsk.getM();
        int j = ii / tsk.getM();

        int res = deltaKronecker(r, j);
        res *= m(x);
        res -= l(x, k);
        res /= Math.pow(m(x), 2);
        double c[] = tsk.getC();
        double b[] = tsk.getB();
        double sigma[] = tsk.getSigma();

        FuzzyLayer fuzzyLayer = tsk.getFuzzyLayer();
        double multiple = 1;
        int M = tsk.getM();
        int N = tsk.getN();
        for (int i = k; i < M * N; i+=M) {
            if(j != i) multiple *= fuzzyLayer.getFuzzyFunction().apply(x[i / M] , sigma[i], c[i], b[i]);
        }
        
        res *= multiple;

        double dnu = (int) (-2 * Math.pow( (x[j] - c[ii]) / sigma[ii], 2 * b[ii] ) * Math.log((x[j] - c[ii]) / sigma[ii]));
        dnu /= Math.pow(1 + Math.pow((x[j] - c[ii]) / sigma[ii], 2 * b[ii]), 2);

        return res * dnu;
    }

    private double dWdsigma(int ii, int r, double[] x) {
        /**
         * ii - номер изменяемого параметра в массиве, 
         *  т.е. ii%M = номер правила
         *          ii/M = номер входного параметра
         * x - текущий вектор параметров ириса
         */
        int k = ii % tsk.getM();
        int j = ii / tsk.getM();

        double res = deltaKronecker(r, j);
        res *= m(x);
        res -= l(x, k);
        res /= Math.pow(m(x), 2);
        double c[] = tsk.getC();
        double b[] = tsk.getB();
        double sigma[] = tsk.getSigma();

        FuzzyLayer fuzzyLayer = tsk.getFuzzyLayer();
        double multiple = 1;
        int M = tsk.getM();
        int N = tsk.getN();
        for (int i = k; i < M * N; i+=M) {
            if(j != i) multiple *= fuzzyLayer.getFuzzyFunction().apply(x[i / M] , sigma[i], c[i], b[i]);
        }
        
        res *= multiple;

        double dnu = 2 * b[ii] / sigma[ii] * Math.pow( (x[j] - c[ii]) / sigma[ii], 2 * b[ii] );
        dnu /= Math.pow(1 + Math.pow((x[j] - c[ii]) / sigma[ii], 2 * b[ii]), 2);

        return res * dnu;
    }

    private double dWdC(int ii, int r, double[] x) {
        /**
         * ii - номер изменяемого параметра в массиве, 
         *  т.е. ii%M = номер правила
         *          ii/M = номер входного параметра
         * x - текущий вектор параметров ириса
         */
        int k = ii % tsk.getM();
        int j = ii / tsk.getM();

        double res = deltaKronecker(r, j);
        res *= m(x);
        res -= l(x, k);
        res /= Math.pow(m(x), 2);
        double c[] = tsk.getC();
        double b[] = tsk.getB();
        double sigma[] = tsk.getSigma();

        FuzzyLayer fuzzyLayer = tsk.getFuzzyLayer();
        double multiple = 1;
        int M = tsk.getM();
        int N = tsk.getN();
        for (int i = k; i < M * N; i+=M) {
            if(j != i/M) multiple *= fuzzyLayer.getFuzzyFunction().apply(x[i / M] , sigma[i], c[i], b[i]);
        }
        
        res *= multiple;

        double dnu = 2 * b[ii] / sigma[ii] * Math.pow( (x[j] - c[ii]) / sigma[ii], 2 * b[ii] - 1);
        dnu /= Math.pow(1 + Math.pow((x[j] - c[ii]) / sigma[ii], 2 * b[ii]), 2);

        return res * dnu;
    }

    private double m(double[] x) {
        double w[] = tsk.getW(x);
        double res = 0;
        for (int i = 0; i < w.length; i++) {
            res += w[i];
        }
        return res;
    }

    private double l(double[] x, int i) {
        double w[] = tsk.getW(x);
        return w[i];
    }

    private static int deltaKronecker(int i, int j) {
        return i==j? 1:0;
    }

    public static double[][] getD(List<Iris> dataset, int i, int batchSize) {
        double[][] d = new double[batchSize][];
        for(int j = 0; j < batchSize; j++) {
            d[j] = new double[1];
            d[j][0] = dataset.get(i+j).getD();
        }
        return d;
    }
    public static RealMatrix pseudoInverse(RealMatrix A) {
        SingularValueDecomposition svd = new SingularValueDecomposition(A);
        DecompositionSolver solver = svd.getSolver();
        RealMatrix pinv = solver.getInverse();
        return pinv;
    }
}