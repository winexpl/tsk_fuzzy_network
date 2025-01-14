package com.example.learning;

import java.util.stream.IntStream;

import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.example.layers.FuzzyLayer;
import com.example.layers.resources.QFunction;
import com.example.model.TSK;

import lombok.AllArgsConstructor;
import lombok.Data;
@AllArgsConstructor
@Data
public class HybridLearningAlgorithm {
    private TSK tsk;
    private double[][] _x;
    private double[] _d;
    private int out = 1;
    public void learningTSKBatchFirstStep(int batchSize, int numBatch) {
        /*
         * batchSize - размер батча
         * numBatch - номер батча (НЕ ИНДЕКС ВЕКТОРА X)
         * Для получения индекса вектора x нужно уножить batchSize * numBatch
         *
         * w_k = MUL по j до N (nu^k(x_j)) / SUM по r до M(MUL по j до N (nu^k(x_j)))
         */

        int startIndex = numBatch * batchSize;
        int endIndex = (numBatch + 1) * batchSize;
        
        RealMatrix D = MatrixUtils.createRealMatrix(
            IntStream.range(startIndex, endIndex)
                    .mapToObj(i -> new double[]{_d[i]})
                    .toArray(double[][]::new)
        );

        double[][] A_matrix = new double[batchSize][];
        double[] x;
        double[] w;
        
        int N = tsk.getN();
        int M = tsk.getM();
        for (int i = 0; i < batchSize; i++) {
            x = _x[i + startIndex];
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

        double nuC = (double)0.5;
        double nuSigma = (double)0.5;
        double nuB = (double)0.5;
        for(int l = 0; l < 10; l++) {
        for (int i = 0; i < batchSize; i++) { // по всему батчу
            double x[] = _x[numBatch*batchSize + i];
            double d = _d[numBatch*batchSize + i];
            double y = tsk.predict1(x);
            
            // System.out.println(y);

            
            double e = y - d; // расчет ошибки
            // e = e < 0.1 ? 0.1 : e;
            for (int k = 0; k < M*N; k++) { // по всем параметрам
                // c
                double newC = oldC[k];
                double dEdc = dE(e, p, x, k, (xi, sigma, c, b) -> {
                    double dnu = 2 * b / sigma * Math.pow( (xi - c) / sigma, 2 * b - 1);
                    dnu /= Math.pow(1 + Math.pow( (xi- c) / sigma, 2 * b), 2);
                    if(Double.isNaN(dnu)) throw new RuntimeException("c is nan \nb="+b+" c="+c+" sigma="+sigma +" e="+e);

                    return dnu;
                });
                
                // sigma
                double newSigma = oldSigma[k];
                double dEdSigma = dE(e, p, x, k, (var xi, var sigma, var c, var b) -> {
                    double dnu = 2 * b / sigma * Math.pow( (xi - c) / sigma, 2 * b);
                    dnu /= Math.pow(1 + Math.pow( (xi- c) / sigma, 2 * b), 2);
                    if(Double.isNaN(dnu)) throw new RuntimeException("sigma is nan \nb="+b+" c="+c+" sigma="+sigma+" e="+e);
                    return dnu;
                });
                
                // System.out.println("newSigma " + newSigma);
                // b
                double newB = oldB[k];
                double dEdb = dE(e, p, x, k, (xi, sigma, c, b) -> {
                    double dnu = -2 * Math.pow( (xi - c) / sigma, 2 * b) * Math.log( Math.abs((xi - c) / sigma) );

                    dnu /= Math.pow(1 + Math.pow( (xi- c) / sigma, 2 * b), 2);
                    if(Double.isNaN(dnu)) throw new RuntimeException("b is nan \nb="+b+" c="+c+" sigma="+sigma+" e="+e);
                    return dnu;
                });
                double updateC = nuC*dEdc;
                double updateB = nuB*dEdb;
                double updateSigma = nuSigma*dEdSigma;
                if(updateC > 100) updateC = 100;
                else if(updateC < -100) updateC = -100;
                if(updateB > 1) updateB = 1;
                else if(updateB < -1) updateB = -1;
                if(updateSigma > 100) updateSigma = 100;
                else if(updateSigma < -100) updateSigma = -100;

                newC -= updateC;
                newSigma -= updateSigma;
                newB -= updateB;
                newB = Math.round(newB);
                // update
                // System.out.print(dEdc+" "+dEdSigma+" "+dEdb+" ");
                // System.out.println("oldC = " + oldC[k] + " newC = " + newC + " dedc = " + dEdc);
                // System.out.println("oldB = " + oldB[k] + " newB = " + newB + " dedb = " + dEdb);
                // System.out.println("oldS = " + oldSigma[k] + " newS = " + newSigma + " deds = " + dEdSigma);

                if(Double.isNaN(newC)) throw new RuntimeException("newC is nan");
                if(Double.isNaN(newSigma)) throw new RuntimeException("newSigma is nan");
                if(Double.isNaN(newB)) throw new RuntimeException("newB is nan");
                
                oldC[k] = newC;
                oldSigma[k] = newSigma;
                oldB[k] = newB;
                
                // System.out.println("newC " + newC + " newSigma " + newSigma + " newB " + newB);
            }
            tsk.setC(oldC);
            tsk.setSigma(oldSigma);
            tsk.setB(oldB);
        }
        }
        System.out.print(".");
    }

    private double dE(double e, double[][] p, double[] x, int ii, QFunction<Double> dnu) {
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
            double dwdc = dW(ii, i, x, dnu);
            temp *= dwdc;
            sum_p += temp;
        }
        return e * sum_p;
    }
    private double dW(int i, int r, double[] x, QFunction<Double> dnu) {
        /**
         * r - номер правила по которому ищем производную
         * i - номер изменяемого параметра в массиве,
         *          т.е. ii%M = номер правила для изменяемого параметра
         *          ii/M = номер входного параметра
         * x - текущий вектор параметров ириса
         */
        int M = tsk.getM();
        int N = tsk.getN();

        int k = r % M;
        int j = r / M;

        double res = deltaKronecker(i, k);
        double mm = m(x);
        res *= mm;
        res -= l(x, k);
        res /= Math.pow(mm, 2);
        double c[] = tsk.getC();
        double b[] = tsk.getB();
        double sigma[] = tsk.getSigma();

        FuzzyLayer fuzzyLayer = tsk.getFuzzyLayer();
        double multiple = 1;
        
        for (int l = k; l < M * N; l+=M) {
            if(j != i) multiple *= fuzzyLayer.getFuzzyFunction().apply(x[l / M] , sigma[l], c[l], b[l]);
        }
        res *= multiple;
        
        res *= dnu.apply(x[j], sigma[r], c[r], b[r]);
        return res;
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

    public static RealMatrix pseudoInverse(RealMatrix A) {
        SingularValueDecomposition svd = new SingularValueDecomposition(A);
        DecompositionSolver solver = svd.getSolver();
        RealMatrix pinv = solver.getInverse();
        return pinv;
    }
}