package com.example.learning;

import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import com.example.models.Iris;
import com.example.models.TSK;

import lombok.AllArgsConstructor;
@AllArgsConstructor
public class HybridLearningAlgorithm {
    private TSK tsk;
    private List<Iris> irises;
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
            for (int j = 0; j < M; j++) {
                A_matrix[i][j * (N+1)] = w[j];
                for (int k = 1; k < N + 1; k++) {
                    A_matrix[i][j * (N+1) + k] = w[j] * x[k - 1];
                }
            }
        }
        RealMatrix A = MatrixUtils.createRealMatrix(A_matrix);
        RealMatrix A_inv = pseudoInverse(A);
        RealMatrix P = A_inv.multiply(D);
        tsk.updateP(P.getData());
    }

    public void learningTSKBatchSecondStep(List<Iris> irises, int batchSize, int numBatch) {
        int N = tsk.getN();
        int M = tsk.getM();
        int out = tsk.getOut();
        for (int i = 0; i < batchSize; i++) { // по всему батчу
            Iris iris = irises.get(numBatch*batchSize + i);
            double[] y = tsk.predict1(iris.getValues());
            double[] d = iris.getD();
            double nu = 0.01;
            for (int j = 0; j < out; j++) { // по всем выходам сети
                double e = y[i] - d[i]; // расчет ошибки

                double[] oldC = tsk.getC();
                double[] oldSigma = tsk.getSigma();
                double[] oldB = tsk.getB();
                for (int k = 0; k < M*N; k++) { // по всем параметрам
                    double[][] p = tsk.getP();
                    // C
                    double newC = oldC[k];
                    newC -= nu*dEdc();

                    double newSigma = oldSigma[k];
                    newSigma -= nu*dEdSigma();

                    double newB = oldB[k];
                    newB -= nu*dEdB();
                }
            }
        }
    }

    private double dEdc(double e, int k, double[][] p, int y_size, double[] x) {
        /**
         * j - номер входного параметра сети
         * M - число правил
         * y_size - число выходов сети
         * x - текущий вектор параметров ириса
         */
        double sum_p = 0;
        for (int i = y_size-1; i < p.length; i+=y_size) {
            sum_p += p[i][0];
            for (int r = 1; r < p[i].length; r++) {
                sum_p += p[i][r] * x[r-1];
            }
            sum_p *= c_dWdC();
            
        }
        return e*sum_p;
    }

    private static double c_dWdC(TSK tsk, int r, int i, int k, double[] x, int M, int N) {
        /**
         * N - число нечетких множеств
         */
        double m = 0;
        double dCron = deltaKronecker(r, k);
        for (int kk = 0; kk < M; kk++) {
            double mul = 1;
            for (int jj = 0; jj < N; jj++) {
                
                mul *= 
            }
        }
    }

    private static int deltaKronecker(int i, int j) {
        return i==j? 1:0;
    }

    public static double[][] getD(List<Iris> dataset, int i, int batchSize) {
        double[][] d = new double[batchSize][];
        for(int j = 0; j < batchSize; j++) {
            d[j] = new double[3];
            d[j][dataset.get(i+j).getSpecies().getI()] = 1;
        }
        return d;
    }
    public static RealMatrix pseudoInverse(RealMatrix A) {
        SingularValueDecomposition SVD = new SingularValueDecomposition(A);

        RealMatrix V = SVD.getV();
        RealMatrix U = SVD.getU();

        double[] singularValues = SVD.getSingularValues();

        RealMatrix sigmaInverse = createSigmaInverse(singularValues);
        // A^-1 = V * Sigma^-1 * U^T
        RealMatrix A_inv = V.multiply(sigmaInverse).multiply(U.transpose());
        
        return A_inv;
    }
    // Метод для создания обратной диагональной матрицы Sigma^-1
    private static RealMatrix createSigmaInverse(double[] singularValues) {
        int n = singularValues.length;
        double[][] sigmaInverseData = new double[n][n];

        // Преобразуем сингулярные значения в диагональную матрицу обратных значений
        for (int i = 0; i < n; i++) {
            if (singularValues[i] != 0) {
                sigmaInverseData[i][i] = 1.0 / singularValues[i];
            }
        }

        return MatrixUtils.createRealMatrix(sigmaInverseData);
    }
}