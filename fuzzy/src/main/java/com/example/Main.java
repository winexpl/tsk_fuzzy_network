package com.example;
import java.io.DataInputStream;
import java.io.IOException;

import com.example.learning.HybridLearningAlgorithm;
import com.example.model.TSK;
import com.example.model.resources.Dataset;
import com.example.model.resources.DatasetIrises;


public class Main {
    public static void main(String[] args) throws IOException {
        Dataset dataset = new DatasetIrises();
        dataset.shuffle();
        
        TSK modelTSK = new TSK(dataset.getXLength(), 10, 1);
        double[] predict = modelTSK.evaluate(dataset.getValues(), dataset.getD());
        System.in.read();
        HybridLearningAlgorithm hla = new HybridLearningAlgorithm(modelTSK, dataset, 1);
        DataInputStream dis = new DataInputStream(System.in);

        int x = dis.readChar();
        int i = 0;

        
        while(true) {
            i++;
            for (int j = 0; j < 15; j++) {
                hla.learningTSKBatchFirstStep(10, j);
                hla.learningTSKBatchSecondStep(10, j);
            }
            if(i % 2 == 0) {
            System.out.println("After "+i+" epoch");
            predict = modelTSK.evaluate(dataset.getValues(), dataset.getD());
            x = dis.readChar();
            if(x == -1) break;
            System.out.println(x);
            }
            
        }
        predict = modelTSK.evaluate(dataset.getValues(), dataset.getD());
    }

    public static int imax(double[] array) {

        if(array.length <= 0) throw new RuntimeException("Array is empty");
        int imax = 0;

        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[imax]) {
                imax = i;  // Обновляем максимальный элемент
            }
        }
        return imax;
    }
}