package com.example;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import com.example.learning.HybridLearningAlgorithm;
import com.example.model.TSK;
import com.example.model.resources.Dataset;
import com.example.model.resources.DatasetBreastCancer;


public class Main {
    public static void main(String[] args) throws IOException {
        Dataset dataset = new DatasetBreastCancer();
        TSK modelTSK = new TSK(dataset.getXLength(), 30, 1);

        dataset.shuffle();

        // System.out.println(Math.log(Math.abs(-0)));
        // String filename = "diabet";
        teach(modelTSK, dataset, 64);
        // TSK modelTSK = loadModel(filename);
        // double[] predict = modelTSK.evaluate(dataset.getValues(), dataset.getD(), dataset.getClassesLength());
    }

    public static void saveModel(String filename, TSK model) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) { 
            oos.writeObject(model);
            System.out.println("Object has been serialized: " + model); 
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static TSK loadModel(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) { 
            TSK model = (TSK) ois.readObject();
            System.out.println("Object has been deserialized: " + model);
            return model;
        } catch (IOException | ClassNotFoundException e) { 
            e.printStackTrace();
            return null;
        }
    }
    public static TSK teach(TSK modelTSK, Dataset dataset, int batchSize) throws FileNotFoundException, IOException {
        double predict[];
        predict = modelTSK.evaluate(dataset.getValues(), dataset.getD(), dataset.getClassesLength());
        HybridLearningAlgorithm hla = new HybridLearningAlgorithm(modelTSK, dataset.getValues(), dataset.getD(), 1);

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        int i = 0;

        boolean teaching = true;
        while(teaching) {
            dataset.shuffle();
            i++;
            for (int j = 0; j < dataset.getDatasetLength() / batchSize; j++) {
                hla.learningTSKBatchFirstStep(batchSize, j);
                System.out.print("first step: ");
                predict = modelTSK.evaluate(dataset.getValues(), dataset.getD(), dataset.getClassesLength());

                hla.learningTSKBatchSecondStep(batchSize, j);
                System.out.print("second step: ");
                predict = modelTSK.evaluate(dataset.getValues(), dataset.getD(), dataset.getClassesLength());
                // System.out.println(Arrays.toString(modelTSK.getB()) +
                //     "\n" + Arrays.toString(modelTSK.getC()) +
                //     "\n" + Arrays.toString(modelTSK.getSigma()) + "\n");
            }
            System.out.println("After "+i+" epoch");
            //predict = modelTSK.evaluate(dataset.getValues(), dataset.getD(), dataset.getClassesLength());
            if(i % 1 == 0) {
                System.out.println("Press 1..3 to save (diabet, iris, breast_cancer), press 4 to remove, press any to continue");
                char x = (char) reader.read();
                boolean isSaving = true;
                while(isSaving) {
                    isSaving = false;
                    teaching = false;
                    switch(reader.read()) {
                        case '1' -> saveModel("diabet", modelTSK);
                        case '2' -> saveModel("iris", modelTSK);
                        case '3' -> saveModel("breast_cancer", modelTSK);
                        case '4' -> System.out.println("removed");
                        default -> {
                            teaching = true;
                        }
                    }
                }
            }
        }
        return modelTSK;
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