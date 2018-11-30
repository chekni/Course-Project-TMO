package com.company;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

class Matrix {
    private String filePath;
    private int rowsCount;
    private int colsCount;
    private double matrixArray[][];
    private Scanner scanner;

    public Matrix(int n) {
        rowsCount = n;
        colsCount = n;
        matrixArray = new double[n][n];
    }


    public Matrix(String filePath) throws FileNotFoundException {
        this.filePath = filePath;
        this.scanner = new Scanner(new File(this.filePath));
        this.rowsCount = scanner.nextInt();
        this.colsCount = scanner.nextInt();
        this.matrixArray = null;
    }

    public void getMatrixFromFile() {
        matrixArray = new double[this.rowsCount][this.colsCount];
        for (int i = 0; i < this.rowsCount; i++) {
            for (int j = 0; j < this.colsCount; j++) {
                matrixArray[i][j] = scanner.nextInt();
            }
        }
    }

    public void printMatrix() {
        if (this.matrixArray != null) {
            for (int i = 0; i < this.rowsCount; i++) {
                for (int j = 0; j < this.colsCount; j++) {
                    System.out.print(matrixArray[i][j] + " ");
                }
                System.out.println();
            }
        }
        System.out.println();
    }

    public Matrix infinitesimalGenerator(Matrix[] arr) {
        Matrix result = new Matrix(2);
        int length = arr.length;
        for (int i = 0; i < length; i++) {
            result = sumTwoMatrix(result, arr[i]);
        }
        result.printMatrix();

        return result;
    }

    public Matrix sumTwoMatrix(Matrix A, Matrix B) {
        int size = A.matrixArray.length;
        Matrix sum = new Matrix(A.matrixArray.length);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sum.matrixArray[i][j] = A.matrixArray[i][j] + B.matrixArray[i][j];
            }
        }

        return sum;
    }

}


public class Main {
    public static void main(String[] args) {
        Matrix D0 = null;
        Matrix D1 = null;
        Matrix D2 = null;
        try {
            D0 = new Matrix("inputData/D0.txt");
            D1 = new Matrix("inputData/D0.txt");
            D2 = new Matrix("inputData/D0.txt");
        } catch (FileNotFoundException exception) {
        }
        ;
        D0.getMatrixFromFile();
        D1.getMatrixFromFile();
        D2.getMatrixFromFile();


        System.out.println("D0:");
        D0.printMatrix();
        System.out.println("D1:");
        D1.printMatrix();
        System.out.println("D2:");
        D2.printMatrix();
        System.out.println();


        Scanner sc;
        int size;
        Matrix infGenerator;
        Matrix[] arrD;
        try {
            sc = new Scanner(new File("inputData/N.txt"));
            size = sc.nextInt();

            infGenerator = new Matrix(size);
            arrD = new Matrix[size + 1];
            arrD[0] = D0;
            arrD[1] = D1;
            arrD[2] = D2;

            infGenerator = infGenerator.infinitesimalGenerator(arrD);
            System.out.println("Инфинитезимальный генератор D(1): " + '\n');
            infGenerator.printMatrix();
        } catch (FileNotFoundException exception) {
        }

    }
}