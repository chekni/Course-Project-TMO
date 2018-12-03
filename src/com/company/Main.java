package com.company;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.Vector;

class Matrix {
    private String filePath;
    private int rowsCount;
    private int colsCount;
    private double matrixArray[][];
    private Scanner scanner;
    private double inverse[][];

    public Matrix(int n) {
        rowsCount = n;
        colsCount = n;
        matrixArray = new double[n][n];
        inverse = new double[n][n];
    }

    public void setMatrixArray(int i, int j, double num) {
        this.matrixArray[i][j] = num;
    }

    public void setMatrixInverse(double[][] inverse) {
        this.inverse = inverse;
    }

    public int getLength() {
        return rowsCount;
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
        inverse = new double[this.rowsCount][this.colsCount];
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

    public void printInverseMatrix() {
        if (this.inverse != null) {
            for (int i = 0; i < this.rowsCount; i++) {
                for (int j = 0; j < this.colsCount; j++) {
                    System.out.print(inverse[i][j] + " ");
                }
                System.out.println();
            }
        }
        System.out.println();
    }

    public void infinitesimalGenerator(Matrix[] arr) {
        Matrix result = new Matrix(2);
        int length = arr.length;
        for (int i = 0; i < length; i++) {
            result = sumTwoMatrix(result, arr[i]);
        }
        result.printMatrix();

        this.matrixArray = result.matrixArray;
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

    public Vector matrixMaltipleVector(Matrix matrix, Vector<Double> vector) {
        int length = matrix.rowsCount;
        double tmp = 0;
        Vector result = new Vector(length);
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < vector.size(); j++) {
                tmp = matrix.matrixArray[i][j] * vector.get(j);
            }
            result.add(i, tmp);
        }
        return result;
    }

    public double[] vectorMaltipleMatrix(double[] vector) {
        int length = this.colsCount;
        double[] result = new double[length];
        //if (vector.length !== this.rowsCount) throw Exception();
        for (int i = 0; i < length; i++) {
            double tmp = 0;
            for (int j = 0; j < this.rowsCount; j++) {
                tmp += vector[j] * this.matrixArray[j][i];
            }
            result[i] = tmp;

        }
        return result;

    }

    public double vectorMultipleVector(double[] v1, double[] v2) {
        //if (v1.length !== v2.length) throw Exception();

        double result = 0;
        for (int i = 0; i < v1.length; i++) {
            result += v1[i] * v2[i];
        }
        return result;
    }

    public double[] calculateLamda(double[] Q, double[] e, Matrix[] Di) {
        int count = Di.length;
        double[] lamda = new double[count];
        for (int i = 0; i < count; i++) {
            double[] tmp = Di[i].vectorMaltipleMatrix(Q);
            lamda[i] = vectorMultipleVector(tmp, e);

        }
        return lamda;
    }

    //(?)
    public static double[] prodVM(double[] a, double[][] B) {
        double[] res = new double[B.length];

        double sum;
        for (int i = 0; i < B.length; i++) {
            sum = 0;
            for (int j = 0; j < B[i].length; j++) {
                sum += a[j] * B[j][i];
            }
            res[i] = sum;
        }
        return res;
    }


    public static double[] Gauss(Matrix matrix, double[] f) {
        int n = matrix.getLength();

        for (int i = 0; i < n; i++) {
            matrix.inverse[i][i] = 1;
        }

        f[0] = 1;
        for (int i = 0; i < n; i++) {
            matrix.matrixArray[i][0] = 1;
        }

        double[] x = new double[n];
        double del;
        for (int i = 0; i < n; i++) {
            del = matrix.matrixArray[i][i];
            f[i] /= del;
            //делим строку
            for (int j = n - 1; j >= i; j--) {
                matrix.matrixArray[i][j] /= del;
                matrix.inverse[i][j] /= del;
            }
            for (int j = i + 1; j < n; j++) {
                del = matrix.matrixArray[j][i];
                f[j] -= del * f[i];
                for (int k = n - 1; k >= i; k--) {
                    matrix.matrixArray[j][k] -= del * matrix.matrixArray[i][k];
                    matrix.inverse[j][k] -= del * matrix.inverse[i][k];
                }
            }
        }
        /*обратный ход*/
        matrix.setMatrixInverse(getInverseMatrix(matrix));
        x[n - 1] = f[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            x[i] = f[i];
            for (int j = i + 1; j < n; j++) {
                x[i] -= matrix.matrixArray[i][j] * x[j];
            }
        }
        return x;
    }

    private static double[][] getInverseMatrix(Matrix matrix) {
        int n = matrix.getLength();
        for (int i = n - 2; i > 0; i--) {
            for (int j = n - 1; j < i; j--) {
                double del = matrix.matrixArray[i][j];
                double[] tmp = matrix.inverse[i - 1];
                for (int k = 0; k < n; k++) {
                    tmp[k] *= del;
                    matrix.inverse[i][k] += tmp[k];
                }
                matrix.inverse[i][j] = 0;
            }
        }
        return matrix.inverse;
    }

    public static Matrix sumMatrix(Matrix a, Matrix b) {
        if (a.rowsCount == b.rowsCount && a.colsCount == b.colsCount) {
            Matrix sum = new Matrix(a.rowsCount);
            for (int i = 0; i < a.rowsCount; ++i) {
                for (int j = 0; j < a.colsCount; ++j) {
                    sum.matrixArray[i][j] = a.matrixArray[i][j] + b.matrixArray[i][j];
                }
            }
            return sum;
        }
        return null;
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
        int size = 0;
        Matrix infGenerator;
        Matrix[] arrD;
        try {
            sc = new Scanner(new File("inputData/N.txt"));
            size = sc.nextInt();


        } catch (FileNotFoundException exception) {
        }

        infGenerator = new Matrix(size);
        arrD = new Matrix[size + 1];
        arrD[0] = D0;
        arrD[1] = D1;
        arrD[2] = D2;

        infGenerator.infinitesimalGenerator(arrD);
        System.out.println("Инфинитезимальный генератор D(1): ");
        infGenerator.printMatrix();


        double[] e = new double[size];
        e[0] = 1;
        for (int i = 1; i < size; i++) {
            e[i] = 0;
        }

        double[] f = new double[size];
        for (int i = 0; i < size; i++) {
            f[i] = 0;
        }
        System.out.println();

        Matrix tmp = new Matrix(size);
        double[] Q = tmp.Gauss(infGenerator, e);
        System.out.println("Print inverse matrix");
        tmp.printInverseMatrix();
        for (int i = 0; i < Q.length; i++) {
            System.out.println(Q[i] + " ");
        }

        double[] lamdaArray = tmp.calculateLamda(Q, e, arrD);
        for (int j = 0; j < lamdaArray.length; j++) {
            System.out.println(lamdaArray[j] + " ");
        }

    }
}