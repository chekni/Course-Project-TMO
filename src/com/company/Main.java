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

    public Matrix(double[][] matrArray) {
        int n = matrArray.length;
        rowsCount = n;
        colsCount = n;
        matrixArray = matrArray;
        inverse = new double[n][n];
    }

    public Matrix(int rCount, int cCount) {
        rowsCount = rCount;
        colsCount = cCount;
        matrixArray = new double[rCount][cCount];
        inverse = new double[rCount][cCount];
    }

    public void setMatrixArray(int i, int j, double num) {
        this.matrixArray[i][j] = num;
    }

    public void setMatrixInverse(double[][] inverse) {
        this.inverse = inverse;
    }

    public double[][] getMatrixInverse() {
        return this.inverse;
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

    public static Matrix sumTwoMatrix(Matrix A, Matrix B) {
        int size = A.matrixArray.length;
        Matrix sum = new Matrix(A.matrixArray.length);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sum.matrixArray[i][j] = A.matrixArray[i][j] + B.matrixArray[i][j];
            }
        }
        return sum;
    }

    public static Matrix subTwoMatrix(Matrix A, Matrix B) {
        int size = A.matrixArray.length;
        Matrix sub = new Matrix(A.matrixArray.length);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sub.matrixArray[i][j] = A.matrixArray[i][j] - B.matrixArray[i][j];
            }
        }
        return sub;
    }

    public double[] vectorMultipleMatrix(double[] vector) {
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

    public static double vectorMultipleVector(double[] v1, double[] v2) {
        //if (v1.length !== v2.length) throw Exception();

        double result = 0;
        for (int i = 0; i < v1.length; i++) {
            result += v1[i] * v2[i];
        }
        return result;
    }

    public double[] calculateLamda(double[] Q, double[] e, Matrix[] Di) {
        int count = Di.length - 1;
        double[] lamda = new double[count];
        for (int i = 1; i < count; i++) {
            double[] tmp = Di[i].vectorMultipleMatrix(Q);
            lamda[i] = vectorMultipleVector(tmp, e);
        }
        return lamda;
    }

    public double[] calculateDispersion(double[] lamda, Matrix[] Di, double[] Q) {
        int n = lamda.length - 1;
        double[] result = new double[n];
        for (int i = 1; i < n; i++) {
            Matrix sum = sumTwoMatrix(Di[0], Di[i]);
            sum.multipleMatrixOnValue(-1);
            double[] f = new double[n];
            Gauss(sum, f);
            Matrix tmp = new Matrix(sum.getMatrixInverse());

            double[] vectorMultMatr = tmp.vectorMultipleMatrix(Q);
            double[] e = new double[vectorMultMatr.length];
            for (int j = 0; j < e.length; j++) {
                e[j] = 1;
            }
            double minuend = vectorMultipleVector(vectorMultMatr, e);
            minuend = minuend * 2 / lamda[i];
            double subtrahend = Math.pow(1 / (lamda[i]), 2);

            result[i] = minuend - subtrahend;
        }
        return result;
    }

    public static double[] matrixMultipleVector(Matrix A, double[] vector) {
        int n = vector.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += A.matrixArray[i][j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    public static Matrix matrixMultipleMatrix(Matrix A, Matrix B) {
        int n = A.getLength();
        double sum = 0;
        Matrix result = new Matrix(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A.matrixArray[i][k] * B.matrixArray[k][j];
                }
                result.setMatrixArray(i, j, sum);
            }
        }
        return result;
    }

    public double[] calculateKoeffKorrel(double[] lamda, Matrix[] Di, double[] Q, double[] Vi) {
        int n = lamda.length;
        double[] result = new double[n - 1];
        for (int i = 1; i < n; i++) {
            Matrix sum = sumTwoMatrix(Di[0], Di[i]);
            double[] f = new double[n];
            Gauss(sum, f);
            Matrix inverse = new Matrix(sum.getMatrixInverse());

            double[] res1 = inverse.vectorMultipleMatrix(Q);
            double[] res2 = Di[i].vectorMultipleMatrix(res1);
            double[] res3 = inverse.vectorMultipleMatrix(res2);
            double[] e = new double[res1.length];
            for (int j = 0; j < e.length; j++) {
                e[j] = 1;
            }
            double res4 = Matrix.vectorMultipleVector(res3, e);
            double minuend = res4 / lamda[i];
            double subtrahend = Math.pow(1 / (lamda[i]), 2);
            result[i] = (minuend - subtrahend) * (1 / Vi[i]);
        }
        return result;
    }


    public double[] Gauss(Matrix matrix, double[] f) {
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

    private void multipleMatrixOnValue(double a) {
        for (int i = 0; i < this.getLength(); i++) {
            for (int j = 0; j < this.getLength(); j++) {
                this.matrixArray[i][j] = a * this.matrixArray[i][j];
            }
        }
    }


    /*(?)*/
    public static Matrix kroneckerMultiple(Matrix A, Matrix B) {
        int k = B.getLength();
        Matrix result = new Matrix(A.getLength() * B.getLength());
        for (int ia = 0; ia < A.getLength(); ia++) {
            for (int ja = 0; ja < A.getLength(); ja++) {

                for (int ib = 0; ib < B.getLength(); ib++) {
                    for (int jb = 0; jb < B.getLength(); jb++) {
                        result.matrixArray[k * ia + ib][k * ja + jb] = A.matrixArray[ia][ja] * B.matrixArray[ib][jb];
                    }
                }

            }
        }

        return result;
    }


    private static Matrix I(int n) {
        Matrix I = new Matrix(n);
        for (int i = 0; i < n; i++) {
            I.matrixArray[i][i] = 1;
        }
        return I;
    }


    private static Matrix createDiagonalMatrix(int n, double m_from, double[] mu) {
        Matrix res = new Matrix(n);
        for (int i = 0; i < n; i++, m_from++) {
            res.matrixArray[i][i] = (n - m_from) * mu[0] + m_from * mu[1];
        }
        return res;
    }

    private static Matrix createUnderDiagonalMatrix(int n, double m_from, double[] mu) {
        Matrix res = new Matrix(n);
        for (int i = 1; i < n; i++, m_from++) {
            res.matrixArray[i][i - 1] = m_from * mu[1];
        }
        return res;
    }

    public static Matrix[][] infGeneratorQ(Matrix[] Di, double[] mu, Matrix[][] Q, int N, int W) {
        Q[0][0] = Di[0];
        for (int i = 1; i < N; i++) {
            Q[i][i] = subTwoMatrix(kroneckerMultiple(Matrix.I(i + 1), Di[0]), kroneckerMultiple(Matrix.createDiagonalMatrix(i, 0, mu), Matrix.I(W + 1)));
        }
        Matrix tmp;
        Matrix infGen = new Matrix(N + 1);
        infGen.infinitesimalGenerator(Di);
        tmp = Matrix.kroneckerMultiple(I(N + 1), infGen);
        Q[N][N] = Matrix.subTwoMatrix(tmp, Matrix.kroneckerMultiple(createDiagonalMatrix(N, 1, mu), I(W + 1)));

        //поддиагональные матрицы
        for (int i = 1; i <= N; i++) {
            tmp = Matrix.sumTwoMatrix(createDiagonalMatrix(i, 0, mu), createUnderDiagonalMatrix(i, 1, mu));
            Q[i][i - 1] = Matrix.kroneckerMultiple(tmp, I(W + 1));
        }

        //создать наддиагональную матрицу


        for (int n = 0; n < N; n++) {
            Matrix Q_tmp = new Matrix((W + 1) * (n + 1), (W + 1) * (n + 2));
            int D_i;
            int size_D = W + 1;
            for (int k = 0; k < n + 1; k++) {
                for (int t = k; t <= k + 1; t++) {
                    if (k == t) D_i = 2;
                    else D_i = 1;
                    //записываем маленькую матрицу в большую
                    for (int i = 0; i < size_D; i++) {
                        for (int j = 0; j < size_D; j++) {
                            Q_tmp.matrixArray[k * size_D + i][t * size_D + j] = Di[D_i].matrixArray[i][j];
                        }
                    }
                }
                //int startIndex = k*size_D
            }
            Q[n][n + 1] = Q_tmp;
        }

        return Q;
    }
}

public class Main {
    public static void main(String[] args) {
        Matrix D0 = null;
        Matrix D1 = null;
        Matrix D2 = null;
        try {
            D0 = new Matrix("inputData/D0.txt");
            D1 = new Matrix("inputData/D1.txt");
            D2 = new Matrix("inputData/D2.txt");
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
        double[] mu;
        Matrix infGenerator;
        Matrix[] arrD;
        try {
            sc = new Scanner(new File("inputData/N.txt"));
            size = sc.nextInt();

        } catch (FileNotFoundException exception) {
        }

        try {
            Scanner sc2 = new Scanner(new File("inputData/Mu.txt"));
            mu = new double[size];
            for (int i = 0; i < size; i++) {
                mu[i] = sc2.nextDouble();
            }


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
        infGenerator.printInverseMatrix();

        double[] lamdaArray = tmp.calculateLamda(Q, e, arrD);
        for (int j = 0; j < lamdaArray.length; j++) {
            System.out.println("lamda " + j + " = " + lamdaArray[j] + " ");
        }
        System.out.println();

        Matrix[][] Q_inf_gen = new Matrix[size][size];
        Q_inf_gen[0][0] = D0;
        double m;
        //Q = Matrix.fill_diagonal_elem(Q, arrD, m, mu);

    }
}