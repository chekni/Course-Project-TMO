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

    public void setMatrixArray(double[][] matrixArray) {
        this.matrixArray = matrixArray;
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

    public static Matrix infinitesimalGenerator(Matrix[] arrD) {
        Matrix result = new Matrix(arrD[0].getLength());
        int length = arrD.length;
        for (int i = 0; i < length; i++) {
            result = sumTwoMatrix(result, arrD[i]);
        }
        return result;
    }

    public static Matrix sumTwoMatrix(Matrix A, Matrix B) {
        int size1 = A.matrixArray.length;
        int size2 = B.matrixArray[0].length;
        Matrix sum = new Matrix(A.matrixArray.length);
        for (int i = 0; i < size1; i++) {
            for (int j = 0; j < size2; j++) {
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
        int count = Di.length;
        double[] lamda = new double[count];
        for (int i = 1; i < count; i++) {
            double[] tmp = Di[i].vectorMultipleMatrix(Q);
            lamda[i] = vectorMultipleVector(tmp, e);
        }
        return lamda;
    }

    public static double[] calculateDispersion(double[] lamda, Matrix[] Di, double[] tau) {
        int n = lamda.length;
        double[] result = new double[n];
        for (int i = 1; i < n; i++) {
            Matrix sum = sumTwoMatrix(Di[0], Di[i]);
            sum.multipleMatrixOnValue(-1);
            double[] f = new double[n];
            Gauss(sum, f);
            Matrix tmp = new Matrix(sum.getMatrixInverse());

            double[] vectorMultMatr = tmp.vectorMultipleMatrix(tau);
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

    public static double[] sumTwoVectors(double[] a, double[] b) {
        double[] sum = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            sum[i] = a[i] + b[i];
        }
        return sum;
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

    public static double[] calculateKoeffKorrel(double[] lamda, Matrix[] Di, double[] tau, double[] dispersion) {
        int n = lamda.length;
        double[] result = new double[n];
        for (int i = 1; i < n; i++) {
            Matrix sum = sumTwoMatrix(Di[0], Di[i]);
            double[] f = new double[n];
            Gauss(sum, f);
            Matrix inverse = new Matrix(sum.getMatrixInverse());

            double[] res1 = inverse.vectorMultipleMatrix(tau);
            double[] res2 = Di[i].vectorMultipleMatrix(res1);
            double[] res3 = inverse.vectorMultipleMatrix(res2);
            double[] e = new double[res1.length];
            for (int j = 0; j < e.length; j++) {
                e[j] = 1;
            }
            double res4 = Matrix.vectorMultipleVector(res3, e);
            double minuend = res4 / lamda[i];
            double subtrahend = Math.pow(1 / (lamda[i]), 2);
            result[i] = (minuend - subtrahend) * (1 / dispersion[i]);
        }
        return result;
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

    private Matrix multipleMatrixOnValue(double a) {
        for (int i = 0; i < this.getLength(); i++) {
            for (int j = 0; j < this.getLength(); j++) {
                this.matrixArray[i][j] = a * this.matrixArray[i][j];
            }
        }
        return this;
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
        //диагональные матрицы
        Q[0][0] = Di[0];
        for (int i = 1; i < N; i++) {
            Matrix I = Matrix.I(i + 1);
            Q[i][i] = subTwoMatrix(kroneckerMultiple(I, Di[0]), kroneckerMultiple(I, Matrix.I(W + 1)));
        }
        Matrix mult1, mult2;
        Matrix infGen = Matrix.infinitesimalGenerator(Di);
        mult1 = Matrix.kroneckerMultiple(I(N + 1), infGen);
        mult2 = Matrix.kroneckerMultiple(createDiagonalMatrix(N + 1, 0, mu), I(W + 1));

        Q[N][N] = Matrix.subTwoMatrix(mult1, mult2);

        //поддиагональные матрицы
        for (int n = 1; n < N + 1; n++) {
            Matrix M1 = new Matrix(n + 1, n);
            Matrix M2 = new Matrix(n + 1, n);
            int m_from = 0;
            for (int j = 0; j < n; j++, m_from++) {
                M1.matrixArray[j][j] = (n - m_from) * mu[1];
            }
            m_from = 1;
            for (int j = 1; j < n; j++, m_from++) {
                M1.matrixArray[j][j] = m_from * mu[0];
            }

            Matrix fed = Matrix.sumTwoMatrix(M1, M2);
            fed.printMatrix();
            Q[n][n - 1] = Matrix.sumTwoMatrix(M1, M2);
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

    public static Matrix[] G_i(Matrix[][] inf_gen_Q, int N) {
        Matrix[] G = new Matrix[N];
        Matrix tmp = inf_gen_Q[N][N].multipleMatrixOnValue(-1);
        double[] f = new double[N];
        for (int i = 1; i < N; i++) {
            f[i] = 0;
        }
        Matrix.Gauss(tmp, f);
        Matrix inversion = new Matrix(N);
        inversion.setMatrixArray(tmp.getMatrixInverse());
        G[N - 1] = Matrix.matrixMultipleMatrix(inversion, inf_gen_Q[N][N - 1]);

        for (int i = N - 2; i >= 0; i--) {
            Matrix mult = Matrix.matrixMultipleMatrix(inf_gen_Q[i + 1][i + 2], G[i + 1]);
            Matrix sum = Matrix.sumTwoMatrix(inf_gen_Q[i + 1][i + 1], mult);
            sum = sum.multipleMatrixOnValue(-1);
            Matrix.Gauss(sum, f);
            inversion.setMatrixArray(tmp.getMatrixInverse());
            Matrix result = Matrix.matrixMultipleMatrix(inversion, inf_gen_Q[i + 1][i]);

            G[i] = result;
        }
        return G;
    }


    public static Matrix[][] new_inf_gen_Q(Matrix[][] Q, Matrix[] G) {
        int N = G.length;
        for (int i = 0; i < N; i++) {
            Q[i][i] = Matrix.sumTwoMatrix(Q[i][i], Matrix.matrixMultipleMatrix(Q[i][i + 1], G[i]));
        }
        return Q;
    }

    public static Matrix[] F(Matrix[][] Q, int W) {
        int N = Q.length;
        Matrix[] F = new Matrix[N + 1];
        F[0] = I(N + 1);
        for (int i = 1; i <= N; i++) {
            Matrix tmp = Matrix.matrixMultipleMatrix(F[i - 1], Q[i - 1][i]);
            double[] f = new double[N];
            for (int j = 1; j < N; j++) {
                f[j] = 0;
            }
            Matrix inverse = Q[i][i];
            inverse.multipleMatrixOnValue(-1);
            Matrix.Gauss(inverse, f);
            inverse = new Matrix(Q[i][i].getMatrixInverse());

            Matrix result = Matrix.matrixMultipleMatrix(tmp, inverse);
            F[i] = result;
        }

        return F;
    }

    public static double[] calculateQ0(Matrix[] F, Matrix[][] Q, int W) {
        int N = F.length;
        int countCols = F[0].colsCount;
        double[] e = new double[countCols];
        for (int j = 0; j < countCols; j++) {
            e[j] = 1;
        }

        double[] sum = new double[countCols];
        for (int i = 0; i < N; i++) {
            sum = Matrix.sumTwoVectors(sum, Matrix.matrixMultipleVector(F[i], e));
        }
        Matrix tmp = Q[0][0];
        double[] f = new double[countCols];

        return Matrix.Gauss(tmp, f);
    }

    private static double[] vectorMultipleMatrix(Matrix A, double[] b) {
        double[] res = new double[A.colsCount];
        double sum = 0;
        for (int i = 0; i < A.colsCount; i++) {
            sum = 0;
            for (int j = 0; j < A.rowsCount; j++) {
                sum += b[j] * A.matrixArray[i][j];
            }
            res[i] = sum;
        }

        return res;
    }

    public static double[][] calculateQi(Matrix[] F, double[] q0, int W) {
        //размещаем вектора qi в матрице горизонтально
        int N = F.length;
        double[][] qi = new double[q0.length][N];
        for (int i = 1; i < N; i++) {
            qi[i] = Matrix.vectorMultipleMatrix(F[i], q0);
        }

        return qi;
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
        int N = 0;
        double[] mu = new double[N];
        int W = 1;
        Matrix infGenerator;
        Matrix[] arrD;
        try {
            sc = new Scanner(new File("inputData/N.txt"));
            N = sc.nextInt();

        } catch (FileNotFoundException exception) {
        }

        try {
            sc = new Scanner(new File("inputData/W.txt"));
            W = sc.nextInt();

        } catch (FileNotFoundException exception) {
        }

        try {
            Scanner sc2 = new Scanner(new File("inputData/Mu.txt"));
            mu = new double[N];
            for (int i = 0; i < N; i++) {
                mu[i] = sc2.nextDouble();
            }


        } catch (FileNotFoundException exception) {
        }

        infGenerator = new Matrix(N);
        arrD = new Matrix[N + 1];
        arrD[0] = D0;
        arrD[1] = D1;
        arrD[2] = D2;

        infGenerator = Matrix.infinitesimalGenerator(arrD);
        System.out.println("D(1): ");
        infGenerator.printMatrix();
        System.out.println();

        double[] e = new double[N];

        Matrix tmp = new Matrix(N);
        double[] tau = tmp.Gauss(infGenerator, e);
        for (int i = 0; i < tau.length; i++) {
            System.out.println(tau[i] + " ");
        }
        System.out.println();

        double[] lamdaArray = tmp.calculateLamda(tau, e, arrD);
        for (int j = 1; j < lamdaArray.length; j++) {
            System.out.println("lamda " + j + " = " + lamdaArray[j] + " ");
        }
        System.out.println();

        double[] dispersion = Matrix.calculateDispersion(lamdaArray, arrD, tau);
        for (int j = 1; j < dispersion.length; j++) {
            System.out.println("dispersion " + j + " = " + dispersion[j] + " ");
        }
        System.out.println();

        double[] koeffKorel = Matrix.calculateKoeffKorrel(lamdaArray, arrD, tau, dispersion);
        for (int j = 1; j < koeffKorel.length; j++) {
            System.out.println("koeff korellaci " + j + " = " + koeffKorel[j] + " ");
        }
        System.out.println();



        /*////////////////////////////////////////*/
        Matrix[][] Q_inf_gen = new Matrix[N + 1][N + 1];
        Q_inf_gen = Matrix.infGeneratorQ(arrD, mu, Q_inf_gen, N, W);

        System.out.println("Q[0][0]:");
        Q_inf_gen[0][0].printMatrix();
        System.out.println();

        System.out.println("Q[0][1]:");
        Q_inf_gen[0][0].printMatrix();
        System.out.println();


//        for (int i = 1; i < N-1; i++) {
//            for (int j = i; j <= i+2; j++) {
//                Q_inf_gen[i][j].printMatrix();
//                System.out.println();
//            }
//        }


    }
}