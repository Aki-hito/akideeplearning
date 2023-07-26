#include <iostream>

using namespace std;

#include <ctime>
//Eigen核心
#include <eigen3/Eigen/Core>
//稠密矩阵的代数运算（逆、特征值等）
#include <eigen3/Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50

//本程序演示Eigen基本类型的使用

int main() {
    //Eigen中所有向量和矩阵都是Eigen：：Matrix，他是一个模板类。它的前三个参数为数据类型、行、列
    //声明一个2*3的float矩阵
    Matrix<float, 2, 3> matrix_23;

    //同时，Eigen通过typedef提供了许多内置类型，不过底层仍是Eigen：：Matrix
    //例如，Vector3d实质上是Eigen：：Matrix<double,3,1>，即三维向量
    Vector3d v_3d;
    //这是一个样的
    Matrix<double, 3, 1> vd_3d;

    //Matrix3d实质上Eigen：：Matrix<double,3,3>
    Matrix3d matrix_33 = Matrix3d::Zero();//初始化为0
    //不确定大小可以用动态大小矩阵
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;
    //更简单的
    MatrixXd matrix_x;//这个是方阵

    //下面是对Eigen阵的操作
    //输入数据（初始化）
    matrix_23 << 1, 2, 3, 4, 5, 6;
    //输出
    cout << "matrix 2x3 from 1 to 6:\n" << matrix_23 << endl;

    //用（）访问矩阵中的元素
    cout << "print matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
        cout << endl;
    }

    //矩阵和向量相乘（实际上是矩阵和矩阵）
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    //在Eigen里不能混合两种不同类型的矩阵，像这样是错的
    //Matrix<double,2,1> result_wrong_type = matrix_23 * v_3d;
    //显示类型转换
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << "[1,2,3,4,5,6] * [3,2,1] = " << result.transpose() << endl;

    Matrix<double, 2, 1> result2 = matrix_23.cast<double>() * vd_3d;
    cout << "[1,2,3,4,5,6] * [4,5,6] = " << result.transpose() << endl;

    //同样不能搞错矩阵的维度
    //Eigen::Matrix<double,2,3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

    //一些矩阵运算
    matrix_33 = Matrix3d::Random(); //随机数矩阵
    cout << "random matrix: \n" << matrix_33 << endl;
    cout << "transpose: \n" << matrix_33.transpose() << endl; //转置
    cout << "sum: " << matrix_33.sum() << endl; //各元素和
    cout << "trace: " << matrix_33.trace() << endl; //迹
    cout << "times 10: \n" << 10 * matrix_33 << endl; //数乘
    cout << "inverse: \n" << matrix_33.inverse() << endl; //逆
    cout << "det: " << matrix_33.determinant() << endl; //行列式

    //特征值
    //实对称矩阵一定可以对角化
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;
    EigenSolver<Matrix3d> test_eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Test Eigen values = \n" << test_eigen_solver.eigenvalues() << endl;
    cout << "Test Eigen vectors = \n" << test_eigen_solver.eigenvectors() << endl;

    //解方程
    //直接求解matrix_NN * x = v_Nd方程
    //N的大小在宏里定义，由随机数生成
    //直接求逆当然可以但是运算量太大了

    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
            = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); //保证半正定
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_st = clock();//计时
    //直接求逆
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inverse is "
         << 1000 * (clock() - time_st) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    //通常实际会用矩阵分解，例如QR分解来求，速度会快很多
    clock_t time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of QR decomposition is "
         << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    //正定矩阵还可以用cholesky分解
    clock_t time_sttt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
         << 1000 * (clock() - time_sttt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    return 0;
}
