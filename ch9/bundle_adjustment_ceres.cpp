#include <iostream>
#include <ceres/ceres.h>
#include "common.h" //使用common.h中定义的BALProblem类读入该文件的内容
#include "SnavelyReprojectionError.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();//归一化 将所有路标点的中心置零，然后做一个合适尺度的缩放
    bal_problem.Perturb(0.1, 0.5, 0.5);//通过Perturb函数给数据加入噪声
    bal_problem.WriteToPLYFile("initial.ply");//存储最初点云
    SolveBA(bal_problem);//BA求解
    bal_problem.WriteToPLYFile("final.ply");//存储最终点云

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();//3
    const int camera_block_size = bal_problem.camera_block_size();//9
    double *points = bal_problem.mutable_points();  //三维点起始地址;
    double *cameras = bal_problem.mutable_cameras();//相机参数起始地址;

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations();//二维观测点起始位置;
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) { // 16个位姿观测;
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        //输入二维观测值;
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);//像边

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.

        // cameras是相机(9维)存储起点; 
        // 当前二维点对应的相机 camera_index_ = new int[num_observations_];
        // 这个数组中存着当前二维点对应的相机索引
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];// 当前二维点对应的三维点

        problem.AddResidualBlock(cost_function, loss_function, camera, point);//感觉像Vertex;
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}