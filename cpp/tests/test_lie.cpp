#include <Eigen/Core>
#include <gtest/gtest.h>
#include <types/Lie.hpp>

TEST(Lie, SO3) {
  using namespace lie;
  using Vec3d = Eigen::Vector3d;
  using Mat3d = Eigen::Matrix3d;
  using R3 = Reals<3>;
  Vec3d xi;
  xi << 0.5, 0.2, 0.7;
  SO3 C = SO3::Exp(xi);

  EXPECT_TRUE((C * C.Inverse()).Value.isApprox(Mat3d::Identity(), 1e-6));
  EXPECT_TRUE(SO3::vee(SO3::wedge(xi)).isApprox(xi));
  EXPECT_TRUE(SO3::Log(SO3::Exp(xi)).isApprox(xi));

  // Left Jacobian is Jacobian of Exponential Map (with left perturbation).
  SO3::TangentMatrix jac = SO3::leftJacobian(xi);
  SO3::TangentMatrix jac_num =
      numericJacobian([](R3 in) { return SO3::Exp(in.x); }, R3{xi}, 1e-6,
                      Direction::Left, Direction::Left);
  EXPECT_TRUE(jac.isApprox(jac_num, 1e-6));
  SO3::TangentMatrix jac_inv = SO3::leftJacobianInverse(xi);
  EXPECT_TRUE((jac * jac_inv).isApprox(Mat3d::Identity()));
}

TEST(Lie, SE3) {
  using namespace lie;
  using Vec3d = Eigen::Vector3d;
  using Vec6d = Eigen::Matrix<double, 6, 1>;
  using Mat3d = Eigen::Matrix3d;
  using Mat6d = Eigen::Matrix<double, 6, 6>;
  using Mat4d = Eigen::Matrix<double, 4, 4>;

  Vec6d xi;
  xi << 0.5, 0.2, 0.7, 0.3, 0.45, 0.76;
  SE3 T = SE3::Exp(xi);

  EXPECT_TRUE((T * T.Inverse()).toMatrix().isApprox(Mat4d::Identity(), 1e-6));
  // Mat3d w = SO3::wedge(xi);
  // Vec3d xi_check = SO3::vee(SO3::wedge(xi));
  // EXPECT_TRUE(SO3::vee(SO3::wedge(xi)).isApprox(xi));
  // EXPECT_TRUE(SO3::Log(SO3::Exp(xi)).isApprox(xi));
}

TEST(Lie, NumericalVector) {
  using namespace lie;
  using Vec2d = Eigen::Vector2d;

  auto f = [](Eigen::Vector2d x) {
    Eigen::Matrix<double, 1, 1> out;
    out(0) = x.norm() * x.norm();
    return out;
  };
  Vec2d xbar;
  xbar << 1, 2;
  Eigen::MatrixXd jac = numericJacobianVector(f, xbar, 1e-6);
  Eigen::MatrixXd jac_true = 2 * xbar.transpose();
  EXPECT_TRUE(jac.isApprox(jac_true, 1e-6));
}