#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace lie {
enum class Direction { Left, Right };

inline Eigen::Matrix3d cross(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return m;
}

// Tangent space is always a VectorXd of varying sizes.
// For Reals this is a bit confusing because we do differentiate
// between Tangent (Eigen::Vector) vs the Lie group element which
// is the struct with all the Plus/Minus shenanigans

template <int N> struct Reals {
  static constexpr int DoF = N;
  Eigen::Matrix<double, N, 1> x;
  using Tangent = Eigen::Matrix<double, N, 1>;
  static Reals Exp(const Tangent &xi) { return Reals{xi}; }
  static Tangent Log(const Reals &x) { return x.x; }
  Reals Inverse() const { return Reals{-x}; }
  Reals operator*(const Reals &X2) const { return Reals{x + X2.x}; }
};

struct SO3 {
  static constexpr int DoF = 3;
  using Tangent = Eigen::Matrix<double, DoF, 1>;
  using TangentMatrix = Eigen::Matrix<double, DoF, DoF>;
  using Element = Eigen::Matrix3d;

  Eigen::Matrix3d Value{Eigen::Matrix3d::Identity()};

  static Tangent vee(const Eigen::Matrix3d &M) {
    Tangent v;
    v << M(2, 1), M(0, 2), M(1, 0);
    return v;
  }

  static TangentMatrix wedge(const Tangent &v) { return cross(v); }

  static SO3 Exp(const Tangent &xi) {
    double phi = xi.norm();
    if (phi < 1E-6) {
      return SO3{Eigen::Matrix3d::Identity() + cross(xi)};
    }
    Eigen::Vector3d u = xi / phi;
    return SO3{Eigen::Matrix3d::Identity() + std::sin(phi) * cross(u) +
               (1.0 - std::cos(phi)) * cross(u) * cross(u)};
  }

  static Tangent Log(const SO3 &C) {
    double cos = std::clamp((C.Value.trace() - 1.0) * 0.5, -1.0, 1.0);
    double theta = std::acos(cos);
    Eigen::Vector3d w = vee(C.Value - C.Value.transpose());

    if (theta < 1e-6)
      return 0.5 * w; // θ/(2 sinθ) → 1/2
    return (theta / (2.0 * std::sin(theta))) * w;
  }

  SO3 Inverse() const { return SO3{Value.transpose()}; }

  static SO3 Eye() { return SO3{}; }

  static TangentMatrix leftJacobian(const Tangent &phi) {
    double t = phi.norm();
    Eigen::Matrix3d W = cross(phi);
    if (t < 1e-8) // W coeff → 1/2, W² coeff → 1/6
      return Eigen::Matrix3d::Identity() + 0.5 * W + (1.0 / 6.0) * W * W;
    double a = (1.0 - std::cos(t)) / (t * t);
    double b = (t - std::sin(t)) / (t * t * t);
    return Eigen::Matrix3d::Identity() + a * W + b * W * W;
  }
  static TangentMatrix leftJacobianInverse(const Tangent &phi) {
    double t = phi.norm();
    Eigen::Matrix3d W = cross(phi);

    if (t < 1e-8) // W coeff → -1/2, W² coeff → 1/12
      return Eigen::Matrix3d::Identity() - 0.5 * W + (1.0 / 12.0) * W * W;

    double a = (1.0 / (t * t)) - (1.0 + std::cos(t)) / (2.0 * t * std::sin(t));
    return Eigen::Matrix3d::Identity() - 0.5 * W + a * W * W;
  }

  std::string to_line() const {
    Eigen::Quaterniond q(Value);
    if (q.w() < 0)
      q.coeffs() = q.coeffs() * -1.0;

    std::ostringstream os;
    os.imbue(std::locale::classic()); // period integer/decimal separation
                                      // regadless of locale
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
    return os.str();
  }

  SO3 operator*(const SO3 &X2) const { return SO3{Value * X2.Value}; }
  Eigen::Vector3d operator*(const Eigen::Vector3d &r) const {
    return Value * r;
  }

  static SO3 read(std::istringstream &is) {
    // ordering corresponds to TUM
    // so that SE3 can reuse this with its owns tream
    double w, x, y, z;
    is >> x >> y >> z >> w;
    if (!is)
      throw std::runtime_error("SO3 reading from line failed");
    Eigen::Quaterniond q(w, x, y, z);
    q.normalize();
    return SO3{q.toRotationMatrix()};
  }

  static SO3 from_line(std::string line) {
    std::istringstream is(line);
    is.imbue(std::locale::classic());
    return SO3::read(is);
  }
};

inline std::ostream &operator<<(std::ostream &os, const SO3 &C) {
  os << "SO3: " << C.Value;
  return os;
}

struct SE3 {
  // For given reference frames A and B
  // the pose is generally T_BtoA
  // With C_BtoA and p_BinA.
  // In our typical notation its T_AB with C_AB, r_A^BA.
  static constexpr int DoF = 6;
  using Tangent = Eigen::Matrix<double, DoF, 1>;
  using TangentMatrix = Eigen::Matrix<double, DoF, DoF>;

  SO3 C{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};

  SE3(){};
  SE3(const SO3 &C_, const Eigen::Vector3d &p_) {
    C = C_;
    p = p_;
  }

  SE3(const Eigen::Matrix4d &T) {
    C = SO3{T.block<3, 3>(0, 0)};
    p = T.block<3, 1>(0, 3);
  }
  static SE3 Exp(const Tangent &xi) {
    // Eigen::Vector3d phi = xi.block<3, 1>(0, 0);
    // Eigen::Vector3d rho = xi.block<3, 1>(3, 0);

    return SE3{SO3::Eye(), Eigen::Vector3d::Zero()};
  }

  static SE3 Eye() { return SE3(); }
  SE3 Inverse() const { return SE3(C.Inverse(), -C.Inverse().Value * p); }

  // These Left Jacobians their inverses were LLM'd but we test them (as
  // Jacobian of exponential map, and the correspodning inverse respecitvely. )
  // Q matrix (180) — helper used by both J_l and J_l^{-1}
  static Eigen::Matrix3d Q(const Eigen::Vector3d &phi,
                           const Eigen::Vector3d &rho) {

    double t = phi.norm();
    Eigen::Matrix3d Wp = cross(rho); // [ρ]×
    Eigen::Matrix3d W = cross(phi);  // [θ]×

    if (t < 1e-8) {
      // Leading-order small-angle expansion of each scalar coefficient
      return 0.5 * Wp + (1.0 / 6.0) * (W * Wp + Wp * W + W * Wp * W) -
             (1.0 / 24.0) * (W * W * Wp + Wp * W * W - 3.0 * W * Wp * W) -
             (1.0 / 120.0) * (W * Wp * W * W + W * W * Wp * W);
    }

    double t2 = t * t, t3 = t2 * t, t4 = t2 * t2, t5 = t4 * t;

    double c1 = (t - std::sin(t)) / t3;
    double c2 = (1.0 - t2 / 2.0 - std::cos(t)) / t4;
    double c3 = 0.5 * (c2 - 3.0 * (t - std::sin(t) - t3 / 6.0) / t5);

    Eigen::Matrix3d W2 = W * W;

    return 0.5 * Wp + c1 * (W * Wp + Wp * W + W * Wp * W) -
           c2 * (W2 * Wp + Wp * W2 - 3.0 * W * Wp * W) -
           c3 * (W * Wp * W2 + W2 * Wp * W);
  }

  static TangentMatrix leftJacobian(const Tangent &xi) {
    Eigen::Vector3d phi = xi.block<3, 1>(0, 0);
    Eigen::Vector3d rho = xi.block<3, 1>(3, 0);

    Eigen::Matrix3d Jl = SO3::leftJacobian(phi);
    Eigen::Matrix3d Qm = Q(phi, rho);

    TangentMatrix J = TangentMatrix::Zero();
    J.block<3, 3>(0, 0) = Jl;
    J.block<3, 3>(0, 3) = Qm;
    J.block<3, 3>(3, 3) = Jl;
    return J;
  }
  static TangentMatrix leftJacobianInverse(const Tangent &xi) {
    Eigen::Vector3d phi = xi.block<3, 1>(0, 0);
    Eigen::Vector3d rho = xi.block<3, 1>(3, 0);
    Eigen::Matrix3d Jli = SO3::leftJacobianInverse(phi);
    Eigen::Matrix3d Qm = Q(phi, rho);

    TangentMatrix Ji = TangentMatrix::Zero();
    Ji.block<3, 3>(0, 0) = Jli;
    Ji.block<3, 3>(0, 3) = -Jli * Qm * Jli;
    Ji.block<3, 3>(3, 3) = Jli;
    return Ji;
  }

  Eigen::Matrix4d toMatrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
    T.block<3, 3>(0, 0) = C.Value;
    T.block<3, 1>(0, 3) = p;
    T(3, 3) = 1.;

    return T;
  }
  SE3 operator*(const SE3 &X2) const {
    SO3 C1 = C;
    SO3 C2 = X2.C;
    Eigen::Vector3d r1 = p;
    Eigen::Vector3d r2 = X2.p;

    return SE3{C1 * C2, C1.Value * r2 + r1};
  }
  Eigen::Vector3d operator*(const Eigen::Vector3d &r) const {
    return C * r + p;
  }
  std::string to_line() const {
    // this corresponds to TUM when combined with the SO3 to_line convnetion.
    std::string C_str = SO3{C}.to_line();
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << p.x() << " " << p.y() << " " << p.z() << " " << C_str;
    return os.str();
  }

  static SE3 read(std::istringstream &is) {
    double x, y, z;
    Eigen::Vector3d r;
    is >> x >> y >> z;
    r << x, y, z;
    SO3 C = SO3::read(is);
    return SE3{C, r};
  }

  static SE3 from_line(std::string line) {
    std::istringstream is(line);
    is.imbue(std::locale::classic());
    return read(is);
  }
};

inline std::ostream &operator<<(std::ostream &os, const SE3 &T) {
  os << T.toMatrix();
  return os;
}

template <typename G> struct State {
  double stamp{0.0};
  G x{};
  std::string to_line() const {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << stamp << " " << x.to_line();
    return os.str();
  }
  State() = default;
  State(double stamp_, G x_) : stamp(stamp_), x(x_) {}
  static State from_line(std::string str) {
    std::istringstream is(str);
    is.imbue(std::locale::classic());
    double stamp;
    is >> stamp;
    G x = G::read(is);
    return State{stamp, x};
  }
};

template <typename G> G Plus(G X, typename G::Tangent tau, Direction dir) {
  G RHS = G::Exp(tau);
  if (dir == Direction::Right)
    return X * RHS;
  if (dir == Direction::Left)
    return RHS * X;
  throw std::runtime_error("Unspecified direction.");
}

template <typename G> typename G::Tangent Minus(G Y, G X, Direction dir) {
  if (dir == Direction::Right)
    return G::Log(X.Inverse() * Y);
  if (dir == Direction::Left)
    return G::Log(Y * X.Inverse());
  throw std::runtime_error("Unspecified direction.");
}

template <typename F>
Eigen::MatrixXd numericJacobianVector(const F &f, const Eigen::VectorXd &x,
                                      double eps) {
  size_t dof = x.size();
  Eigen::VectorXd fbar = f(x);
  size_t out_size = fbar.size();

  Eigen::MatrixXd jac(out_size, dof);

  for (size_t i = 0; i < dof; i++) {
    Eigen::VectorXd e(dof);
    e.setZero();
    e(i) = 0.5 * eps;
    Eigen::VectorXd df = f(x + e) - f(x - e);
    Eigen::VectorXd Ji = df / eps;
    jac.block(0, i, out_size, 1) = Ji;
  }
  return jac;
}

// general lie input version
// have to specify left/right for input and for output.
template <typename F, typename InputGroup>
Eigen::MatrixXd numericJacobian(const F &f, const InputGroup &X, double eps,
                                Direction input_dir, Direction output_dir) {
  //
  using OutputGroup = std::invoke_result_t<F, InputGroup>;

  size_t dof = InputGroup::DoF;
  OutputGroup Fbar = f(X);
  size_t out_size = OutputGroup::DoF;
  Eigen::MatrixXd jac(out_size, dof);

  for (size_t i = 0; i < dof; i++) {
    Eigen::VectorXd e(dof);
    e.setZero();
    e(i) = 0.5 * eps;

    OutputGroup F2 = f(Plus(X, e, input_dir));
    OutputGroup F1 = f(Plus(X, -e, input_dir));

    Eigen::VectorXd df = Minus(F2, F1, output_dir);
    Eigen::VectorXd Ji = df / eps;
    jac.block(0, i, out_size, 1) = Ji;
  }
  return jac;
}

} // namespace lie