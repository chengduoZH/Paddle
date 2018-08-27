/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <atomic>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace operators {
namespace math {

template <typename T, typename BinaryFunctor, typename UnaryFunctor>
struct BinaryCompoundFunctor {
  BinaryCompoundFunctor(const BinaryFunctor func1, const UnaryFunctor func2)
      : func1_(func1), func2_(func2) {}
  // Z = BinaryFunctor(X, UnaryFunctor(Y))

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(x, func2_(y)); }

  inline HOSTDEVICE T GetOutUseIntermediateOut(T x, T intermediat_out) {
    return func1_(x, intermediat_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return func2_(y); }

  BinaryFunctor func1_;
  UnaryFunctor func2_;
};

template <typename T, typename UnaryFunctor, typename BinaryFunctor>
struct UnaryCompoundFunctor {
  UnaryCompoundFunctor(const UnaryFunctor func1, const BinaryFunctor func2)
      : func1_(func1), func2_(func2) {}
  // Z = UnaryFunctor(BinaryFunctor(X, Y))

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(func2_(x, y)); }

  inline HOSTDEVICE T GetOutUseIntermediateOut(T x, T intermediat_out) {
    return func1_(intermediat_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return func2_(x, y); }

  UnaryFunctor func1_;
  BinaryFunctor func2_;
};

// FIXME(zcd): DBinaryFun and DUnaryFun have to method to get
// the dx, one is to use the 'out', and the other is not to use it.
// the former method will save the time of recomputing the
// 'out', but it must occupy the memory to store the 'out'.
// While the later method can avoid occupying this memory,
// but it must recompute the 'out'.
template <typename T, typename DBinaryFun, typename UnaryFun>
struct BinaryCompoundGradDxFunctor {
  BinaryCompoundGradDxFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    return dout * d_binary_fun_.Dx(x, unary_fun_(y));
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    return dout * d_binary_fun_.Dx(x, intermediate_out);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

template <typename T, typename DBinaryFun, typename UnaryFun,
          typename DUnaryFun>
struct BinaryCompoundGradDyFunctor {
  BinaryCompoundGradDyFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun,
                              const DUnaryFun &d_unary_fun)
      : d_binary_fun_(d_binary_fun),
        unary_fun_(unary_fun),
        d_unary_fun_(d_unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    return dout * d_binary_fun_.Dy(x, unary_fun_(y)) * d_unary_fun_(y);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    return dout * d_binary_fun_.Dy(x, intermediate_out) *
           d_unary_fun_(y, intermediate_out);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
  DUnaryFun d_unary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDxFunctor {
  UnaryCompoundGradDxFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_.Dx(x, y);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(intermediate_out);
    } else {
      base = dout * d_unary_fun_(intermediate_out, out);
    }
    return base * d_binary_fun_.Dx(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDyFunctor {
  UnaryCompoundGradDyFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_.Dy(x, y);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(intermediate_out);
    } else {
      base = dout * d_unary_fun_(intermediate_out, out);
    }
    return base * d_binary_fun_.Dy(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

class CompoundFunctor {
 public:
  template <typename DeviceContext, typename T>
  virtual void Compute(const framework::ExecutionContext &ctx,
                       const framework::Tensor &in_x,
                       const framework::Tensor &in_y,
                       std::vector<framework::Tensor *> *outputs) const = 0;
};

class Registrar {
 public:
  // In our design, various kinds of compound_functores,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_COMPOUNDFUNCTOR
  // macros to
  // call this method. So, as long as the callee code calls USE_COMPOUNDFUNCTOR,
  // the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

class CompoundFunctorRegistry {
 public:
  using CompoundFunctorCreator =
      std::function<std::unique_ptr<CompoundFunctor>()>;

  static CompoundFunctorRegistry &Instance();

  bool Has(const std::string &functor_type) const {
    return map_.find(functor_type) != map_.end();
  }

  void Insert(const std::string &functor_type,
              const CompoundFunctorCreator &compound_functor) {
    PADDLE_ENFORCE(!Has(functor_type), "Functor %s has been registered",
                   functor_type);
    map_.insert({functor_type, compound_functor});
  }

  std::unique_ptr<CompoundFunctor> Get(const std::string &functor_type) const {
    PADDLE_ENFORCE(Has(functor_type),
                   "CompoundFunctor %s has not been registered", functor_type);
    return map_.at(functor_type)();
  }

 private:
  CompoundFunctorRegistry() = default;

  std::unordered_map<std::string, CompoundFunctorCreator> map_;

  DISABLE_COPY_AND_ASSIGN(CompoundFunctorRegistry);
};

template <typename CompoundFunctor>
struct CompoundFunctorRegistrar : public Registrar {
  explicit CompoundFunctorRegistrar(const char *functor_type) {
    PADDLE_ENFORCE(!CompoundFunctorRegistry::Instance().Has(functor_type),
                   "'%s' is registered more than once.", functor_type);
    CompoundFunctorRegistry::Instance().Insert(
        functor_type, [this]() -> std::unique_ptr<CompoundFunctor> {
          std::unique_ptr<CompoundFunctor> compound_functor(
              new CompoundFunctor());
          return compound_functor;
        });
  }
};

}  // namespace math

#define STATIC_ASSERT_COMPOUNDFUNCTOR_GLOBAL_NAMESPACE(uniq_name, msg)        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

// Register a new compound_functor that can be applied on the
// fused_elemwise_activation_op.
#define REGISTER_COMPOUNDFUNCTOR(compound_functor_type, compound_functor)      \
  STATIC_ASSERT_COMPOUNDFUNCTOR_GLOBAL_NAMESPACE(                              \
      __reg_compound_functor__##compound_functor_type,                         \
      "REGISTER_COMPOUNDFUNCTOR must be called in global namespace");          \
  static ::paddle::operators::math::CompoundFunctorRegistrar<compound_functor> \
      __compound_functor_registrar_##compound_functor_type##__(                \
          #compound_functor_type);                                             \
  int TouchCompoundFunctorRegistrar_##compound_functor_type() {                \
    __compound_functor_registrar_##compound_functor_type##__.Touch();          \
    return 0;                                                                  \
  }

#define USE_COMPOUNDFUNCTOR(compound_functor_type)                    \
  STATIC_ASSERT_COMPOUNDFUNCTOR_GLOBAL_NAMESPACE(                     \
      __use_compound_functor_itself_##compound_functor_type,          \
      "USE_COMPOUNDFUNCTOR must be called in global namespace");      \
  extern int TouchCompoundFunctorRegistrar_##compound_functor_type(); \
  static int use_compound_functor_itself_##compound_functor_type##_   \
      __attribute__((unused)) =                                       \
          TouchCompoundFunctorRegistrar_##compound_functor_type()

}  // namespace operators
}  // namespace paddle
