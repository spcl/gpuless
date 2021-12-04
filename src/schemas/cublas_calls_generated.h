// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_CUBLASCALLS_H_
#define FLATBUFFERS_GENERATED_CUBLASCALLS_H_

#include "flatbuffers/flatbuffers.h"

struct FBCublasCreateV2;
struct FBCublasCreateV2Builder;

struct FBCublasSetStreamV2;
struct FBCublasSetStreamV2Builder;

struct FBCublasSetMathMode;
struct FBCublasSetMathModeBuilder;

struct FBCublasSgemmV2;
struct FBCublasSgemmV2Builder;

struct FBCublasLtCreate;
struct FBCublasLtCreateBuilder;

struct FBCublasLtMatmulDescCreate;
struct FBCublasLtMatmulDescCreateBuilder;

struct FBCublasLtMatmulDescDestroy;
struct FBCublasLtMatmulDescDestroyBuilder;

struct FBCublasLtMatmulDescSetAttribute;
struct FBCublasLtMatmulDescSetAttributeBuilder;

struct FBCublasLtMatmul;
struct FBCublasLtMatmulBuilder;

struct FBCublasLtMatrixLayoutCreate;
struct FBCublasLtMatrixLayoutCreateBuilder;

struct FBCublasLtMatrixLayoutDestroy;
struct FBCublasLtMatrixLayoutDestroyBuilder;

struct FBCublasLtMatrixLayoutSetAttribute;
struct FBCublasLtMatrixLayoutSetAttributeBuilder;

struct FBCublasCreateV2 FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasCreateV2Builder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_HANDLE = 4
  };
  uint64_t virtual_handle() const {
    return GetField<uint64_t>(VT_VIRTUAL_HANDLE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_HANDLE) &&
           verifier.EndTable();
  }
};

struct FBCublasCreateV2Builder {
  typedef FBCublasCreateV2 Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_handle(uint64_t virtual_handle) {
    fbb_.AddElement<uint64_t>(FBCublasCreateV2::VT_VIRTUAL_HANDLE, virtual_handle, 0);
  }
  explicit FBCublasCreateV2Builder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasCreateV2> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasCreateV2>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasCreateV2> CreateFBCublasCreateV2(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_handle = 0) {
  FBCublasCreateV2Builder builder_(_fbb);
  builder_.add_virtual_handle(virtual_handle);
  return builder_.Finish();
}

struct FBCublasSetStreamV2 FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasSetStreamV2Builder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_HANDLE = 4,
    VT_STREAM = 6
  };
  uint64_t virtual_handle() const {
    return GetField<uint64_t>(VT_VIRTUAL_HANDLE, 0);
  }
  uint64_t stream() const {
    return GetField<uint64_t>(VT_STREAM, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_HANDLE) &&
           VerifyField<uint64_t>(verifier, VT_STREAM) &&
           verifier.EndTable();
  }
};

struct FBCublasSetStreamV2Builder {
  typedef FBCublasSetStreamV2 Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_handle(uint64_t virtual_handle) {
    fbb_.AddElement<uint64_t>(FBCublasSetStreamV2::VT_VIRTUAL_HANDLE, virtual_handle, 0);
  }
  void add_stream(uint64_t stream) {
    fbb_.AddElement<uint64_t>(FBCublasSetStreamV2::VT_STREAM, stream, 0);
  }
  explicit FBCublasSetStreamV2Builder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasSetStreamV2> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasSetStreamV2>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasSetStreamV2> CreateFBCublasSetStreamV2(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_handle = 0,
    uint64_t stream = 0) {
  FBCublasSetStreamV2Builder builder_(_fbb);
  builder_.add_stream(stream);
  builder_.add_virtual_handle(virtual_handle);
  return builder_.Finish();
}

struct FBCublasSetMathMode FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasSetMathModeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_HANDLE = 4,
    VT_MATH_MODE = 6
  };
  uint64_t virtual_handle() const {
    return GetField<uint64_t>(VT_VIRTUAL_HANDLE, 0);
  }
  uint64_t math_mode() const {
    return GetField<uint64_t>(VT_MATH_MODE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_HANDLE) &&
           VerifyField<uint64_t>(verifier, VT_MATH_MODE) &&
           verifier.EndTable();
  }
};

struct FBCublasSetMathModeBuilder {
  typedef FBCublasSetMathMode Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_handle(uint64_t virtual_handle) {
    fbb_.AddElement<uint64_t>(FBCublasSetMathMode::VT_VIRTUAL_HANDLE, virtual_handle, 0);
  }
  void add_math_mode(uint64_t math_mode) {
    fbb_.AddElement<uint64_t>(FBCublasSetMathMode::VT_MATH_MODE, math_mode, 0);
  }
  explicit FBCublasSetMathModeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasSetMathMode> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasSetMathMode>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasSetMathMode> CreateFBCublasSetMathMode(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_handle = 0,
    uint64_t math_mode = 0) {
  FBCublasSetMathModeBuilder builder_(_fbb);
  builder_.add_math_mode(math_mode);
  builder_.add_virtual_handle(virtual_handle);
  return builder_.Finish();
}

struct FBCublasSgemmV2 FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasSgemmV2Builder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_HANDLE = 4,
    VT_TRANSA_OP = 6,
    VT_TRANSB_OP = 8,
    VT_M = 10,
    VT_N = 12,
    VT_K = 14,
    VT_ALPHA = 16,
    VT_BETA = 18,
    VT_A = 20,
    VT_B = 22,
    VT_C = 24,
    VT_LDA = 26,
    VT_LDB = 28,
    VT_LDC = 30
  };
  uint64_t virtual_handle() const {
    return GetField<uint64_t>(VT_VIRTUAL_HANDLE, 0);
  }
  uint64_t transa_op() const {
    return GetField<uint64_t>(VT_TRANSA_OP, 0);
  }
  uint64_t transb_op() const {
    return GetField<uint64_t>(VT_TRANSB_OP, 0);
  }
  int32_t m() const {
    return GetField<int32_t>(VT_M, 0);
  }
  int32_t n() const {
    return GetField<int32_t>(VT_N, 0);
  }
  int32_t k() const {
    return GetField<int32_t>(VT_K, 0);
  }
  float alpha() const {
    return GetField<float>(VT_ALPHA, 0.0f);
  }
  float beta() const {
    return GetField<float>(VT_BETA, 0.0f);
  }
  uint64_t a() const {
    return GetField<uint64_t>(VT_A, 0);
  }
  uint64_t b() const {
    return GetField<uint64_t>(VT_B, 0);
  }
  uint64_t c() const {
    return GetField<uint64_t>(VT_C, 0);
  }
  int32_t lda() const {
    return GetField<int32_t>(VT_LDA, 0);
  }
  int32_t ldb() const {
    return GetField<int32_t>(VT_LDB, 0);
  }
  int32_t ldc() const {
    return GetField<int32_t>(VT_LDC, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_HANDLE) &&
           VerifyField<uint64_t>(verifier, VT_TRANSA_OP) &&
           VerifyField<uint64_t>(verifier, VT_TRANSB_OP) &&
           VerifyField<int32_t>(verifier, VT_M) &&
           VerifyField<int32_t>(verifier, VT_N) &&
           VerifyField<int32_t>(verifier, VT_K) &&
           VerifyField<float>(verifier, VT_ALPHA) &&
           VerifyField<float>(verifier, VT_BETA) &&
           VerifyField<uint64_t>(verifier, VT_A) &&
           VerifyField<uint64_t>(verifier, VT_B) &&
           VerifyField<uint64_t>(verifier, VT_C) &&
           VerifyField<int32_t>(verifier, VT_LDA) &&
           VerifyField<int32_t>(verifier, VT_LDB) &&
           VerifyField<int32_t>(verifier, VT_LDC) &&
           verifier.EndTable();
  }
};

struct FBCublasSgemmV2Builder {
  typedef FBCublasSgemmV2 Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_handle(uint64_t virtual_handle) {
    fbb_.AddElement<uint64_t>(FBCublasSgemmV2::VT_VIRTUAL_HANDLE, virtual_handle, 0);
  }
  void add_transa_op(uint64_t transa_op) {
    fbb_.AddElement<uint64_t>(FBCublasSgemmV2::VT_TRANSA_OP, transa_op, 0);
  }
  void add_transb_op(uint64_t transb_op) {
    fbb_.AddElement<uint64_t>(FBCublasSgemmV2::VT_TRANSB_OP, transb_op, 0);
  }
  void add_m(int32_t m) {
    fbb_.AddElement<int32_t>(FBCublasSgemmV2::VT_M, m, 0);
  }
  void add_n(int32_t n) {
    fbb_.AddElement<int32_t>(FBCublasSgemmV2::VT_N, n, 0);
  }
  void add_k(int32_t k) {
    fbb_.AddElement<int32_t>(FBCublasSgemmV2::VT_K, k, 0);
  }
  void add_alpha(float alpha) {
    fbb_.AddElement<float>(FBCublasSgemmV2::VT_ALPHA, alpha, 0.0f);
  }
  void add_beta(float beta) {
    fbb_.AddElement<float>(FBCublasSgemmV2::VT_BETA, beta, 0.0f);
  }
  void add_a(uint64_t a) {
    fbb_.AddElement<uint64_t>(FBCublasSgemmV2::VT_A, a, 0);
  }
  void add_b(uint64_t b) {
    fbb_.AddElement<uint64_t>(FBCublasSgemmV2::VT_B, b, 0);
  }
  void add_c(uint64_t c) {
    fbb_.AddElement<uint64_t>(FBCublasSgemmV2::VT_C, c, 0);
  }
  void add_lda(int32_t lda) {
    fbb_.AddElement<int32_t>(FBCublasSgemmV2::VT_LDA, lda, 0);
  }
  void add_ldb(int32_t ldb) {
    fbb_.AddElement<int32_t>(FBCublasSgemmV2::VT_LDB, ldb, 0);
  }
  void add_ldc(int32_t ldc) {
    fbb_.AddElement<int32_t>(FBCublasSgemmV2::VT_LDC, ldc, 0);
  }
  explicit FBCublasSgemmV2Builder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasSgemmV2> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasSgemmV2>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasSgemmV2> CreateFBCublasSgemmV2(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_handle = 0,
    uint64_t transa_op = 0,
    uint64_t transb_op = 0,
    int32_t m = 0,
    int32_t n = 0,
    int32_t k = 0,
    float alpha = 0.0f,
    float beta = 0.0f,
    uint64_t a = 0,
    uint64_t b = 0,
    uint64_t c = 0,
    int32_t lda = 0,
    int32_t ldb = 0,
    int32_t ldc = 0) {
  FBCublasSgemmV2Builder builder_(_fbb);
  builder_.add_c(c);
  builder_.add_b(b);
  builder_.add_a(a);
  builder_.add_transb_op(transb_op);
  builder_.add_transa_op(transa_op);
  builder_.add_virtual_handle(virtual_handle);
  builder_.add_ldc(ldc);
  builder_.add_ldb(ldb);
  builder_.add_lda(lda);
  builder_.add_beta(beta);
  builder_.add_alpha(alpha);
  builder_.add_k(k);
  builder_.add_n(n);
  builder_.add_m(m);
  return builder_.Finish();
}

struct FBCublasLtCreate FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtCreateBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_HANDLE = 4
  };
  uint64_t virtual_handle() const {
    return GetField<uint64_t>(VT_VIRTUAL_HANDLE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_HANDLE) &&
           verifier.EndTable();
  }
};

struct FBCublasLtCreateBuilder {
  typedef FBCublasLtCreate Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_handle(uint64_t virtual_handle) {
    fbb_.AddElement<uint64_t>(FBCublasLtCreate::VT_VIRTUAL_HANDLE, virtual_handle, 0);
  }
  explicit FBCublasLtCreateBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtCreate> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtCreate>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtCreate> CreateFBCublasLtCreate(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_handle = 0) {
  FBCublasLtCreateBuilder builder_(_fbb);
  builder_.add_virtual_handle(virtual_handle);
  return builder_.Finish();
}

struct FBCublasLtMatmulDescCreate FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtMatmulDescCreateBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_MMD = 4,
    VT_COMPUTE_TYPE = 6,
    VT_SCALE_TYPE = 8
  };
  uint64_t virtual_mmd() const {
    return GetField<uint64_t>(VT_VIRTUAL_MMD, 0);
  }
  uint64_t compute_type() const {
    return GetField<uint64_t>(VT_COMPUTE_TYPE, 0);
  }
  uint64_t scale_type() const {
    return GetField<uint64_t>(VT_SCALE_TYPE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_MMD) &&
           VerifyField<uint64_t>(verifier, VT_COMPUTE_TYPE) &&
           VerifyField<uint64_t>(verifier, VT_SCALE_TYPE) &&
           verifier.EndTable();
  }
};

struct FBCublasLtMatmulDescCreateBuilder {
  typedef FBCublasLtMatmulDescCreate Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_mmd(uint64_t virtual_mmd) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmulDescCreate::VT_VIRTUAL_MMD, virtual_mmd, 0);
  }
  void add_compute_type(uint64_t compute_type) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmulDescCreate::VT_COMPUTE_TYPE, compute_type, 0);
  }
  void add_scale_type(uint64_t scale_type) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmulDescCreate::VT_SCALE_TYPE, scale_type, 0);
  }
  explicit FBCublasLtMatmulDescCreateBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtMatmulDescCreate> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtMatmulDescCreate>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtMatmulDescCreate> CreateFBCublasLtMatmulDescCreate(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_mmd = 0,
    uint64_t compute_type = 0,
    uint64_t scale_type = 0) {
  FBCublasLtMatmulDescCreateBuilder builder_(_fbb);
  builder_.add_scale_type(scale_type);
  builder_.add_compute_type(compute_type);
  builder_.add_virtual_mmd(virtual_mmd);
  return builder_.Finish();
}

struct FBCublasLtMatmulDescDestroy FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtMatmulDescDestroyBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_MMD = 4
  };
  uint64_t virtual_mmd() const {
    return GetField<uint64_t>(VT_VIRTUAL_MMD, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_MMD) &&
           verifier.EndTable();
  }
};

struct FBCublasLtMatmulDescDestroyBuilder {
  typedef FBCublasLtMatmulDescDestroy Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_mmd(uint64_t virtual_mmd) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmulDescDestroy::VT_VIRTUAL_MMD, virtual_mmd, 0);
  }
  explicit FBCublasLtMatmulDescDestroyBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtMatmulDescDestroy> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtMatmulDescDestroy>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtMatmulDescDestroy> CreateFBCublasLtMatmulDescDestroy(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_mmd = 0) {
  FBCublasLtMatmulDescDestroyBuilder builder_(_fbb);
  builder_.add_virtual_mmd(virtual_mmd);
  return builder_.Finish();
}

struct FBCublasLtMatmulDescSetAttribute FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtMatmulDescSetAttributeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_MMD = 4,
    VT_ATTR = 6,
    VT_BUF = 8
  };
  uint64_t virtual_mmd() const {
    return GetField<uint64_t>(VT_VIRTUAL_MMD, 0);
  }
  uint64_t attr() const {
    return GetField<uint64_t>(VT_ATTR, 0);
  }
  const flatbuffers::Vector<uint8_t> *buf() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BUF);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_MMD) &&
           VerifyField<uint64_t>(verifier, VT_ATTR) &&
           VerifyOffset(verifier, VT_BUF) &&
           verifier.VerifyVector(buf()) &&
           verifier.EndTable();
  }
};

struct FBCublasLtMatmulDescSetAttributeBuilder {
  typedef FBCublasLtMatmulDescSetAttribute Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_mmd(uint64_t virtual_mmd) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmulDescSetAttribute::VT_VIRTUAL_MMD, virtual_mmd, 0);
  }
  void add_attr(uint64_t attr) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmulDescSetAttribute::VT_ATTR, attr, 0);
  }
  void add_buf(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buf) {
    fbb_.AddOffset(FBCublasLtMatmulDescSetAttribute::VT_BUF, buf);
  }
  explicit FBCublasLtMatmulDescSetAttributeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtMatmulDescSetAttribute> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtMatmulDescSetAttribute>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtMatmulDescSetAttribute> CreateFBCublasLtMatmulDescSetAttribute(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_mmd = 0,
    uint64_t attr = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buf = 0) {
  FBCublasLtMatmulDescSetAttributeBuilder builder_(_fbb);
  builder_.add_attr(attr);
  builder_.add_virtual_mmd(virtual_mmd);
  builder_.add_buf(buf);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCublasLtMatmulDescSetAttribute> CreateFBCublasLtMatmulDescSetAttributeDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_mmd = 0,
    uint64_t attr = 0,
    const std::vector<uint8_t> *buf = nullptr) {
  auto buf__ = buf ? _fbb.CreateVector<uint8_t>(*buf) : 0;
  return CreateFBCublasLtMatmulDescSetAttribute(
      _fbb,
      virtual_mmd,
      attr,
      buf__);
}

struct FBCublasLtMatmul FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtMatmulBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_HANDLE = 4,
    VT_VIRTUAL_MMD = 6,
    VT_ALPHA = 8,
    VT_BETA = 10,
    VT_A = 12,
    VT_B = 14,
    VT_C = 16,
    VT_D = 18,
    VT_VIRTUAL_ML_A_DESC = 20,
    VT_VIRTUAL_ML_B_DESC = 22,
    VT_VIRTUAL_ML_C_DESC = 24,
    VT_VIRTUAL_ML_D_DESC = 26,
    VT_ALGO = 28,
    VT_ALGO_IS_NULL = 30,
    VT_WORKSPACE = 32,
    VT_WORKSPACE_SIZE_IN_BYTES = 34,
    VT_STREAM = 36
  };
  uint64_t virtual_handle() const {
    return GetField<uint64_t>(VT_VIRTUAL_HANDLE, 0);
  }
  uint64_t virtual_mmd() const {
    return GetField<uint64_t>(VT_VIRTUAL_MMD, 0);
  }
  const flatbuffers::Vector<uint8_t> *alpha() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_ALPHA);
  }
  const flatbuffers::Vector<uint8_t> *beta() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BETA);
  }
  uint64_t a() const {
    return GetField<uint64_t>(VT_A, 0);
  }
  uint64_t b() const {
    return GetField<uint64_t>(VT_B, 0);
  }
  uint64_t c() const {
    return GetField<uint64_t>(VT_C, 0);
  }
  uint64_t d() const {
    return GetField<uint64_t>(VT_D, 0);
  }
  uint64_t virtual_ml_a_desc() const {
    return GetField<uint64_t>(VT_VIRTUAL_ML_A_DESC, 0);
  }
  uint64_t virtual_ml_b_desc() const {
    return GetField<uint64_t>(VT_VIRTUAL_ML_B_DESC, 0);
  }
  uint64_t virtual_ml_c_desc() const {
    return GetField<uint64_t>(VT_VIRTUAL_ML_C_DESC, 0);
  }
  uint64_t virtual_ml_d_desc() const {
    return GetField<uint64_t>(VT_VIRTUAL_ML_D_DESC, 0);
  }
  const flatbuffers::Vector<uint64_t> *algo() const {
    return GetPointer<const flatbuffers::Vector<uint64_t> *>(VT_ALGO);
  }
  bool algo_is_null() const {
    return GetField<uint8_t>(VT_ALGO_IS_NULL, 0) != 0;
  }
  uint64_t workspace() const {
    return GetField<uint64_t>(VT_WORKSPACE, 0);
  }
  uint64_t workspace_size_in_bytes() const {
    return GetField<uint64_t>(VT_WORKSPACE_SIZE_IN_BYTES, 0);
  }
  uint64_t stream() const {
    return GetField<uint64_t>(VT_STREAM, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_HANDLE) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_MMD) &&
           VerifyOffset(verifier, VT_ALPHA) &&
           verifier.VerifyVector(alpha()) &&
           VerifyOffset(verifier, VT_BETA) &&
           verifier.VerifyVector(beta()) &&
           VerifyField<uint64_t>(verifier, VT_A) &&
           VerifyField<uint64_t>(verifier, VT_B) &&
           VerifyField<uint64_t>(verifier, VT_C) &&
           VerifyField<uint64_t>(verifier, VT_D) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_ML_A_DESC) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_ML_B_DESC) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_ML_C_DESC) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_ML_D_DESC) &&
           VerifyOffset(verifier, VT_ALGO) &&
           verifier.VerifyVector(algo()) &&
           VerifyField<uint8_t>(verifier, VT_ALGO_IS_NULL) &&
           VerifyField<uint64_t>(verifier, VT_WORKSPACE) &&
           VerifyField<uint64_t>(verifier, VT_WORKSPACE_SIZE_IN_BYTES) &&
           VerifyField<uint64_t>(verifier, VT_STREAM) &&
           verifier.EndTable();
  }
};

struct FBCublasLtMatmulBuilder {
  typedef FBCublasLtMatmul Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_handle(uint64_t virtual_handle) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_VIRTUAL_HANDLE, virtual_handle, 0);
  }
  void add_virtual_mmd(uint64_t virtual_mmd) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_VIRTUAL_MMD, virtual_mmd, 0);
  }
  void add_alpha(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> alpha) {
    fbb_.AddOffset(FBCublasLtMatmul::VT_ALPHA, alpha);
  }
  void add_beta(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> beta) {
    fbb_.AddOffset(FBCublasLtMatmul::VT_BETA, beta);
  }
  void add_a(uint64_t a) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_A, a, 0);
  }
  void add_b(uint64_t b) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_B, b, 0);
  }
  void add_c(uint64_t c) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_C, c, 0);
  }
  void add_d(uint64_t d) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_D, d, 0);
  }
  void add_virtual_ml_a_desc(uint64_t virtual_ml_a_desc) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_VIRTUAL_ML_A_DESC, virtual_ml_a_desc, 0);
  }
  void add_virtual_ml_b_desc(uint64_t virtual_ml_b_desc) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_VIRTUAL_ML_B_DESC, virtual_ml_b_desc, 0);
  }
  void add_virtual_ml_c_desc(uint64_t virtual_ml_c_desc) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_VIRTUAL_ML_C_DESC, virtual_ml_c_desc, 0);
  }
  void add_virtual_ml_d_desc(uint64_t virtual_ml_d_desc) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_VIRTUAL_ML_D_DESC, virtual_ml_d_desc, 0);
  }
  void add_algo(flatbuffers::Offset<flatbuffers::Vector<uint64_t>> algo) {
    fbb_.AddOffset(FBCublasLtMatmul::VT_ALGO, algo);
  }
  void add_algo_is_null(bool algo_is_null) {
    fbb_.AddElement<uint8_t>(FBCublasLtMatmul::VT_ALGO_IS_NULL, static_cast<uint8_t>(algo_is_null), 0);
  }
  void add_workspace(uint64_t workspace) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_WORKSPACE, workspace, 0);
  }
  void add_workspace_size_in_bytes(uint64_t workspace_size_in_bytes) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_WORKSPACE_SIZE_IN_BYTES, workspace_size_in_bytes, 0);
  }
  void add_stream(uint64_t stream) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatmul::VT_STREAM, stream, 0);
  }
  explicit FBCublasLtMatmulBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtMatmul> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtMatmul>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtMatmul> CreateFBCublasLtMatmul(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_handle = 0,
    uint64_t virtual_mmd = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> alpha = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> beta = 0,
    uint64_t a = 0,
    uint64_t b = 0,
    uint64_t c = 0,
    uint64_t d = 0,
    uint64_t virtual_ml_a_desc = 0,
    uint64_t virtual_ml_b_desc = 0,
    uint64_t virtual_ml_c_desc = 0,
    uint64_t virtual_ml_d_desc = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint64_t>> algo = 0,
    bool algo_is_null = false,
    uint64_t workspace = 0,
    uint64_t workspace_size_in_bytes = 0,
    uint64_t stream = 0) {
  FBCublasLtMatmulBuilder builder_(_fbb);
  builder_.add_stream(stream);
  builder_.add_workspace_size_in_bytes(workspace_size_in_bytes);
  builder_.add_workspace(workspace);
  builder_.add_virtual_ml_d_desc(virtual_ml_d_desc);
  builder_.add_virtual_ml_c_desc(virtual_ml_c_desc);
  builder_.add_virtual_ml_b_desc(virtual_ml_b_desc);
  builder_.add_virtual_ml_a_desc(virtual_ml_a_desc);
  builder_.add_d(d);
  builder_.add_c(c);
  builder_.add_b(b);
  builder_.add_a(a);
  builder_.add_virtual_mmd(virtual_mmd);
  builder_.add_virtual_handle(virtual_handle);
  builder_.add_algo(algo);
  builder_.add_beta(beta);
  builder_.add_alpha(alpha);
  builder_.add_algo_is_null(algo_is_null);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCublasLtMatmul> CreateFBCublasLtMatmulDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_handle = 0,
    uint64_t virtual_mmd = 0,
    const std::vector<uint8_t> *alpha = nullptr,
    const std::vector<uint8_t> *beta = nullptr,
    uint64_t a = 0,
    uint64_t b = 0,
    uint64_t c = 0,
    uint64_t d = 0,
    uint64_t virtual_ml_a_desc = 0,
    uint64_t virtual_ml_b_desc = 0,
    uint64_t virtual_ml_c_desc = 0,
    uint64_t virtual_ml_d_desc = 0,
    const std::vector<uint64_t> *algo = nullptr,
    bool algo_is_null = false,
    uint64_t workspace = 0,
    uint64_t workspace_size_in_bytes = 0,
    uint64_t stream = 0) {
  auto alpha__ = alpha ? _fbb.CreateVector<uint8_t>(*alpha) : 0;
  auto beta__ = beta ? _fbb.CreateVector<uint8_t>(*beta) : 0;
  auto algo__ = algo ? _fbb.CreateVector<uint64_t>(*algo) : 0;
  return CreateFBCublasLtMatmul(
      _fbb,
      virtual_handle,
      virtual_mmd,
      alpha__,
      beta__,
      a,
      b,
      c,
      d,
      virtual_ml_a_desc,
      virtual_ml_b_desc,
      virtual_ml_c_desc,
      virtual_ml_d_desc,
      algo__,
      algo_is_null,
      workspace,
      workspace_size_in_bytes,
      stream);
}

struct FBCublasLtMatrixLayoutCreate FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtMatrixLayoutCreateBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_ML = 4,
    VT_DATA_TYPE = 6,
    VT_ROWS = 8,
    VT_COLS = 10,
    VT_LD = 12
  };
  uint64_t virtual_ml() const {
    return GetField<uint64_t>(VT_VIRTUAL_ML, 0);
  }
  uint64_t data_type() const {
    return GetField<uint64_t>(VT_DATA_TYPE, 0);
  }
  uint64_t rows() const {
    return GetField<uint64_t>(VT_ROWS, 0);
  }
  uint64_t cols() const {
    return GetField<uint64_t>(VT_COLS, 0);
  }
  int64_t ld() const {
    return GetField<int64_t>(VT_LD, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_ML) &&
           VerifyField<uint64_t>(verifier, VT_DATA_TYPE) &&
           VerifyField<uint64_t>(verifier, VT_ROWS) &&
           VerifyField<uint64_t>(verifier, VT_COLS) &&
           VerifyField<int64_t>(verifier, VT_LD) &&
           verifier.EndTable();
  }
};

struct FBCublasLtMatrixLayoutCreateBuilder {
  typedef FBCublasLtMatrixLayoutCreate Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_ml(uint64_t virtual_ml) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatrixLayoutCreate::VT_VIRTUAL_ML, virtual_ml, 0);
  }
  void add_data_type(uint64_t data_type) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatrixLayoutCreate::VT_DATA_TYPE, data_type, 0);
  }
  void add_rows(uint64_t rows) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatrixLayoutCreate::VT_ROWS, rows, 0);
  }
  void add_cols(uint64_t cols) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatrixLayoutCreate::VT_COLS, cols, 0);
  }
  void add_ld(int64_t ld) {
    fbb_.AddElement<int64_t>(FBCublasLtMatrixLayoutCreate::VT_LD, ld, 0);
  }
  explicit FBCublasLtMatrixLayoutCreateBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtMatrixLayoutCreate> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtMatrixLayoutCreate>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtMatrixLayoutCreate> CreateFBCublasLtMatrixLayoutCreate(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_ml = 0,
    uint64_t data_type = 0,
    uint64_t rows = 0,
    uint64_t cols = 0,
    int64_t ld = 0) {
  FBCublasLtMatrixLayoutCreateBuilder builder_(_fbb);
  builder_.add_ld(ld);
  builder_.add_cols(cols);
  builder_.add_rows(rows);
  builder_.add_data_type(data_type);
  builder_.add_virtual_ml(virtual_ml);
  return builder_.Finish();
}

struct FBCublasLtMatrixLayoutDestroy FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtMatrixLayoutDestroyBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_ML = 4
  };
  uint64_t virtual_ml() const {
    return GetField<uint64_t>(VT_VIRTUAL_ML, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_ML) &&
           verifier.EndTable();
  }
};

struct FBCublasLtMatrixLayoutDestroyBuilder {
  typedef FBCublasLtMatrixLayoutDestroy Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_ml(uint64_t virtual_ml) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatrixLayoutDestroy::VT_VIRTUAL_ML, virtual_ml, 0);
  }
  explicit FBCublasLtMatrixLayoutDestroyBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtMatrixLayoutDestroy> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtMatrixLayoutDestroy>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtMatrixLayoutDestroy> CreateFBCublasLtMatrixLayoutDestroy(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_ml = 0) {
  FBCublasLtMatrixLayoutDestroyBuilder builder_(_fbb);
  builder_.add_virtual_ml(virtual_ml);
  return builder_.Finish();
}

struct FBCublasLtMatrixLayoutSetAttribute FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCublasLtMatrixLayoutSetAttributeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_VIRTUAL_ML = 4,
    VT_ATTR = 6,
    VT_BUF = 8
  };
  uint64_t virtual_ml() const {
    return GetField<uint64_t>(VT_VIRTUAL_ML, 0);
  }
  uint64_t attr() const {
    return GetField<uint64_t>(VT_ATTR, 0);
  }
  const flatbuffers::Vector<uint8_t> *buf() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BUF);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_VIRTUAL_ML) &&
           VerifyField<uint64_t>(verifier, VT_ATTR) &&
           VerifyOffset(verifier, VT_BUF) &&
           verifier.VerifyVector(buf()) &&
           verifier.EndTable();
  }
};

struct FBCublasLtMatrixLayoutSetAttributeBuilder {
  typedef FBCublasLtMatrixLayoutSetAttribute Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_virtual_ml(uint64_t virtual_ml) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatrixLayoutSetAttribute::VT_VIRTUAL_ML, virtual_ml, 0);
  }
  void add_attr(uint64_t attr) {
    fbb_.AddElement<uint64_t>(FBCublasLtMatrixLayoutSetAttribute::VT_ATTR, attr, 0);
  }
  void add_buf(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buf) {
    fbb_.AddOffset(FBCublasLtMatrixLayoutSetAttribute::VT_BUF, buf);
  }
  explicit FBCublasLtMatrixLayoutSetAttributeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCublasLtMatrixLayoutSetAttribute> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCublasLtMatrixLayoutSetAttribute>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCublasLtMatrixLayoutSetAttribute> CreateFBCublasLtMatrixLayoutSetAttribute(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_ml = 0,
    uint64_t attr = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buf = 0) {
  FBCublasLtMatrixLayoutSetAttributeBuilder builder_(_fbb);
  builder_.add_attr(attr);
  builder_.add_virtual_ml(virtual_ml);
  builder_.add_buf(buf);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCublasLtMatrixLayoutSetAttribute> CreateFBCublasLtMatrixLayoutSetAttributeDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t virtual_ml = 0,
    uint64_t attr = 0,
    const std::vector<uint8_t> *buf = nullptr) {
  auto buf__ = buf ? _fbb.CreateVector<uint8_t>(*buf) : 0;
  return CreateFBCublasLtMatrixLayoutSetAttribute(
      _fbb,
      virtual_ml,
      attr,
      buf__);
}

#endif  // FLATBUFFERS_GENERATED_CUBLASCALLS_H_
