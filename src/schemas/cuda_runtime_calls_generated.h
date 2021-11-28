// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_CUDARUNTIMECALLS_H_
#define FLATBUFFERS_GENERATED_CUDARUNTIMECALLS_H_

#include "flatbuffers/flatbuffers.h"

struct FBParamBuffer;
struct FBParamBufferBuilder;

struct FBParamInfo;
struct FBParamInfoBuilder;

struct FBDim3;

struct FBCudaMalloc;
struct FBCudaMallocBuilder;

struct FBCudaMemcpyH2D;
struct FBCudaMemcpyH2DBuilder;

struct FBCudaMemcpyD2H;
struct FBCudaMemcpyD2HBuilder;

struct FBCudaMemcpyD2D;
struct FBCudaMemcpyD2DBuilder;

struct FBCudaMemcpyAsyncH2D;
struct FBCudaMemcpyAsyncH2DBuilder;

struct FBCudaMemcpyAsyncD2H;
struct FBCudaMemcpyAsyncD2HBuilder;

struct FBCudaMemcpyAsyncD2D;
struct FBCudaMemcpyAsyncD2DBuilder;

struct FBCudaLaunchKernel;
struct FBCudaLaunchKernelBuilder;

struct FBCudaFree;
struct FBCudaFreeBuilder;

struct FBCudaStreamSynchronize;
struct FBCudaStreamSynchronizeBuilder;

struct FBCudaGetDeviceProperties;
struct FBCudaGetDevicePropertiesBuilder;

enum FBPtxParameterType : int8_t {
  FBPtxParameterType_s8 = 0,
  FBPtxParameterType_s16 = 1,
  FBPtxParameterType_s32 = 2,
  FBPtxParameterType_s64 = 3,
  FBPtxParameterType_u8 = 4,
  FBPtxParameterType_u16 = 5,
  FBPtxParameterType_u32 = 6,
  FBPtxParameterType_u64 = 7,
  FBPtxParameterType_f16 = 8,
  FBPtxParameterType_f16x2 = 9,
  FBPtxParameterType_f32 = 10,
  FBPtxParameterType_f64 = 11,
  FBPtxParameterType_b8 = 12,
  FBPtxParameterType_b16 = 13,
  FBPtxParameterType_b32 = 14,
  FBPtxParameterType_b64 = 15,
  FBPtxParameterType_pred = 16,
  FBPtxParameterType_invalid = 17,
  FBPtxParameterType_MIN = FBPtxParameterType_s8,
  FBPtxParameterType_MAX = FBPtxParameterType_invalid
};

inline const FBPtxParameterType (&EnumValuesFBPtxParameterType())[18] {
  static const FBPtxParameterType values[] = {
    FBPtxParameterType_s8,
    FBPtxParameterType_s16,
    FBPtxParameterType_s32,
    FBPtxParameterType_s64,
    FBPtxParameterType_u8,
    FBPtxParameterType_u16,
    FBPtxParameterType_u32,
    FBPtxParameterType_u64,
    FBPtxParameterType_f16,
    FBPtxParameterType_f16x2,
    FBPtxParameterType_f32,
    FBPtxParameterType_f64,
    FBPtxParameterType_b8,
    FBPtxParameterType_b16,
    FBPtxParameterType_b32,
    FBPtxParameterType_b64,
    FBPtxParameterType_pred,
    FBPtxParameterType_invalid
  };
  return values;
}

inline const char * const *EnumNamesFBPtxParameterType() {
  static const char * const names[19] = {
    "s8",
    "s16",
    "s32",
    "s64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f16",
    "f16x2",
    "f32",
    "f64",
    "b8",
    "b16",
    "b32",
    "b64",
    "pred",
    "invalid",
    nullptr
  };
  return names;
}

inline const char *EnumNameFBPtxParameterType(FBPtxParameterType e) {
  if (flatbuffers::IsOutRange(e, FBPtxParameterType_s8, FBPtxParameterType_invalid)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesFBPtxParameterType()[index];
}

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) FBDim3 FLATBUFFERS_FINAL_CLASS {
 private:
  uint64_t x_;
  uint64_t y_;
  uint64_t z_;

 public:
  FBDim3()
      : x_(0),
        y_(0),
        z_(0) {
  }
  FBDim3(uint64_t _x, uint64_t _y, uint64_t _z)
      : x_(flatbuffers::EndianScalar(_x)),
        y_(flatbuffers::EndianScalar(_y)),
        z_(flatbuffers::EndianScalar(_z)) {
  }
  uint64_t x() const {
    return flatbuffers::EndianScalar(x_);
  }
  uint64_t y() const {
    return flatbuffers::EndianScalar(y_);
  }
  uint64_t z() const {
    return flatbuffers::EndianScalar(z_);
  }
};
FLATBUFFERS_STRUCT_END(FBDim3, 24);

struct FBParamBuffer FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBParamBufferBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_BUFFER = 4
  };
  const flatbuffers::Vector<uint8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BUFFER);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           verifier.EndTable();
  }
};

struct FBParamBufferBuilder {
  typedef FBParamBuffer Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer) {
    fbb_.AddOffset(FBParamBuffer::VT_BUFFER, buffer);
  }
  explicit FBParamBufferBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBParamBuffer> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBParamBuffer>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBParamBuffer> CreateFBParamBuffer(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer = 0) {
  FBParamBufferBuilder builder_(_fbb);
  builder_.add_buffer(buffer);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBParamBuffer> CreateFBParamBufferDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint8_t> *buffer = nullptr) {
  auto buffer__ = buffer ? _fbb.CreateVector<uint8_t>(*buffer) : 0;
  return CreateFBParamBuffer(
      _fbb,
      buffer__);
}

struct FBParamInfo FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBParamInfoBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_PTX_PARAM_TYPE = 6,
    VT_TYPE_SIZE = 8,
    VT_ALIGN = 10,
    VT_SIZE = 12
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  FBPtxParameterType ptx_param_type() const {
    return static_cast<FBPtxParameterType>(GetField<int8_t>(VT_PTX_PARAM_TYPE, 0));
  }
  uint64_t type_size() const {
    return GetField<uint64_t>(VT_TYPE_SIZE, 0);
  }
  uint64_t align() const {
    return GetField<uint64_t>(VT_ALIGN, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyField<int8_t>(verifier, VT_PTX_PARAM_TYPE) &&
           VerifyField<uint64_t>(verifier, VT_TYPE_SIZE) &&
           VerifyField<uint64_t>(verifier, VT_ALIGN) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           verifier.EndTable();
  }
};

struct FBParamInfoBuilder {
  typedef FBParamInfo Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(FBParamInfo::VT_NAME, name);
  }
  void add_ptx_param_type(FBPtxParameterType ptx_param_type) {
    fbb_.AddElement<int8_t>(FBParamInfo::VT_PTX_PARAM_TYPE, static_cast<int8_t>(ptx_param_type), 0);
  }
  void add_type_size(uint64_t type_size) {
    fbb_.AddElement<uint64_t>(FBParamInfo::VT_TYPE_SIZE, type_size, 0);
  }
  void add_align(uint64_t align) {
    fbb_.AddElement<uint64_t>(FBParamInfo::VT_ALIGN, align, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBParamInfo::VT_SIZE, size, 0);
  }
  explicit FBParamInfoBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBParamInfo> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBParamInfo>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBParamInfo> CreateFBParamInfo(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    FBPtxParameterType ptx_param_type = FBPtxParameterType_s8,
    uint64_t type_size = 0,
    uint64_t align = 0,
    uint64_t size = 0) {
  FBParamInfoBuilder builder_(_fbb);
  builder_.add_size(size);
  builder_.add_align(align);
  builder_.add_type_size(type_size);
  builder_.add_name(name);
  builder_.add_ptx_param_type(ptx_param_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBParamInfo> CreateFBParamInfoDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    FBPtxParameterType ptx_param_type = FBPtxParameterType_s8,
    uint64_t type_size = 0,
    uint64_t align = 0,
    uint64_t size = 0) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  return CreateFBParamInfo(
      _fbb,
      name__,
      ptx_param_type,
      type_size,
      align,
      size);
}

struct FBCudaMalloc FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaMallocBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DEV_PTR = 4,
    VT_SIZE = 6
  };
  uint64_t dev_ptr() const {
    return GetField<uint64_t>(VT_DEV_PTR, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DEV_PTR) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           verifier.EndTable();
  }
};

struct FBCudaMallocBuilder {
  typedef FBCudaMalloc Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dev_ptr(uint64_t dev_ptr) {
    fbb_.AddElement<uint64_t>(FBCudaMalloc::VT_DEV_PTR, dev_ptr, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBCudaMalloc::VT_SIZE, size, 0);
  }
  explicit FBCudaMallocBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaMalloc> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaMalloc>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaMalloc> CreateFBCudaMalloc(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dev_ptr = 0,
    uint64_t size = 0) {
  FBCudaMallocBuilder builder_(_fbb);
  builder_.add_size(size);
  builder_.add_dev_ptr(dev_ptr);
  return builder_.Finish();
}

struct FBCudaMemcpyH2D FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaMemcpyH2DBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DST = 4,
    VT_SRC = 6,
    VT_SIZE = 8,
    VT_BUFFER = 10
  };
  uint64_t dst() const {
    return GetField<uint64_t>(VT_DST, 0);
  }
  uint64_t src() const {
    return GetField<uint64_t>(VT_SRC, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  const flatbuffers::Vector<uint8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BUFFER);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DST) &&
           VerifyField<uint64_t>(verifier, VT_SRC) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           verifier.EndTable();
  }
};

struct FBCudaMemcpyH2DBuilder {
  typedef FBCudaMemcpyH2D Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dst(uint64_t dst) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyH2D::VT_DST, dst, 0);
  }
  void add_src(uint64_t src) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyH2D::VT_SRC, src, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyH2D::VT_SIZE, size, 0);
  }
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer) {
    fbb_.AddOffset(FBCudaMemcpyH2D::VT_BUFFER, buffer);
  }
  explicit FBCudaMemcpyH2DBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaMemcpyH2D> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaMemcpyH2D>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaMemcpyH2D> CreateFBCudaMemcpyH2D(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer = 0) {
  FBCudaMemcpyH2DBuilder builder_(_fbb);
  builder_.add_size(size);
  builder_.add_src(src);
  builder_.add_dst(dst);
  builder_.add_buffer(buffer);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCudaMemcpyH2D> CreateFBCudaMemcpyH2DDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    const std::vector<uint8_t> *buffer = nullptr) {
  auto buffer__ = buffer ? _fbb.CreateVector<uint8_t>(*buffer) : 0;
  return CreateFBCudaMemcpyH2D(
      _fbb,
      dst,
      src,
      size,
      buffer__);
}

struct FBCudaMemcpyD2H FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaMemcpyD2HBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DST = 4,
    VT_SRC = 6,
    VT_SIZE = 8,
    VT_BUFFER = 10
  };
  uint64_t dst() const {
    return GetField<uint64_t>(VT_DST, 0);
  }
  uint64_t src() const {
    return GetField<uint64_t>(VT_SRC, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  const flatbuffers::Vector<uint8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BUFFER);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DST) &&
           VerifyField<uint64_t>(verifier, VT_SRC) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           verifier.EndTable();
  }
};

struct FBCudaMemcpyD2HBuilder {
  typedef FBCudaMemcpyD2H Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dst(uint64_t dst) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyD2H::VT_DST, dst, 0);
  }
  void add_src(uint64_t src) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyD2H::VT_SRC, src, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyD2H::VT_SIZE, size, 0);
  }
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer) {
    fbb_.AddOffset(FBCudaMemcpyD2H::VT_BUFFER, buffer);
  }
  explicit FBCudaMemcpyD2HBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaMemcpyD2H> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaMemcpyD2H>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaMemcpyD2H> CreateFBCudaMemcpyD2H(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer = 0) {
  FBCudaMemcpyD2HBuilder builder_(_fbb);
  builder_.add_size(size);
  builder_.add_src(src);
  builder_.add_dst(dst);
  builder_.add_buffer(buffer);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCudaMemcpyD2H> CreateFBCudaMemcpyD2HDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    const std::vector<uint8_t> *buffer = nullptr) {
  auto buffer__ = buffer ? _fbb.CreateVector<uint8_t>(*buffer) : 0;
  return CreateFBCudaMemcpyD2H(
      _fbb,
      dst,
      src,
      size,
      buffer__);
}

struct FBCudaMemcpyD2D FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaMemcpyD2DBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DST = 4,
    VT_SRC = 6,
    VT_SIZE = 8
  };
  uint64_t dst() const {
    return GetField<uint64_t>(VT_DST, 0);
  }
  uint64_t src() const {
    return GetField<uint64_t>(VT_SRC, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DST) &&
           VerifyField<uint64_t>(verifier, VT_SRC) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           verifier.EndTable();
  }
};

struct FBCudaMemcpyD2DBuilder {
  typedef FBCudaMemcpyD2D Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dst(uint64_t dst) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyD2D::VT_DST, dst, 0);
  }
  void add_src(uint64_t src) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyD2D::VT_SRC, src, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyD2D::VT_SIZE, size, 0);
  }
  explicit FBCudaMemcpyD2DBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaMemcpyD2D> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaMemcpyD2D>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaMemcpyD2D> CreateFBCudaMemcpyD2D(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0) {
  FBCudaMemcpyD2DBuilder builder_(_fbb);
  builder_.add_size(size);
  builder_.add_src(src);
  builder_.add_dst(dst);
  return builder_.Finish();
}

struct FBCudaMemcpyAsyncH2D FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaMemcpyAsyncH2DBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DST = 4,
    VT_SRC = 6,
    VT_SIZE = 8,
    VT_STREAM = 10,
    VT_BUFFER = 12
  };
  uint64_t dst() const {
    return GetField<uint64_t>(VT_DST, 0);
  }
  uint64_t src() const {
    return GetField<uint64_t>(VT_SRC, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  uint64_t stream() const {
    return GetField<uint64_t>(VT_STREAM, 0);
  }
  const flatbuffers::Vector<uint8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BUFFER);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DST) &&
           VerifyField<uint64_t>(verifier, VT_SRC) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           VerifyField<uint64_t>(verifier, VT_STREAM) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           verifier.EndTable();
  }
};

struct FBCudaMemcpyAsyncH2DBuilder {
  typedef FBCudaMemcpyAsyncH2D Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dst(uint64_t dst) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncH2D::VT_DST, dst, 0);
  }
  void add_src(uint64_t src) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncH2D::VT_SRC, src, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncH2D::VT_SIZE, size, 0);
  }
  void add_stream(uint64_t stream) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncH2D::VT_STREAM, stream, 0);
  }
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer) {
    fbb_.AddOffset(FBCudaMemcpyAsyncH2D::VT_BUFFER, buffer);
  }
  explicit FBCudaMemcpyAsyncH2DBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaMemcpyAsyncH2D> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaMemcpyAsyncH2D>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaMemcpyAsyncH2D> CreateFBCudaMemcpyAsyncH2D(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    uint64_t stream = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer = 0) {
  FBCudaMemcpyAsyncH2DBuilder builder_(_fbb);
  builder_.add_stream(stream);
  builder_.add_size(size);
  builder_.add_src(src);
  builder_.add_dst(dst);
  builder_.add_buffer(buffer);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCudaMemcpyAsyncH2D> CreateFBCudaMemcpyAsyncH2DDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    uint64_t stream = 0,
    const std::vector<uint8_t> *buffer = nullptr) {
  auto buffer__ = buffer ? _fbb.CreateVector<uint8_t>(*buffer) : 0;
  return CreateFBCudaMemcpyAsyncH2D(
      _fbb,
      dst,
      src,
      size,
      stream,
      buffer__);
}

struct FBCudaMemcpyAsyncD2H FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaMemcpyAsyncD2HBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DST = 4,
    VT_SRC = 6,
    VT_SIZE = 8,
    VT_STREAM = 10,
    VT_BUFFER = 12
  };
  uint64_t dst() const {
    return GetField<uint64_t>(VT_DST, 0);
  }
  uint64_t src() const {
    return GetField<uint64_t>(VT_SRC, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  uint64_t stream() const {
    return GetField<uint64_t>(VT_STREAM, 0);
  }
  const flatbuffers::Vector<uint8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_BUFFER);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DST) &&
           VerifyField<uint64_t>(verifier, VT_SRC) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           VerifyField<uint64_t>(verifier, VT_STREAM) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           verifier.EndTable();
  }
};

struct FBCudaMemcpyAsyncD2HBuilder {
  typedef FBCudaMemcpyAsyncD2H Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dst(uint64_t dst) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2H::VT_DST, dst, 0);
  }
  void add_src(uint64_t src) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2H::VT_SRC, src, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2H::VT_SIZE, size, 0);
  }
  void add_stream(uint64_t stream) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2H::VT_STREAM, stream, 0);
  }
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer) {
    fbb_.AddOffset(FBCudaMemcpyAsyncD2H::VT_BUFFER, buffer);
  }
  explicit FBCudaMemcpyAsyncD2HBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaMemcpyAsyncD2H> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaMemcpyAsyncD2H>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaMemcpyAsyncD2H> CreateFBCudaMemcpyAsyncD2H(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    uint64_t stream = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer = 0) {
  FBCudaMemcpyAsyncD2HBuilder builder_(_fbb);
  builder_.add_stream(stream);
  builder_.add_size(size);
  builder_.add_src(src);
  builder_.add_dst(dst);
  builder_.add_buffer(buffer);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCudaMemcpyAsyncD2H> CreateFBCudaMemcpyAsyncD2HDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    uint64_t stream = 0,
    const std::vector<uint8_t> *buffer = nullptr) {
  auto buffer__ = buffer ? _fbb.CreateVector<uint8_t>(*buffer) : 0;
  return CreateFBCudaMemcpyAsyncD2H(
      _fbb,
      dst,
      src,
      size,
      stream,
      buffer__);
}

struct FBCudaMemcpyAsyncD2D FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaMemcpyAsyncD2DBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DST = 4,
    VT_SRC = 6,
    VT_SIZE = 8,
    VT_STREAM = 10
  };
  uint64_t dst() const {
    return GetField<uint64_t>(VT_DST, 0);
  }
  uint64_t src() const {
    return GetField<uint64_t>(VT_SRC, 0);
  }
  uint64_t size() const {
    return GetField<uint64_t>(VT_SIZE, 0);
  }
  uint64_t stream() const {
    return GetField<uint64_t>(VT_STREAM, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DST) &&
           VerifyField<uint64_t>(verifier, VT_SRC) &&
           VerifyField<uint64_t>(verifier, VT_SIZE) &&
           VerifyField<uint64_t>(verifier, VT_STREAM) &&
           verifier.EndTable();
  }
};

struct FBCudaMemcpyAsyncD2DBuilder {
  typedef FBCudaMemcpyAsyncD2D Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dst(uint64_t dst) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2D::VT_DST, dst, 0);
  }
  void add_src(uint64_t src) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2D::VT_SRC, src, 0);
  }
  void add_size(uint64_t size) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2D::VT_SIZE, size, 0);
  }
  void add_stream(uint64_t stream) {
    fbb_.AddElement<uint64_t>(FBCudaMemcpyAsyncD2D::VT_STREAM, stream, 0);
  }
  explicit FBCudaMemcpyAsyncD2DBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaMemcpyAsyncD2D> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaMemcpyAsyncD2D>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaMemcpyAsyncD2D> CreateFBCudaMemcpyAsyncD2D(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dst = 0,
    uint64_t src = 0,
    uint64_t size = 0,
    uint64_t stream = 0) {
  FBCudaMemcpyAsyncD2DBuilder builder_(_fbb);
  builder_.add_stream(stream);
  builder_.add_size(size);
  builder_.add_src(src);
  builder_.add_dst(dst);
  return builder_.Finish();
}

struct FBCudaLaunchKernel FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaLaunchKernelBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_SYMBOL = 4,
    VT_REQUIRED_MODULES = 6,
    VT_REQUIRED_FUNCTION_SYMBOLS = 8,
    VT_GRID_DIM = 10,
    VT_BLOCK_DIM = 12,
    VT_SHARED_MEM = 14,
    VT_STREAM = 16,
    VT_PARAM_BUFFERS = 18,
    VT_PARAM_INFOS = 20
  };
  const flatbuffers::String *symbol() const {
    return GetPointer<const flatbuffers::String *>(VT_SYMBOL);
  }
  const flatbuffers::Vector<uint64_t> *required_modules() const {
    return GetPointer<const flatbuffers::Vector<uint64_t> *>(VT_REQUIRED_MODULES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *required_function_symbols() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_REQUIRED_FUNCTION_SYMBOLS);
  }
  const FBDim3 *grid_dim() const {
    return GetStruct<const FBDim3 *>(VT_GRID_DIM);
  }
  const FBDim3 *block_dim() const {
    return GetStruct<const FBDim3 *>(VT_BLOCK_DIM);
  }
  uint64_t shared_mem() const {
    return GetField<uint64_t>(VT_SHARED_MEM, 0);
  }
  uint64_t stream() const {
    return GetField<uint64_t>(VT_STREAM, 0);
  }
  const flatbuffers::Vector<flatbuffers::Offset<FBParamBuffer>> *param_buffers() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<FBParamBuffer>> *>(VT_PARAM_BUFFERS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<FBParamInfo>> *param_infos() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<FBParamInfo>> *>(VT_PARAM_INFOS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_SYMBOL) &&
           verifier.VerifyString(symbol()) &&
           VerifyOffset(verifier, VT_REQUIRED_MODULES) &&
           verifier.VerifyVector(required_modules()) &&
           VerifyOffset(verifier, VT_REQUIRED_FUNCTION_SYMBOLS) &&
           verifier.VerifyVector(required_function_symbols()) &&
           verifier.VerifyVectorOfStrings(required_function_symbols()) &&
           VerifyField<FBDim3>(verifier, VT_GRID_DIM) &&
           VerifyField<FBDim3>(verifier, VT_BLOCK_DIM) &&
           VerifyField<uint64_t>(verifier, VT_SHARED_MEM) &&
           VerifyField<uint64_t>(verifier, VT_STREAM) &&
           VerifyOffset(verifier, VT_PARAM_BUFFERS) &&
           verifier.VerifyVector(param_buffers()) &&
           verifier.VerifyVectorOfTables(param_buffers()) &&
           VerifyOffset(verifier, VT_PARAM_INFOS) &&
           verifier.VerifyVector(param_infos()) &&
           verifier.VerifyVectorOfTables(param_infos()) &&
           verifier.EndTable();
  }
};

struct FBCudaLaunchKernelBuilder {
  typedef FBCudaLaunchKernel Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_symbol(flatbuffers::Offset<flatbuffers::String> symbol) {
    fbb_.AddOffset(FBCudaLaunchKernel::VT_SYMBOL, symbol);
  }
  void add_required_modules(flatbuffers::Offset<flatbuffers::Vector<uint64_t>> required_modules) {
    fbb_.AddOffset(FBCudaLaunchKernel::VT_REQUIRED_MODULES, required_modules);
  }
  void add_required_function_symbols(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> required_function_symbols) {
    fbb_.AddOffset(FBCudaLaunchKernel::VT_REQUIRED_FUNCTION_SYMBOLS, required_function_symbols);
  }
  void add_grid_dim(const FBDim3 *grid_dim) {
    fbb_.AddStruct(FBCudaLaunchKernel::VT_GRID_DIM, grid_dim);
  }
  void add_block_dim(const FBDim3 *block_dim) {
    fbb_.AddStruct(FBCudaLaunchKernel::VT_BLOCK_DIM, block_dim);
  }
  void add_shared_mem(uint64_t shared_mem) {
    fbb_.AddElement<uint64_t>(FBCudaLaunchKernel::VT_SHARED_MEM, shared_mem, 0);
  }
  void add_stream(uint64_t stream) {
    fbb_.AddElement<uint64_t>(FBCudaLaunchKernel::VT_STREAM, stream, 0);
  }
  void add_param_buffers(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FBParamBuffer>>> param_buffers) {
    fbb_.AddOffset(FBCudaLaunchKernel::VT_PARAM_BUFFERS, param_buffers);
  }
  void add_param_infos(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FBParamInfo>>> param_infos) {
    fbb_.AddOffset(FBCudaLaunchKernel::VT_PARAM_INFOS, param_infos);
  }
  explicit FBCudaLaunchKernelBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaLaunchKernel> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaLaunchKernel>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaLaunchKernel> CreateFBCudaLaunchKernel(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> symbol = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint64_t>> required_modules = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> required_function_symbols = 0,
    const FBDim3 *grid_dim = 0,
    const FBDim3 *block_dim = 0,
    uint64_t shared_mem = 0,
    uint64_t stream = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FBParamBuffer>>> param_buffers = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FBParamInfo>>> param_infos = 0) {
  FBCudaLaunchKernelBuilder builder_(_fbb);
  builder_.add_stream(stream);
  builder_.add_shared_mem(shared_mem);
  builder_.add_param_infos(param_infos);
  builder_.add_param_buffers(param_buffers);
  builder_.add_block_dim(block_dim);
  builder_.add_grid_dim(grid_dim);
  builder_.add_required_function_symbols(required_function_symbols);
  builder_.add_required_modules(required_modules);
  builder_.add_symbol(symbol);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCudaLaunchKernel> CreateFBCudaLaunchKernelDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *symbol = nullptr,
    const std::vector<uint64_t> *required_modules = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *required_function_symbols = nullptr,
    const FBDim3 *grid_dim = 0,
    const FBDim3 *block_dim = 0,
    uint64_t shared_mem = 0,
    uint64_t stream = 0,
    const std::vector<flatbuffers::Offset<FBParamBuffer>> *param_buffers = nullptr,
    const std::vector<flatbuffers::Offset<FBParamInfo>> *param_infos = nullptr) {
  auto symbol__ = symbol ? _fbb.CreateString(symbol) : 0;
  auto required_modules__ = required_modules ? _fbb.CreateVector<uint64_t>(*required_modules) : 0;
  auto required_function_symbols__ = required_function_symbols ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*required_function_symbols) : 0;
  auto param_buffers__ = param_buffers ? _fbb.CreateVector<flatbuffers::Offset<FBParamBuffer>>(*param_buffers) : 0;
  auto param_infos__ = param_infos ? _fbb.CreateVector<flatbuffers::Offset<FBParamInfo>>(*param_infos) : 0;
  return CreateFBCudaLaunchKernel(
      _fbb,
      symbol__,
      required_modules__,
      required_function_symbols__,
      grid_dim,
      block_dim,
      shared_mem,
      stream,
      param_buffers__,
      param_infos__);
}

struct FBCudaFree FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaFreeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DEV_PTR = 4
  };
  uint64_t dev_ptr() const {
    return GetField<uint64_t>(VT_DEV_PTR, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DEV_PTR) &&
           verifier.EndTable();
  }
};

struct FBCudaFreeBuilder {
  typedef FBCudaFree Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dev_ptr(uint64_t dev_ptr) {
    fbb_.AddElement<uint64_t>(FBCudaFree::VT_DEV_PTR, dev_ptr, 0);
  }
  explicit FBCudaFreeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaFree> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaFree>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaFree> CreateFBCudaFree(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t dev_ptr = 0) {
  FBCudaFreeBuilder builder_(_fbb);
  builder_.add_dev_ptr(dev_ptr);
  return builder_.Finish();
}

struct FBCudaStreamSynchronize FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaStreamSynchronizeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_STREAM = 4
  };
  uint64_t stream() const {
    return GetField<uint64_t>(VT_STREAM, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_STREAM) &&
           verifier.EndTable();
  }
};

struct FBCudaStreamSynchronizeBuilder {
  typedef FBCudaStreamSynchronize Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_stream(uint64_t stream) {
    fbb_.AddElement<uint64_t>(FBCudaStreamSynchronize::VT_STREAM, stream, 0);
  }
  explicit FBCudaStreamSynchronizeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaStreamSynchronize> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaStreamSynchronize>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaStreamSynchronize> CreateFBCudaStreamSynchronize(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t stream = 0) {
  FBCudaStreamSynchronizeBuilder builder_(_fbb);
  builder_.add_stream(stream);
  return builder_.Finish();
}

struct FBCudaGetDeviceProperties FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FBCudaGetDevicePropertiesBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_PROPERTIES_DATA = 4
  };
  const flatbuffers::Vector<uint8_t> *properties_data() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_PROPERTIES_DATA);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_PROPERTIES_DATA) &&
           verifier.VerifyVector(properties_data()) &&
           verifier.EndTable();
  }
};

struct FBCudaGetDevicePropertiesBuilder {
  typedef FBCudaGetDeviceProperties Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_properties_data(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> properties_data) {
    fbb_.AddOffset(FBCudaGetDeviceProperties::VT_PROPERTIES_DATA, properties_data);
  }
  explicit FBCudaGetDevicePropertiesBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  flatbuffers::Offset<FBCudaGetDeviceProperties> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FBCudaGetDeviceProperties>(end);
    return o;
  }
};

inline flatbuffers::Offset<FBCudaGetDeviceProperties> CreateFBCudaGetDeviceProperties(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> properties_data = 0) {
  FBCudaGetDevicePropertiesBuilder builder_(_fbb);
  builder_.add_properties_data(properties_data);
  return builder_.Finish();
}

inline flatbuffers::Offset<FBCudaGetDeviceProperties> CreateFBCudaGetDevicePropertiesDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint8_t> *properties_data = nullptr) {
  auto properties_data__ = properties_data ? _fbb.CreateVector<uint8_t>(*properties_data) : 0;
  return CreateFBCudaGetDeviceProperties(
      _fbb,
      properties_data__);
}

#endif  // FLATBUFFERS_GENERATED_CUDARUNTIMECALLS_H_
