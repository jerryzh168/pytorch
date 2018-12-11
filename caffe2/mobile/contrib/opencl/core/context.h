#ifndef CAFFE2_OPENCL_CONTEXT_H_
#define CAFFE2_OPENCL_CONTEXT_H_

#ifdef CAFFE2_OPENCL_BACKEND
#error Can only build one OpenCL backend at a time.
#else
#define CAFFE2_OPENCL_BACKEND
#endif

#include "caffe2/core/allocator.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/timer.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTuner.h"

#include "arm_compute/core/Types.h"
#include "utils/Utils.h"
#include "include/half/half.hpp"

#include "arm_compute/runtime/CPP/CPPFunctions.h"

namespace caffe2 {

typedef half_float::half half;
//#define ACL_USE_FLOAT32
#ifdef ACL_USE_FLOAT32
 typedef float DataType;
#else
 typedef half DataType;
#endif

template <typename T> class OpenCLTensor;

// TODO: inherit from BaseContext
class CLContext final {
public:
  static bool initialized;
  explicit CLContext();
  explicit CLContext(const DeviceOption &option) {
    DCHECK_EQ(option.device_type(), PROTO_OPENCL);
    CLContext();
  }
  explicit CLContext(const at::Device& device)
      : CLContext(DeviceToOption(device)) {}
  ~CLContext() {}

  static void sync() { arm_compute::CLScheduler::get().sync(); }

  template <typename T>
  using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;
  template <typename T>
  static deleted_unique_ptr<const OpenCLTensor<T>> getCLTensor(const Blob *b, const OpenCLTensor<T>* X_old = nullptr) {
    if (BlobIsTensorType(*b, CPU)) {

      auto &Xcpu = b->Get<TensorCPU>();
      OpenCLTensor<T> *X_raw_ptr;
      if (X_old) {
        X_raw_ptr = const_cast<OpenCLTensor<T> *>(X_old);
        X_raw_ptr->ResizeLike(Xcpu);
        deleted_unique_ptr<const OpenCLTensor<T>> X_unique_ptr(X_raw_ptr, EmptyDeleter<T>);
        return X_unique_ptr;
      } else {
        X_raw_ptr = new OpenCLTensor<T>();
        X_raw_ptr->ResizeLike(Xcpu);
        deleted_unique_ptr<const OpenCLTensor<T>> X_unique_ptr(X_raw_ptr, OpenCLTensorDeleter<T>);
        return X_unique_ptr;
      }
    }
    const OpenCLTensor<T> *X_raw_ptr;
    X_raw_ptr = &b->Get<OpenCLTensor<T>>();
    deleted_unique_ptr<const OpenCLTensor<T>> X_unique_ptr(X_raw_ptr, EmptyDeleter<T>);
    return X_unique_ptr;
  }

  /*
   * Everything below is basically boiler plate for Context classes
   */
  static std::pair<void *, MemoryDeleter> New(size_t nbytes) {
    return std::pair<void *, MemoryDeleter>(malloc(nbytes), CLContext::Delete);
  }

  static void Delete(void *data) {
    if (data != nullptr) {
      free(data);
    }
  }

  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void *src, void *dst) {}

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T *src, T *dst) {
    CopyBytes<SrcContext, DstContext>(n * sizeof(T),
                                      static_cast<const void *>(src),
                                      static_cast<void *>(dst));
  }

  template <class SrcContext, class DstContext>
  inline void CopyItems(const TypeMeta &meta, size_t n, const void *src,
                        void *dst) {
    CAFFE_ENFORCE(!meta.copy(), "CLContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  void SwitchToDevice(int a, ...) { /* TODO */
  }
  void SwitchToDevice() { SwitchToDevice(0); }

  inline void WaitEvent(const Event &ev) { /* TODO */
  }
  void FinishDeviceComputation() { /* TODO */
  }
  inline void Record(Event *ev, const char *&) const { /* TODO */
  }
  static bool IsStreamFree(const DeviceOption& /* unused */, int /* unused */) {
    return true;
  }
  bool HasAsyncPartDefault() const { return false; }
  bool SupportsAsyncScheduling() const { return false; }

  at::Device device() const {
    return at::Device(OPENCL);
  }

  DeviceType device_type() const {
    return OPENCL;
  }

  static constexpr DeviceType GetDeviceType() {
    return OPENCL;
  }

private:
  template <typename T>
  static void OpenCLTensorDeleter(const OpenCLTensor<T> *X) {
    delete X;
  }

  template <typename T>
  static void EmptyDeleter(const OpenCLTensor<T> *X) {
    return;
  }

};

template <typename T> class OpenCLTensor {
public:
  OpenCLTensor() { tensor_ = make_unique<arm_compute::CLTensor>(); }
  ~OpenCLTensor() { tensor_->allocator()->free(); }

  template <typename TensorType> bool ResizeLike(TensorType &X, bool free = false) {
    bool need_allocation = SetDims(X.dims());
    for (int i = 0; i < dims_.size(); i++) {
      shape_.set(dims_.size() - i - 1, dims_[i]);
    }

    if (need_allocation) {
      if (free) {
        tensor_->allocator()->free();
      }
      #ifdef ACL_USE_FLOAT32
      tensor_->allocator()->init(
                                 arm_compute::TensorInfo(shape_, 1, arm_compute::DataType::F32));
      #else
      tensor_->allocator()->init(
                                 arm_compute::TensorInfo(shape_, 1, arm_compute::DataType::F16));
      #endif
    } else {
      tensor_->info()->set_tensor_shape(shape_);
    }
    return need_allocation;
  }

  template <typename... Ts> bool Resize(Ts... dim_source) {
    bool need_allocation = SetDims(dim_source...);
    for (int i = 0; i < dims_.size(); i++) {
      shape_.set(dims_.size() - i - 1, dims_[i]);
    }
    if (need_allocation) {
      tensor_->allocator()->free();
      #ifdef ACL_USE_FLOAT32
      tensor_->allocator()->init(arm_compute::TensorInfo(shape_, 1, arm_compute::DataType::F32));
      #else
      tensor_->allocator()->init(arm_compute::TensorInfo(shape_, 1, arm_compute::DataType::F16));
      #endif
    } else {
      tensor_->info()->set_tensor_shape(shape_);
    }
    return need_allocation;
  }

  // Allocates and copies data if needed
  void lazy_allocate(const Blob *b, bool allocate_tensor, bool try_to_copy_from_cpu) const {
    if (try_to_copy_from_cpu) {
      // we skip OpenCLTensors, nothing to copy
      if (!b->IsType<OpenCLTensor>()) {
        // typically only called on the second run
        if (allocate_tensor) {
          allocate();
        }
        Timer timer;
        fillCLTensor(b);
        auto millis = timer.MilliSeconds();
        VLOG(2) << "[C2DEBUG] fillOpenCLTensor timer: " << millis;
      }
    }
  }

  void allocate() const {
    tensor_->allocator()->allocate();
  }

  void fillCLTensor(const Blob *b) const {
    if (b->IsType<TensorCPU>()) {
      auto &Xcpu = b->Get<TensorCPU>();
      VLOG(2) << "[C2DEBUG] fillOpenCLTensor dims: " << Xcpu.dims();
      T *buffer = map();
      char *byte_buffer = (char *)buffer;
      auto info = tensor_->info();
      arm_compute::Window it_window;
      it_window.use_tensor_dimensions(info->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY); // Iterate through the rows (not each element)
      arm_compute::Iterator it(get_underlying(), it_window);
      if (Xcpu.ndim() == 4) {
        auto C = Xcpu.dim32(1);
        auto H = Xcpu.dim32(2);
        auto W = Xcpu.dim32(3);
        arm_compute::execute_window_loop(it_window, [&](const arm_compute::Coordinates & id) {
            std::copy_n(Xcpu.data<float>() + id[3] * (C * W * H) + id.z() * (W * H) + id.y() * W, W, reinterpret_cast<T *>(it.ptr()));
          },
          it);
      } else if (Xcpu.ndim() == 3) {
        auto H = Xcpu.dim32(1);
        auto W = Xcpu.dim32(2);
        arm_compute::execute_window_loop(it_window, [&](const arm_compute::Coordinates & id) {
            std::copy_n(Xcpu.data<float>() + (id.z() * (W * H) + id.y() * W), W, reinterpret_cast<T *>(it.ptr()));
        },
        it);
      } else if (Xcpu.ndim() == 2) {
        auto W = Xcpu.dim32(1);
        arm_compute::execute_window_loop(it_window, [&](const arm_compute::Coordinates & id) {
            std::copy_n(Xcpu.data<float>() + id.y() * W, W, reinterpret_cast<T *>(it.ptr()));
        },
        it);
      } else {
        arm_compute::Window w;
        w.use_tensor_dimensions(info->tensor_shape());
        arm_compute::Iterator i(get_underlying(), w);
        auto size = Xcpu.dim32(0);
        std::copy_n(Xcpu.data<float>(), size, reinterpret_cast<T *>(i.ptr()));
      }
      unmap();
    }
  }


  int32_t ndim() const { return dims_.size(); }

  vector<int64_t> dims() const { return dims_; }

  int32_t dim32(const int index) const { return dims_.at(index); }

  int32_t size() const {
    int32_t s = 1;
    for (int i = 0; i < dims_.size(); i++) {
      s *= dims_[i];
    }
    return s;
  }

  arm_compute::CLTensor *get_underlying() const { return tensor_.get(); }

  T *map() const {
    CLContext::sync();
    tensor_->map(true);
    return reinterpret_cast<T *>(tensor_->buffer());
  }

  void unmap() const { return tensor_->unmap(); }

  void sync() const {
    CLContext::sync();
    tensor_->map();
    tensor_->unmap();
  }

private:
  bool SetDims(at::IntList dims) {
    return SetDims(dims.vec());
  }

  template <typename TI, typename = typename std::enable_if<
                             std::is_integral<TI>::value>::type>
  bool SetDims(const vector<TI> &src) {
    auto old_size = size_;
    dims_.resize(src.size());
    int64_t new_size = 1;
    for (unsigned int i = 0; i < src.size(); ++i) {
      new_size *= src[i];
      dims_[i] = src[i];
    }
    size_ = new_size;
    return size_ > old_size;
  }

  bool SetDims() {
    auto old_size = size_;
    dims_.resize(0);
    size_ = 1;
    return size_ > old_size;
  }

  bool SetDims(const int64_t d0) {
    auto old_size = size_;
    dims_.resize(1);
    dims_[0] = d0;
    size_ = d0;
    return size_ > old_size;
  }

  bool SetDims(const int64_t d0, const int64_t d1) {
    auto old_size = size_;
    dims_.resize(2);
    dims_[0] = d0;
    dims_[1] = d1;
    size_ = d0 * d1;
    return size_ > old_size;
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) {
    auto old_size = size_;
    dims_.resize(3);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    size_ = d0 * d1 * d2;
    return size_ > old_size;
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2,
               const int64_t d3) {
    auto old_size = size_;
    dims_.resize(4);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    dims_[3] = d3;
    size_ = d0 * d1 * d2 * d3;
    return size_ > old_size;
  }

  vector<int64_t> dims_;
  int64_t size_ = -1;
  arm_compute::TensorShape shape_;
  unique_ptr<arm_compute::CLTensor> tensor_;
};

template<typename T = DataType>
void getTensorCPU(const OpenCLTensor<T>& g_, Tensor& g) {
  g_.map();
  auto tensor = g_.get_underlying();
  auto info = tensor->info();
  arm_compute::Window it_window;
  it_window.use_tensor_dimensions(info->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY); // Iterate through the rows (not each element)
  arm_compute::Iterator it(tensor, it_window);
  if (g_.ndim() == 4) {
    auto C = g_.dim32(1);
    auto H = g_.dim32(2);
    auto W = g_.dim32(3);
    arm_compute::execute_window_loop(it_window, [&](const arm_compute::Coordinates & id) {
        std::copy_n(reinterpret_cast<T *>(it.ptr()), W, g.mutable_data<float>() + id[3] * (C * W * H) + id.z() * (W * H) + id.y() * W);
      },
      it);
  } else if (g_.ndim() == 3) {
    auto H = g_.dim32(1);
    auto W = g_.dim32(2);
    arm_compute::execute_window_loop(it_window, [&](const arm_compute::Coordinates & id) {
        std::copy_n(reinterpret_cast<T *>(it.ptr()), W, g.mutable_data<float>() + (id.z() * (W * H) + id.y() * W));
      },
      it);
  } else if (g_.ndim() == 2) {
    auto W = g_.dim32(1);
    arm_compute::execute_window_loop(it_window, [&](const arm_compute::Coordinates & id) {
        std::copy_n(reinterpret_cast<T *>(it.ptr()), W, g.mutable_data<float>() + id.y() * W);
      },
      it);
  } else {
    arm_compute::Window w;
    w.use_tensor_dimensions(info->tensor_shape());
    arm_compute::Iterator i(tensor, w);
    auto size = g_.dim32(0);
    std::copy_n(reinterpret_cast<T *>(i.ptr()), size, g.mutable_data<float>());
  }
  g_.unmap();
}

std::pair<arm_compute::TensorShape, arm_compute::Coordinates> compute_output_shape(arm_compute::TensorShape input_shape, unsigned int num_splits, unsigned int axis, unsigned int idx);


} // namespace caffe2

#endif /* CAFFE2_OPENCL_CONTEXT_H_ */
