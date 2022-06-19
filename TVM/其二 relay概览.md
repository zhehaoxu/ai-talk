Relay IR是tvm中的一个高层级中间表示，它用来表示计算图。深度学习模型使用计算图来表示，这就需要IR不仅能够表示简单的线性模型，还需要能够表示复杂的包含控制流的模型，relay正是为此推出的。

下面来看怎么构建计算图。计算图使用Var来表示输入，然后依次调用函数表示计算，接着使用Function表示一个计算图，最后将计算图传递给IRModule类。构建完计算图后可以查看相对应的relay IR表示：

```python
import tvm
from tvm import relay

x = relay.var("x", shape=(2, 3), dtype="float32")
v1 = relay.log(x)
v2 = relay.add(v1, v1)
f = relay.Function([x], v2)
mod = tvm.IRModule.from_expr(f)
print(mod)
# relay IR
# def @main(%x: Tensor[(2, 3), float32]) {
#   %0 = log(%x);
#   add(%0, %0)
# }
```

c++接口构建计算图代码如下，可以发现函数调用是Call来完成的，在后续文章中会有介绍。

```c++
auto tensor_type = relay::TensorType({2, 3}, DataType::Float(32));
auto x = relay::Var("x", tensor_type);
auto log_op = relay::Op::Get("log");
auto v1 = relay::Call(log_op, {x}, tvm::Attrs(), {});
auto add_op = relay::Op::Get("add");
auto v2 = relay::Call(add_op, {v1, v1}, tvm::Attrs(), {});
auto func = relay::Function(relay::FreeVars(v2), v2, relay::Type(), {});
auto mod = tvm::IRModule::FromExpr(func);
```

# frontend前端

已经知道怎样构建计算图，那么就可以构建复杂的深度学习模型了，tvm支持解析Pytorch、TfLite、Onnx等多种深度学习框架模型，在看完如何使用后梳理一下解析过程，以tflite模型为例：

```python
import tflite
from tvm import relay, transform

model_buf = open("mobilenet.tflite", "rb").read()
model = tflite.Model.GetRootAsModel(model_buf, 0)
# 1. 解析tflite模型
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "float32"
mod, params = relay.frontend.from_tflite(
    model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)
# 2. 优化、编译模型
target = "llvm"
with transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)
# 3. 执行推理
module = runtime.GraphModule(lib["default"](tvm.cpu()))
image_data = ...
module.set_input(input_tensor, tvm.nd.array(image_data))
module.run()
output = module.get_output(0).numpy()
```

在`from_tflite`中解析：第一步是获取模型的输入节点并转换成Var；接着是遍历模型中的算子然后调用relay中相应的函数，以ELU算子为例，假设模型中有一个ELU算子那么要使用relay来表示需要获取算子的参数、输入数据等，然后再调用relay中的函数实现ELU计算：

```python
def convert_elu(self, op):
  input_tensors = self.get_input_tensors(op)
  input_tensor = input_tensors[0]
  # 获取算子输入
  in_expr = self.get_expr(input_tensor.tensor_idx)
  exp_type = self.get_tensor_type_str(input_tensor.tensor.Type())
  # 函数调用实现elu
  out = relay.const(-1.0, exp_type) * _op.nn.relu(
    relay.const(1.0, exp_type) - _op.exp(in_expr)
    ) + _op.nn.relu(in_expr)
  # 保存结果，供下一个算子使用
  return out
```

最后使用Function构成一张计算图。这样来说要添加新的算子支持只要在frontend中调用tvm中算子的实现或者通过已有算子的组合即可，无需再添加更多的代码。

# transform图优化

定义完计算图理所当然下一步是要执行计算，但是在此之前需要对计算图进行优化。计算图优化是深度学习部署中重要的一环，计算图中难免会有一些无效的算子、重复计算的算子等，通过优化只保留必要的算子对节省资源、减少推理时间至关重要。tvm中已经提供了一些优化的策略如FoldConstant常量折叠、DeadCodeElimination死代码去除等，下面可以看到使用的方法和最终的结果：

```python
from tvm import relay
import numpy as np
import tvm

c_data = np.array([1, 2, 3]).astype("float32")
c = relay.const(c_data)
x = relay.var("x", t)
y = relay.add(c, c)
y = relay.multiply(y, relay.const(2, "float32"))
y = relay.add(x, y)
z = relay.add(y, c)
f = relay.Function([x], z)
mod = tvm.IRModule.from_expr(f)
# before
# def @main(%x: Tensor[(1, 2, 3), float32]) {
#   %0 = add(meta[relay.Constant][0], meta[relay.Constant][0]);
#   %1 = multiply(%0, 2f);
#   %2 = add(%x, %1);
#   add(%2, meta[relay.Constant][0])
# }

mod = relay.transform.FoldConstant()(mod)
# after
# def @main(%x: Tensor[(1, 2, 3), float32]) -> Tensor[(1, 2, 3), float32] {
#   %0 = add(%x, meta[relay.Constant][0], Tensor[(1, 2, 3), float32], Tensor[(3), float32]);
#   add(%0, meta[relay.Constant][1], Tensor[(1, 2, 3), float32], Tensor[(3), float32])
}
```

可以看到第一个`add`和`multiply`被消除了。要使用多个优化方案的话可以用Sequential接口来管理：

```python
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ]
)
mod = seq(mod)
```

# build代码编译

优化完计算图后接着是编译生成代码。Relay IR是高层级的中间表示，包含深度学习模型的计算语义，并没有算子的具体函数实现。在生成代码前需要将relay IR转换成低层级的Tensor IR中间表示，这一过程称为lowering。在这一过程中主要是通过人工定义schedule模板或者使用智能算法自动生成schedule，最后才是根据schedule来生成实际可执行的代码。下面通过te定义一个计算函数生成Tensor IR到C代码的例子来看：

```python
import tvm
from tvm import te

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)
tgt = tvm.target.Target(target="c", host="c")

tir = tvm.lower(s, [A, B, C])
print(tir)
# tensor ir
# @main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
#   attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
#   buffers = {C: Buffer(C_2: Pointer(float32), float32, [n: int32], [stride: int32], type="auto"),
#              A: Buffer(A_2: Pointer(float32), float32, [n], [stride_1: int32], type="auto"),
#              B: Buffer(B_2: Pointer(float32), float32, [n], [stride_2: int32], type="auto")}
#   buffer_map = {A_1: A, B_1: B, C_1: C} {
#   for (i: int32, 0, n) {
#     C_2[(i*stride)] = ((float32*)A_2[(i*stride_1)] + (float32*)B_2[(i*stride_2)])
#   }
# }
fadd = tvm.build(s, [A, B, C], tgt)
print(fadd.get_source())
# c代码
# // tvm target: c -keys=cpu -link-params=0
# #define TVM_EXPORTS
# #include "tvm/runtime/c_runtime_api.h"
# #include "tvm/runtime/c_backend_api.h"
# #include <math.h>
# #ifdef __cplusplus
# extern "C"
# #endif
# TVM_DLL int32_t default_function(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
#   void* arg0 = (((TVMValue*)args)[0].v_handle);
#   int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
#   void* arg1 = (((TVMValue*)args)[1].v_handle);
#   int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
#   void* arg2 = (((TVMValue*)args)[2].v_handle);
#   int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
#   void* A = (((DLTensor*)arg0)[0].data);
#   void* arg0_shape = (((DLTensor*)arg0)[0].shape);
#   int32_t n = ((int32_t)((int64_t*)arg0_shape)[(0)]);
#   void* arg0_strides = (((DLTensor*)arg0)[0].strides);
#   int32_t stride = ((n == 1) ? 0 : ((arg0_strides == NULL) ? 1 : ((int32_t)((int64_t*)arg0_strides)[(0)])));
#   int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
#   void* B = (((DLTensor*)arg1)[0].data);
#   void* arg1_shape = (((DLTensor*)arg1)[0].shape);
#   void* arg1_strides = (((DLTensor*)arg1)[0].strides);
#   int32_t stride1 = ((n == 1) ? 0 : ((arg1_strides == NULL) ? 1 : ((int32_t)((int64_t*)arg1_strides)[(0)])));
#   void* C = (((DLTensor*)arg2)[0].data);
#   void* arg2_shape = (((DLTensor*)arg2)[0].shape);
#   void* arg2_strides = (((DLTensor*)arg2)[0].strides);
#   int32_t stride2 = ((n == 1) ? 0 : ((arg2_strides == NULL) ? 1 : ((int32_t)((int64_t*)arg2_strides)[(0)])));
#   for (int32_t i = 0; i < n; ++i) {
#     ((float*)C)[((i * stride2))] = (((float*)A)[((i * stride))] + ((float*)B)[((i * stride1))]);
#   }
#   return 0;
# }
# 
# // CodegenC: NOTE: Auto-generated entry function
# #ifdef __cplusplus
# extern "C"
# #endif
# TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
#   return default_function(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
# }
```

本节对relay中的模块做了大致的介绍，方便了解使用的方法和模块在系统中所处的位置。每个模块具体的原理、实现将在后续的文章中一一介绍，如如何自定义transform、智能算法怎样生成schedule。

# 参考

[1] [https://zhuanlan.zhihu.com/p/386942253](https://zhuanlan.zhihu.com/p/386942253)