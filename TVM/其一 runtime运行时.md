# 1. 部署模型

部署模型时需要运行时库，编译完tvm工程后会得到libtvm_runtime.so运行时库，由于运行时库只用于推理模型所以库的大小只有几MB。部署模型首先需要使用tvm来编译生成模型，有两种形式的模型：一种是普通的函数，通过compute和schedule来定义，并最终生成函数的实现保存到`.so`共享库中，如下代码：

```python
n = te.var("n")
A = te.placeholder((n,), name="A")
# 1. 定义计算模式
B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
# 2. 创建函数的schedule
s = te.create_schedule(B.op)
# 3. 编译、生成函数实现
fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
# 保存模型
dylib_path = os.path.join(base_path, "test_addone_dll.so")
fadd_dylib.export_library(dylib_path)
```

第二种是通过relay来构建的深度学习计算图图模型，如下代码：

```python
x = relay.var("x", shape=(2, 2), dtype="float32")
y = relay.var("y", shape=(2, 2), dtype="float32")
params = {"y": np.ones((2, 2), dtype="float32")}
# 1. 创建模型
mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
# 2. 编译模型
compiled_lib = relay.build(mod, tvm.target.create("llvm"), params=params)
# 3. 保存到so文件中
dylib_path = os.path.join(base_path, "test_relay_add.so")
compiled_lib.export_library(dylib_path)
```

得到模型后接下来就是将模型部署到设备上，流程是：首先加载模型，为输入、输出分配内存等，调用函数执行计算，最后释放内存。普通函数模型部署代码如下：

```cpp
// 1. 加载共享库
tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("test_addone_dll.so");
// 2. 根据函数名获取函数实现，函数名要跟创建时指定的一样
tvm::runtime::PackedFunc f = mod.GetFunction("addone");
// 3. 设置输入、输出，分配内存
DLTensor* x; DLTensor* y; int ndim = 1; int dtype_code = kDLFloat; int dtype_bits = 32;
int dtype_lanes = 1; int device_type = kDLCPU; int device_id = 0; int64_t shape[1] = {10};
TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
// 4. 为输入设置数据
for (int i = 0; i < shape[0]; ++i) {
	static_cast<float*>(x->data)[i] = i;
}
// 5. 激活函数，执行推理
f(x, y);
// 6. 读取输出结果
for (int i = 0; i < shape[0]; ++i) {
	ICHECK_EQ(static_cast<float*>(y->data)[i], i + 1.0f);
}
//7. 释放内存
TVMArrayFree(x);
TVMArrayFree(y);
```

深度学习计算图模型部署流程大体类似，代码如下：

```cpp
DLDevice dev{kDLCPU, 0};
// 1. 加载共享库
tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/test_relay_add.so");
// 2. 获取模块及相应的函数
tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
tvm::runtime::PackedFunc run = gmod.GetFunction("run");
// 3. 设置输入、输出，分配内存
tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);
tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({2, 2}, DLDataType{kDLFloat, 32, 1}, dev);
// 4. 为输入设置数据
for (int i = 0; i < 2; ++i) {
  for (int j = 0; j < 2; ++j) {
    static_cast<float*>(x->data)[i * 2 + j] = i * 2 + j;
  }
}
// 5. 为模型添加输入
set_input("x", x);
// 6. 执行推理
run();
// 7. 获取输出
get_output(0, y);
// 8. 读取输出数据
for (int i = 0; i < 2; ++i) {
  for (int j = 0; j < 2; ++j) {
    ICHECK_EQ(static_cast<float*>(y->data)[i * 2 + j], i * 2 + j + 1);
  }
}
```

# 2. runtime实现

## 2.1 模型加载LoadFromFile解析

上面部署模型第一步就是加载模型共享库文件，下面就介绍LoadFromFile是如何从文件中加载函数的。

```cpp
Module Module::LoadFromFile(const std::string& file_name, const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  if (fmt == "dll" || fmt == "dylib" || fmt == "dso") {
    fmt = "so";
  }
  std::string load_f_name = "runtime.module.loadfile_" + fmt;
  const PackedFunc* f = Registry::Get(load_f_name);
  Module m = (*f)(file_name, format);
  return m;
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_so").set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectPtr<Library> n = CreateDSOLibraryObject(args[0]);
  *rv = CreateModuleFromLibrary(n);
});
```

加载so文件如上面代码所示实际上会调用`runtime.module.loadfile_so`指向的函数。在tvm中有大量的函数是通过Registry类来注册和获取的，注册时通过宏`TVM_REGISTER_GLOBAL`创建一个Registry实例并将该实例添加到Manager单例类中，在获取函数时根据名称从Manager单例类中检索得到相应的实现。在`runtime.module.loadfile_so`指向的函数中首先是创建了DSOLibrary的实例并且在初始化过程中通过dlopen函数打开共享库，然后再调用CreateModuleFromLibrary函数加载部分功能最终返回一个Module实例。在上面模型部署中可以知道普通函数和计算图模型略有不同，所以下面的代码分别做了处理，`else`代码块中用来加载普通函数而`if`代码块中用来加载计算图。ps：可以使用`objdump -t xxx.so`命令查看so文件中的字段。

```cpp
Module CreateModuleFromLibrary(ObjectPtr<Library> lib, PackedFuncWrapper packed_func_wrapper) {
  InitContextFunctions([lib](const char* fname) { return lib->GetSymbol(fname); });
  auto n = make_object<LibraryModuleNode>(lib, packed_func_wrapper);
  // 判断是否是计算图
  const char* dev_mblob =
      reinterpret_cast<const char*>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));

  Module root_mod;
  runtime::ModuleNode* dso_ctx_addr = nullptr;
  if (dev_mblob != nullptr) {
		// 加载计算图
    ProcessModuleBlob(dev_mblob, lib, packed_func_wrapper, &root_mod, &dso_ctx_addr);
  } else {
    // 加载普通函数
    root_mod = Module(n);
    dso_ctx_addr = root_mod.operator->();
  }
  return root_mod;
}
```

普通函数模型要获取函数实现只需要为GetFunction传入函数名即可，GetFunction内部是通过调用dlsym来实现的。计算图模型先是使用字符串default获取到module，接着再从module中得到set_input、get_output、run函数。

## 2.2 PackedFunc机制理解

PackedFunc在tvm中到处都有用到，那它究竟是什么呢？正如名称所表示的，它用来对函数封装。tvm代码库中有大量的函数，函数的参数、返回等各不相同，PackedFunc将函数封装后可以使用统一的方式来使用，使用示例如下代码：

```cpp
// 封装函数
PackedFunc addone([&](TVMArgs args, TVMRetValue* rv) { 
											int ret = args[0] + 1;
											*rv = ret; });
// 使用函数
int ret = addone(123);
```

在封装函数时需要遵循一定的规则：函数调用时的实参都存储在第一个形参TVMArgs变量中，通过下标来获取相应的数据；第二个形参TVMRetValue变量用来存储返回值。其中涉及到许多细节，下面通过代码来介绍。

PackedFunc构造函数接受`void(TVMArgs, TVMRetValue*)`形式的函数，将其赋值给内部变量`data_`，重点在于函数的调用。如上面示例代码所示，PackedFunc类的实例如同函数般使用`()`直接执行，这是通过运算符重载来实现的，跟python类中实现__call__函数是一样的道理。

```cpp
template <typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&&... args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  // 将参数保存到TVMValue中
  detail::for_each(TVMArgsSetter(values, type_codes), std::forward<Args>(args)...);
  TVMRetValue rv;
  // 函数执行并将结果保存到TVMRetValue中
  (static_cast<PackedFuncObj*>(data_.get()))
      ->CallPacked(TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}
```

首先是将函数调用传入的实参保存到TVMValue类型的数组中，TVMValue是union联合体类型可以保存不同类型的数据。由于参数有多种类型如int、float、double等，所以为了记录参数的类型使用type_codes变量数组方便获取数据。如下样例所示：

```cpp
函数:             f(5, 3.14f, "hello")
values:          [5, 3.14, "hello"]
type_codes:      [kDLInt, kDLFloat, kTVMStr]
```

接着是执行PackedFunc封装的函数，即`data_`变量指向的函数。结果保存在TVMRetValue类型变量中并且返回给调用者，但是我们从上面使用示例中发现返回的类型并非TVMRetValue而是int，这也是通过函数重载实现数据类型隐式转换。在TVMRetValue父类TVMPODValue_中重载了一些类型转换函数：

```cpp
operator double() const {
  if (type_code_ == kDLInt) {
    return static_cast<double>(value_.v_int64);
  }
  TVM_CHECK_TYPE_CODE(type_code_, kDLFloat);
  return value_.v_float64;
}
operator int64_t() const {
  TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
  return value_.v_int64;
}
...
```

其实前面在通过下标获取TVMArgs类型数据时已经用到了隐式类型转换，感兴趣的可以查看TVMArgs的`[]`函数实现。

# 3. python接口

tvm项目提供多种语言的接口，他们底层都是调用C++的实现。下面介绍python接口是如何调用C++代码的。

安装tvm python库有两种方式：第一种是编译完工程后在python文件夹下执行`python setup.py install`，这种方式会将libtvm.so共享库拷贝到python第三方库的位置；另一种是设置环境变量引用编译后的共享库，这种方式很方便需要经常修改源码的情况。设置如下：

```bash
// 方法一
cd python
python setup.py install
// 方法二
export TVM_HOME=/path/to/tvm/
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

tvm python接口使用ctypes库来加载动态连接库，加载后就可以通过函数名来调用。python库中的函数都可以调用，不妨来看看运行结果，如下：

```python
def _load_lib():
    lib_path = libinfo.find_lib_path()
    # 加载动态链接库
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    lib.TVMGetLastError.restype = ctypes.c_char_p
    return lib, os.path.basename(lib_path[0])

>>> from tvm._ffi.base import _load_lib
>>> lib, lib_name = _load_lib()
>>> lib  # <CDLL '/path/tvm/build/libtvm.so', handle ...>
>>> lib_name  # libtvm.so
```

tvm中有大量的函数，可以使用`list_global_func_names()`获取函数名称，它是调用C++的TVMFuncListGlobalNames函数来实现的。在C++中对外公开的函数都使用Registry类注册到单例类实例中，然后就可以从实例中获取所有的函数了。另外，不同的函数属于不同的模块，因此在python中将不同类型的函数封装在对应的模块中，如”topi.nn.bias_add”函数被封装在topi模块下的nn模块中。在tvm的python库模块文件夹下有`_ffi_api.py`文件，在其中加载对应模块函数，如runtime的加载命令`tvm._ffi._init_api("runtime", __name__)`。_init_api中所做的就是根据函数的名称进行模块的划分：

```python
for name in list_global_func_names():
  # 1. 函数名称
  fname = name[len(prefix) + 1 :]
  target_module = module
  # 2. 获取函数实现
  f = get_global_func(name)
  ff = _get_api(f)
  ff.__name__ = fname
  ff.__doc__ = "TVM PackedFunc %s. " % fname
  # 3. 将函数添加到模块中
  setattr(target_module, ff.__name__, ff)
```

最终，”tvm.runtime.MapCount”函数被映射成runtime模块下的_ffi_api模块下的MapCount函数。映射后的函数参数类型与python类型是不匹配的，所以在`get_global_func`获取函数实现后传递给PackedFunc类实例的handle变量，实例执行__call__重载函数时对参数类型进行转换：

```python
def __call__(self, *args):
	temp_args = []
  # 1. 将参数封装成TVMValue
  values, tcodes, num_args = _make_tvm_args(args, temp_args)
  ret_val = TVMValue()
  ret_tcode = ctypes.c_int()
  if (_LIB.TVMFuncCall(
	    self.handle, values, tcodes, ctypes.c_int(num_args),
      ctypes.byref(ret_val), ctypes.byref(ret_tcode),) != 0):
      raise get_last_ffi_error()
	_ = temp_args
  _ = args
  # 2. 将返回转换成相应类型
  return RETURN_SWITCH[ret_tcode.value](ret_val)
```