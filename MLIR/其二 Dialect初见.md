# Dialect初见

Dialect直译为方言，是MLIR可扩展特性中重要的一环。它可以看作是一个容器，官方文档里称它`provide a grouping mechanism for abstraction under a unique namespace`，它包含许多`operations`算子、`types`类型、`attributes`属性等。`Operation`可以译做算子或者操作，是MLIR中核心的单元。只介绍概念不能让人理解其中的含义，举个例子：以人类语言为例，Dialect相当于中文，而Operation相当于单词；以Toy语言为例，Dialect相当于Toy编程语言，Operation相当于`+、-、*、/`等算数运算，`transpose`、`print`函数计算。当然Toy语言不止有operations，还有数据类型`types`如`double`，还有属性`attributes`如变量的`shape`。下面就创建Toy的Dialect吧！
    
创建Toy Dialect需要继承Dialect类并实现部分接口，就如下面这样：

```cpp
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  // toy dialect的命名空间
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  // 初始化函数，在实例化时调用
  void initialize();
};
```

仅此而已吗？确实！但是我们有更好的办法。使用C++我们需要写很多的模板代码而实际核心的代码却很少，那么能不能让程序帮我们生成这些代码呢？`tablegen`此时闪亮登场！我们只要遵循一定的规范将核心代码实现，其余的统统让tablegen来生成。下面来看怎样在`.td`文件中编写代码：

```cpp
// tablegen.td
include "mlir/IR/OpBase.td"  // 导入依赖的文件

def Toy_Dialect : Dialect {
  let name = "toy";  // dialect的名称
  // 一行简短的说明
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";
  // 更详细的说明
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];
  // 生成C++代码的namespace命名空间
  let cppNamespace = "toy";
}
```

在文件中使用`def`关键字定义dialect，`:`后面表示要继承的类，然后就是设置所需的字段了，如`name`、`summary`、`cppNamespace`等，还有更多的字段留在后面的时候再介绍。定义完dialect后接下来就是生成真正的C++代码了，通过`mlir-tblgen`工具：

```cpp
./build/bin/mlir-tblgen -gen-dialect-decls tablegen.td -I ./mlir/include/
```

`-I`用来指定依赖的文件。此时生成ToyDialect类的声明：

```cpp
namespace toy {

class ToyDialect : public ::mlir::Dialect {
  explicit ToyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<ToyDialect>()) {

    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ToyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("toy");
  }
};
} // namespace toy
```

`mlir-tblgen`还有更多的选项可以生成不同的东西，使用`mlir-tblgen -h`看一看：

```cpp
USAGE: mlir-tblgen [options] <input file>

OPTIONS:

General options:

  -D=<macro name>                - Name of the macro to be defined
  -I=<directory>                 - Directory of include files
  --asmformat-error-is-fatal     - Emit a fatal error if format parsing fails
  -d=<filename>                  - Dependency filename
  Generator to run
      --gen-attr-interface-decls            - 生成属性接口声明
      --gen-attr-interface-defs             - 生成属性接口定义
      --gen-attr-interface-docs             - 生成属性接口文档
      --gen-dialect-decls                   - 生成dialect声明
      --gen-dialect-defs                    - 生成dialect定义
      --gen-dialect-doc                     - 生成dialect文档
      --gen-enum-decls                      - Generate enum utility declarations
      --gen-enum-defs                       - Generate enum utility definitions
      --gen-op-decls                        - Generate op declarations
      --gen-op-defs                         - Generate op definitions
      --gen-op-doc                          - Generate dialect documentation
      --gen-pass-decls                      - Generate operation documentation
      --gen-pass-doc                        - Generate pass documentation
      --gen-rewriters                       - Generate pattern rewriters
...
```

接着使用`-gen-dialect-defs`生成ToyDialect的定义：

```cpp
DEFINE_EXPLICIT_TYPE_ID(toy::ToyDialect)
namespace toy {

ToyDialect::~ToyDialect() = default;

} // namespace toy
```

唔呣，其实什么也没生成，此时`initialize`函数还没有实现，而这正是我们需要自己实现的。在`initialize`中一般会添加`Operation`、`Interface`、`Type`等到ToyDialect中，就如：

```cpp
void ToyDialect::initialize() {
  // 添加Operation
  addOperations<
      #define GET_OP_LIST
      #include "Ops.cpp.inc"  // 导入生成的operation
  >();
}
```

定义Dialect与Operation、Type等是大大不同的，就让我们来看看如何定义Opeartion吧！

> 不用担心工程中太多的.td文件需要我们一一生成代码，在CMakeLists文件中添加指令会在构建工程时自动帮我们生成。😃

# 定义Operation

Operation用来表示程序的语义信息，比如`+`运算使用`AddOp`来表示，那么后续就可以对`AddOp`进行分析从而优化，要注意的是程序中的常量constant是Operation，而Operation的输入、输出是Operand，实际上程序就是由许多的Operation和Operand连接起来的计算图。看看如何实现一个常量算子：

```cpp
class ConstantOp : public mlir::Op<
                     ConstantOp,
                     // ConstantOp没有输入，程序中的字面值存储在value属性中
                     mlir::OpTrait::ZeroOperands,
                     // ConstantOp返回一个结果
                     mlir::OpTrait::OneResult,
                     // 返回结果的数据类型
                     mlir::OpTraits::OneTypedResult<TensorType>::Impl> {

 public:
  using Op::Op;

  // 获取算子的名字。前面必须添加dialect的名称
  static llvm::StringRef getOperationName() { return "toy.constant"; }
  // 获取ConstantOp的value属性
  mlir::DenseElementsAttr getValue();
  // 用来验证算子的定义
  LogicalResult verify();

  // 用来构造算子，通过mlir::OpBuilder::create<ConstantOp>(...)使用
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

定义Operation需要继承`mlir::Op`模板类，模板的第一个参数是算子本身，如ConstantOp，这在C++中叫做`CRTP`(奇异递归模板模式)；接下来的参数是一些可选的`traits`(特性萃取)，可以简单地看作是对Operation的约束，比如ConstantOp没有输入，所以使用`OpTrait::ZeroOperands`；只有一个输出，所以使用`OpTrait::OneResult`。不同算子有不同的特性，所以需要使用相应的`OpTrait`来进行定义。接着就是实现一些接口，如`verify`和用来构造算子的`build`函数。定义一个算子我们同样需要编写大量的模板代码，这时`ODS`(Operation Definition Specification)就派上了用场。其实和定义Dialect时使用tablegen一样，ODS也是使用tablegen来生成C++代码。

```cpp
class Toy_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

首先定义一个基类方便算子继承。`mnemonic`用来指定算子的名称，`traits`用来指定算子特性。

```cpp
def ConstantOp : Toy_Op<"constant"> {
	// 定义输入并为其指定value变量名
  let arguments = (ins F64ElementsAttr:$value);
	// 定义输出，64位浮点tensor类型
  let results = (outs F64Tensor);
}
```

通过`def`来定义算子并且继承我们之前定义的基类，与C++定义不同这里没有指定traits，tablegen会根据`arguments`和`results`进行推断并为生成的C++代码中添加相应的traits。`arguments`字段用来指定算子输入，`results`用来指定算子输出。还有一些字段如使用`summary`添加一行简短说明；使用`description`添加更加详细的说明；另外还可以使用`verifier`对算子进行验证。tablegen能生成简单的`build`接口，如果需要更多的自定义接口则可以指定`builders`字段，如：

```cpp
def ConstantOp : Toy_Op<"constant"> {
  ...

  let builders = [
    // 通过一个常量tensor来构建算子.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // 通过一个常量来构建算子
    OpBuilder<(ins "double":$value)>
  ];
}
```

可以看到`builders`是一个数组并且每一个build接口都是使用`OpBuilder`来定义的。其中`build($_builder, $_state, value.getType(), value)`是调用默认的build接口，`$_`开头的变量是内置变量，后续再详细地介绍。至此一个算子就定义完成了，其他算子与此类似，下面看一下生成的具体C++代码是什么样的，使用`./build/bin/mlir-tblgen -gen-op-decls examples/toy/Ch2/include/toy/Ops.td -I mlir/include/`生成声明代码，除了生成算子外还会生成相应的Adaptor类，如下：

```cpp
namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ConstantOp declarations
//===----------------------------------------------------------------------===//

class ConstantOpAdaptor {
	...
};
class ConstantOp : public ::mlir::Op<ConstantOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  ...
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.constant");
  }
  ...
  ::mlir::DenseElementsAttr value();

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, double value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  ...
};
} // namespace toy
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::ConstantOp)
```

使用`-gen-op-defs`生成算子定义。

> mlir项目编译后toy示例生成的代码可以在`/build/tools/mlir/examples/toy/`目录下找到。

# IR转换

定义完Dialect和Operation接着就是使用算子来表示toy AST。与AST结构类似，使用ModuleOp来表示整个程序，使用FuncOp来表示函数，使用MulOp来表示相乘等，使用算子分别对应AST中的Expr最终得到一个`mlir::OwningModuleRef`类型的实例。

```cpp
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

上面toy代码通过`/bin/toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo`生成的IR形式如下：

```cpp
module  {
  func @multiply_transpose(%arg0: tensor<*xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":4:1)) -> tensor<*xf64> {
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("./test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("./test/Examples/Toy/Ch2/codegen.toy":4:1)
  func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("./test/Examples/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("./test/Examples/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("./test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("./test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```