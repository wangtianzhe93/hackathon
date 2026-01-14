# torch 2.x.x

## torch compile

    torch.complie目的是提高计算速度, 使用了torch之后许多功能只需要一行代码即可完成, 例如: model = torch.compile(model).
    之所以一行能让整个pytorch运算提速, 是因为complie是一个高级接口.
    它背后使用了 TorchDynamo, AOTAutograd和TorchInductor等工具链来对模型的计算图进行分析, 优化和编译.
    torch.complie是与开发者关系最大的, 最有别于1.x的特性, 它背后对于计算图的分析, 优化和编译是本次更新的核心构成.
    但对于普通用户而言, 了解好torch.compile的接口, 了解其可提高模型计算速度就可以.

## torch dynamo
    TorchDynamo是支撑torch.compile的工具, 它可进行快速地捕获计算图(Graph), 计算图在深度学习中至关重要.
    它描述了数据在网络中的流动形式, 在早期, pytorch团队已经对计算图的捕获进行了一些列工具开发, 例如TorchScript.
    但TorchDynamo相较于之前的工具, 在速度上有了更大提升, 并且在99%的情况下都能正确 安全地获取计算图.

## autograd
    AOTAutograd的目的是希望在计算运行之前, 捕获计算的反向传播过程, 即"ahead of time Autograd".
    AOTAutograd通过重用和扩展PyTorch的现有自动微分系统, 实现提高训练速度.

## torch inductor
    TorchInductor是一个新的编译器后端, 可以为多个硬件平台进行生成优化的代码, 可以针对NVIDIA和AMD的GPU.
    使用triton作为目标语言, 针对CPU可生成C++代码. TorchInductor能够为多种加速器和后端生成快速的代码.

## prim torch
    PrimTorch是将PyTorch底层操作符(operators)进行归约 精简, 使下游编译器开发更容易和高效.
    PyTorch包含1200+操作符, 算上重载有2000+操作符, 对于后端和编译器开发极不友好.
    为了简化后端开发, 提高效率, PrimTorch项目整理了两大类基础操作符, 包括:
        Prim操作符：相对底层的约250个操作符
        ATen操作符：约750个操作符, 适合直接导出

## torch.compile interface
    根据官方文档定义, "Optimizes given model/function using TorchDynamo and specified backend."
    torch.compile是采用TorchDynamo和指定的后端对模型/计算进行优化, 期望使模型/函数在未来应用时, 计算速度更快.
    在使用上, torch.compile接收一个可调用对象(Callable), 返回一个可调用对象(Callable), 对于用户只需要一行代码就可以调用compile进行优化.

### parameters
    model(Callable)
        Module或者是Function, 这个Function可以是pytorch的函数, 也可以是numpy语句, compile也支持numpy的加速优化.

    mode
        优化模式的选择, 目前提供了四种模式, 区别在于不同的存储消耗、时间消耗、性能之间的权衡.
        default
            默认模式, 在性能和开销之间有不错的平衡
        reduce-overhead
            这个模式旨在减少使用CUDA图时的Python开销.
            该模式会增加内存占用, 提高速度, 并且不保证总是有效.
            目前, 这种方法只适用于那些不改变输入的CUDA图.
        max-autotune
            基于Triton的矩阵乘法和卷积来提高性能.
        max-autotune-no-cudagraphs
            与max-autotune一样, 但是不会使用CUDA计算图.

    fullgraph(bool)
        是否将整个对象构建为单个图(a single graph), 否认是False, 即根据compile的机制拆分为多个子图.

    dynamic(bool or None)
        是否采用动态形状追踪, 默认为None, 对于输入形状是变化的, compile会尝试生成对应的kernel来适应动态形状, 从而减少重复编译.
        但并不是所有动态形状都能这样操作, 这个过程可以设置TORCH_LOGS=dynamic来观察日志信息.

    backend(str or Callable)
        选择所用的后端, 默认是"inductor", 可以较好平衡性能和开销.
        可用的后端可以通过torch._dynamo.list_backends()查看, 注册自定义后端库, 可参考 https://pytorch.org/docs/main/compile/custom-backends.html

    options(dict)
        用于向后端传入额外数据信息, key-value可以自定义, 只要后端可读取即可, 这个参数预留了较好的接口.

    disable(bool)
        Turn torch.compile() into a no-op for testing
