# dynamic and static graph
* PyTorch的动态图（eager execution）和TensorFlow的静态图:
- 是两种不同的计算图构建方式;
- 它们最核心的区别在于计算图的构建和执行方式;
- 在PyTorch的动态图中，每个操作都是立即执行的(在Python中编写代码的同时进行计算，这使得代码编写和调试非常方便).
- 同时，由于计算图是动态构建的，因此可以在运行时根据需要动态改变计算图的结构和形状。这种动态构建计算图的方式使得PyTorch非常适合快速原型开发和实验，同时也使得代码编写和调试变得更加直观和容易。

- 相比之下，TensorFlow的静态图需要先定义计算图的结构，然后再将数据输入到计算图中进行计算。
- 这种静态构建计算图的方式使得TensorFlow能够进行更多的优化和预处理，例如算子融合、图优化等。
- 同时，在计算图构建完成后，TensorFlow可以将计算图编译为高效的本机代码，以提高计算效率。
- 这种静态构建计算图的方式使得TensorFlow非常适合生产环境和大规模训练任务。 *

** 结论 **<br>
因此，PyTorch的动态图和TensorFlow的静态图在计算图的构建和执行方式上有着本质的区别。动态图更灵活，更适合快速原型开发和实验，而静态图则更加高效，更适合生产环境和大规模训练任务。

# pytorch jit mode
** 在PyTorch中，eager mode是默认的运行模式，它允许您在Python中编写和调试代码，并立即看到结果。而JIT（Just-In-Time）编译器则可以将PyTorch的计算图编译为高效的本机代码，以便在生产环境中使用。

在eager mode下，每个PyTorch操作都是由Python解释器执行的，每个操作都可能涉及到数据的创建、分配、释放等操作，因此会产生一些Python的开销。此外，在eager mode下，PyTorch的计算图是动态构建的，即每次运行时都会重新构建，这会导致一些额外的开销。

而在JIT编译器中，PyTorch的计算图被静态地构建，并且可以优化为高效的本机代码。这意味着一些Python的开销可以被消除，从而提高运行效率。此外，JIT编译器还可以进行一些优化，例如常量折叠、循环展开、内存分配优化等，以进一步提高性能。

当您使用torch.jit.trace()函数将PyTorch模型转换为JIT图时，最终得到的是一个静态的计算图，其中所有的操作都被编译为本机代码，从而可以在生产环境中高效地运行。与eager mode下的计算图相比，JIT图的构建和优化过程需要一些额外的时间，但是可以获得更高的性能和更小的内存占用。

需要注意的是，JIT图和eager mode下的计算图并不完全相同，因为JIT图是静态构建的，而eager mode下的计算图是动态构建的。这意味着，在某些情况下，两者可能会有一些细微的差异，例如操作顺序、内存布局等。因此，在使用JIT编译器时，建议使用与eager mode下相同的数据集和参数进行测试，以确保结果的正确性。 **

# pytorch export onnx 和 jit：
* torch.jit.trace()和torch.onnx.export()是PyTorch中用于将模型转换为可部署形式的两种不同方法。

torch.jit.trace()函数将PyTorch模型转换为ScriptModule对象，它是一种轻量级的序列化模型表示形式，可以在Python环境中运行，并且支持跨平台和跨语言部署。使用torch.jit.trace()函数，您可以将一个具有固定输入和输出的模型转换为ScriptModule对象，然后将其保存为文件或在Python中使用。

torch.onnx.export()函数则将PyTorch模型转换为ONNX（Open Neural Network Exchange）格式，这是一种开放标准的模型表示形式，支持跨平台和跨语言部署。使用torch.onnx.export()函数，您可以将一个具有固定输入和输出的模型转换为ONNX格式，然后将其保存为文件或在其他支持ONNX格式的框架中使用。

因此，torch.jit.trace()和torch.onnx.export()的最大区别在于输出的模型格式不同。ScriptModule对象是PyTorch独有的模型表示形式，它可以直接在Python中运行，而ONNX格式是一种通用的模型表示形式，可以在多个框架和平台上运行。

另外，torch.jit.trace()和torch.onnx.export()的输入和输出都需要是静态的，即输入和输出的形状和类型需要在运行时已知且不变。如果您的模型包含动态的控制流程或变长的输入序列，可能需要使用其他方法将其转换为可部署形式。 *



