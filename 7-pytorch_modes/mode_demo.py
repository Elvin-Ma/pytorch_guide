import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# 定义数据集
class MyDataset(data.Dataset):
    def __init__(self):
        # fake data
        self.data = torch.randn(100, 10)
        self.target = torch.randint(0, 2, (100,))
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.data)

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# 定义训练函数
def train(model, dataset, optimizer, criterion, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in dataset:
            # 将数据和目标转换为Tensor类型
            data = torch.tensor(data)
            target = torch.tensor(target)

            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和正确率
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # 打印训练信息
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
              .format(epoch+1, epochs, running_loss/len(dataset), 100*correct/total))

def script_demo():
    # 实例化模型
    model = MyModel()
    # 将模型转换为script模式
    scripted_model = torch.jit.script(model)
    # 定义优化器和损失函数
    optimizer = optim.SGD(scripted_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()    
    # 实例化数据集和数据加载器
    dataset = MyDataset()
    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)

    # 调用训练函数
    train(scripted_model, dataloader, optimizer, criterion, epochs=10)
    scripted_model.save("scripted_model.pt")
    
def stat_dict_demo():
    model = MyModel()
    
    aa = {"model_statdict":model.state_dict()}
    torch.save(aa, "state_dict.ckpt")
    
def traced_demo():
    model = MyModel()
    scripted_model = torch.jit.trace(model, torch.randn(1, 10))

    # 保存模型到文件
    scripted_model.save("traced_model.pt")

    # 重新加载模型
    loaded_model = torch.jit.load("traced_model.pt")

    # 重新运行模型
    input_data = torch.randn(1, 10)
    output_data = loaded_model(input_data)
    print("traced model output: ", output_data)
    
def onnx_demo():
    model = MyModel()
    torch.onnx.export(model, torch.randn(4, 10), "onnx_model.onnx")
       
def onnx_infer():
    input = torch.randn(4,10)
    # 加载模型并运行
    import onnxruntime as ort
    ort_session = ort.InferenceSession("onnx_model.onnx") # 加载模型到 session中
    ort_inputs = {ort_session.get_inputs()[0].name: input.numpy()} # 设置我们input --> numpy 格式的数据
    ort_outputs = ort_session.run(None, ort_inputs) # 开始run --> outputs --
    print("onnx run output: ", ort_outputs[0]) # 取出结果
    
        
# def dynamo_demo():
#     # 方式一：      
#     def train(model, dataloader):
#         model = torch.compile(model) # 是有开销的
#         for batch in dataloader:
#             run_epoch(model, batch)

#         def infer(model, input):
#             model = torch.compile(model)
#             return model(\*\*input)
        
#     # 方式二：
#     @optimize('inductor')
#     def forward(self, imgs, labels, mode):
#         x = self.resnet(imgs)
#         if mode == 'loss':
#             return {'loss': F.cross_entropy(x, labels)}
#         elif mode == 'predict':
#             return x, labels

def run_script_model():
    model = torch.jit.load("scripted_model.pt")
    output = model(torch.rand(4, 10))
    print("output: ", output)
    
def eager_mode():
    model = MyModel()
    torch.save(model, "model.pt")
        
if __name__ == "__main__":
    # script_demo()
    # traced_demo()
    # onnx_demo()
    # run_script_model()
    # onnx_infer()
    # eager_mode()
    stat_dict_demo()
    print("run mode_demo.py successfully!!!")