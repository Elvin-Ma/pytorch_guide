import onnx
import onnxruntime
import os
import csv
import numpy as np
from onnxsim import simplify

def onnx_model_simplify(onnx_model, new_path=""):
  onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
  new_model, check = simplify(onnx_model)
  if (new_path):
    onnx.save(new_model, new_path)

  return new_model

class onnx_tensor_info:
  def __init__(self, tensor_name : 'str', data_type : 'int', data_shape : 'list'):
    self.name = tensor_name
    self.type = data_type
    self.shape = data_shape

def delete_nodes(nodes_to_delete, onnx_model, connect_node = False):
  for i in range(10):
    for node in onnx_model.graph.node:
      if node.name in nodes_to_delete:
        if connect_node:
          if node.op_type != "Constant":
            node_input_name = node.input[0]
          node_output_names = [output for output in node.output]
          for dst_node in onnx_model.graph.node:
            for index, input in enumerate(dst_node.input):
              if input in node_output_names:
                dst_node.input[index] = node_input_name
        onnx_model.graph.node.remove(node)
  return 1

def change_input_dims(onnx_model, input_name, new_shape):
  for input in onnx_model.graph.input:
    if input.name == input_name:
      assert(len(input.type.tensor_type.shape.dim) == len(new_shape))
      for i in range(len(new_shape)):
        dim_proto = input.type.tensor_type.shape.dim[i]
        dim_proto.dim_value = new_shape[i]

def delete_useless_weight(onnx_model):
  inputnamelist=[] # all input list for graph
  for node in onnx_model.graph.node:
    for inputname in node.input:
      if inputname not in inputnamelist:
        inputnamelist.append(inputname)

  for i in range(10):
    for ini in onnx_model.graph.initializer:
      if ini.name not in inputnamelist:
        onnx_model.graph.initializer.remove(ini)

  return 1

def get_weight_from_name(onnx_model, weight_name):
  for ini in onnx_model.graph.initializer:
    if weight_name == ini.name:
      raw_data = np.frombuffer(ini.raw_data, dtype=np.float32)
      return raw_data

def delete_output(onnx_model):
  for i in range(5):
    for output in onnx_model.graph.output:
      onnx_model.graph.output.remove(output)

  return 1

def delete_value_info(onnx_model):
  for i in range(20):
    for value_info in onnx_model.graph.value_info:
      # print(value_info)
      onnx_model.graph.value_info.remove(value_info)

  value_info = onnx_model.graph.value_info
  assert(not value_info)
  return -1

def append_output(output_names, onnx_model):
  for output_name in output_names:
    onnx_model.graph.output.append(onnx.ValueInfoProto(name=output_name))
  print("append_output func run successfully !!!")
  return 1

def make_output(output_info : onnx_tensor_info, onnx_model):
  """
  message TensorProto {
    enum DataType {
      UNDEFINED = 0;
      FLOAT = 1;   // float
      UINT8 = 2;   // uint8_t
      INT8 = 3;    // int8_t
      UINT16 = 4;  // uint16_t
      INT16 = 5;   // int16_t
      INT32 = 6;   // int32_t
      INT64 = 7;   // int64_t
      STRING = 8;  // string
      BOOL = 9;    // bool
      FLOAT16 = 10;
      DOUBLE = 11;
      UINT32 = 12;
      UINT64 = 13;
      COMPLEX64 = 14;     // complex with float32 real and imaginary components
      COMPLEX128 = 15;    // complex with float64 real and imaginary components
      BFLOAT16 = 16;
    }
  """
  output= onnx.helper.make_tensor_value_info(output_info.name,
                                             output_info.type,
                                             output_info.shape)
  onnx_model.graph.output.append(output)

def make_input(input_info : onnx_tensor_info, onnx_model):
  """
  message TensorProto {
    enum DataType {
      UNDEFINED = 0;
      FLOAT = 1;   // float
      UINT8 = 2;   // uint8_t
      INT8 = 3;    // int8_t
      UINT16 = 4;  // uint16_t
      INT16 = 5;   // int16_t
      INT32 = 6;   // int32_t
      INT64 = 7;   // int64_t
      STRING = 8;  // string
      BOOL = 9;    // bool
      FLOAT16 = 10;
      DOUBLE = 11;
      UINT32 = 12;
      UINT64 = 13;
      COMPLEX64 = 14;     // complex with float32 real and imaginary components
      COMPLEX128 = 15;    // complex with float64 real and imaginary components
      BFLOAT16 = 16;
    }
  """
  input = onnx.helper.make_tensor_value_info(input_info.name,
                                             input_info.type,
                                             input_info.shape)
  onnx_model.graph.input.append(input)

def shape_infer(onnx_model):
  model = onnx.shape_inference.infer_shapes(onnx_model)
  print("shape_infer fun run successfully !!!")
  return model

def find_next_nodes(onnx_model, cur_node):
  output_names = [output for output in cur_node.output]

  dst_nodes = []
  for output_name in output_names:
    for node in onnx_model.graph.node:
      input_names = [input for input in node.input]
      if output_name in input_names and node not in dst_nodes:
        dst_nodes.append(node.name)
  return dst_nodes

def onnx_model_infer(input_data_list : list, onnx_model):
  session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
  input_name = [item.name for item in session.get_inputs()]
  assert(len(input_data_list) == len(input_name))
  input_dict = dict()
  for i, data in enumerate(input_data_list):
    input_dict[input_name[i]] = data

  outputs = session.run([], input_feed = input_dict)
  print("onnx_model_infer run successfully !!!")
  return outputs

def update_onnx_batch(new_batch, onnx_model, output_spec = False, skip_item = []):
  for input in onnx_model.graph.input:
    if input.name not in skip_item:
      dim_proto0 = input.type.tensor_type.shape.dim[0]
      dim_proto0.dim_value = new_batch

  for output in onnx_model.graph.output:
    if output.name not in skip_item:
      if output_spec :
        dim_proto0 = output.type.tensor_type.shape.dim[0]
        dim_proto0.dim_value = new_batch

  delete_value_info(onnx_model)
  return onnx_model

def updata_initilizer(ini_name, onnx_model, new_data):
  for ini in onnx_model.graph.initializer:
    if ini.name == ini_name:
      ini.raw_data = new_data.tobytes()

# add output for each onnx node
def output_add(onnx_model):
  org_outputs = onnx_model.graph.output
  for node in onnx_model.graph.node:
      for output in node.output:
          if output not in org_outputs:
              output_proto = onnx.ValueInfoProto(name=output)
              if output_proto not in org_outputs:
                onnx_model.graph.output.extend([output_proto])
              org_outputs.append(output_proto)

  return 1

def golden_gen(input_data_list : list, onnx_model : onnx.ModelProto, output_path = "/home/mtn/Projects/Onnx_process/bin_datas/facenet_golden/"):
  if os.path.exists(output_path):
    os.system("rm -fr " + output_path + "/*")
  else:
    os.mkdir(output_path)

  onnx_sesssion = onnxruntime.InferenceSession(onnx_model.SerializeToString())
  input_names = [item.name for item in onnx_sesssion.get_inputs()]
  output_names = [item.name for item in onnx_sesssion.get_outputs()]
  outputs = onnx_model_infer(input_data_list, onnx_model)

  assert(len(outputs) == len(output_names))
  for index, output in enumerate(outputs):
    output_name = output_names[index].replace('/', '-')
    output_name = output_name.split("::")[-1]
    output_shape = [str(i) for i in output.shape]
    shape_str = 'x'.join(output_shape)
    output.tofile(output_path + output_name + '-' + shape_str + '-float32.bin')

  print("{} golden files generate successfully and save to : \n {}".format(len(outputs), output_path))
  return outputs

def ops_info_dump(onnx_model, save_path):
  onnx_model = shape_infer(onnx_model)

  filed_names = ["op_type", "op_name", "input_shape", "output_shape", "kernel_shape", "stride", "pads"]

  with open(save_path, mode='a', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=filed_names)
    if not os.path.getsize('facenet_subnet.csv'):
      writer.writeheader()
    op_index = 0
    for node in onnx_model.graph.node:
      op_type = node.op_type
      skip_op_type = ["Relu", "Mul", "Shape", "Cast", "BatchNormalization", "Add"]
      if (node.op_type in skip_op_type):
        continue

      # if op_type == "Conv":
      #   outputs_name = []
      #   op_type = "Cbr"
      #   for output in node.output:
      #     outputs_name.append(output)
      #     for node in onnx_model.graph.node:
      #       node_input = [name for name in node.input]
      #       if outputs_name[0] in node_input and node.op_type == "Mul":
      #         cbar = True
      #         op_type = "Cbar"


      print("=============== op count: ", op_index, "=============")
      print("1. op_type: ", node.op_type)
      print("2. op_name: ", node.name)
      print("3. op input shape: ", )
      input_shapes = []
      for input in node.input:
        for value_info in onnx_model.graph.value_info:
          if (value_info.name == input):
            input_shape = [item.dim_value for item in value_info.type.tensor_type.shape.dim]
            input_shapes.append(input_shape)
            print(input_shape)
      print("4. op output shape: ", )
      output_shapes = []
      for output in node.output:
        for value_info in onnx_model.graph.value_info:
          if (value_info.name == output):
            output_shape = [item.dim_value for item in value_info.type.tensor_type.shape.dim]
            output_shapes.append(output_shape)
            print(output_shape)
      print("5. op attrs: ")
      print(node.attribute)
      kernel_shape = []
      stride = []
      padding = []
      if (node.op_type == "Conv" or node.op_type == "MaxPool"):
        for attr in node.attribute:
          if (attr.name == "kernel_shape"):
            kernel_shape = attr.ints
          elif (attr.name == "strides"):
            stride = attr.ints
          elif (attr.name == "pads"):
            padding = attr.ints

      if op_type == "Conv":
        outputs_name = []
        op_type = "Cbr"
        for output in node.output:
          outputs_name.append(output)
          for node in onnx_model.graph.node:
            node_input = [name for name in node.input]
            if outputs_name[0] in node_input and node.op_type == "Mul":
              cbar = True
              op_type = "Cbar"

      item = [{"op_type":op_type, "op_name":node.name, "input_shape":input_shapes, "output_shape": output_shapes, "kernel_shape":kernel_shape, "stride":stride, "pads": padding}]
      writer.writerows(item)
      op_index += 1

def delete_input(onnx_model, delete_names:list):
  for i in range(10):
    for input in onnx_model.graph.input:
      if input.name in delete_names:
        onnx_model.graph.input.remove(input)

def extract_graph_after_node(onnx_model, node_name, input_tensor_info):
  onnx_model = shape_infer(onnx_model)
  new_input_name = []
  node_to_save = []
  label = 0
  for node in onnx_model.graph.node:
    if (node.name == node_name):
      for input in node.input:
        new_input_name.append(input)
      label = 1
    if (label == 1):
      node_to_save.append(node.name)

  for i in range(10):
    for node in onnx_model.graph.node:
      if node.name not in node_to_save:
        onnx_model.graph.node.remove(node)

  for input in onnx_model.graph.input:
    onnx_model.graph.input.remove(input)

  make_input(input_tensor_info, onnx_model)
  delete_useless_weight(onnx_model)

  return onnx_model

def extract_graph_before_node_v1(onnx_model, node_name, output_tensor_info : onnx_tensor_info):
  onnx_model = shape_infer(onnx_model)
  new_input_name = []
  node_to_save = []
  for node in onnx_model.graph.node:
    node_to_save.append(node.name)
    if (node.name == node_name):
      break

  for i in range(10):
    for node in onnx_model.graph.node:
      if node.name not in node_to_save:
        onnx_model.graph.node.remove(node)

  for output in onnx_model.graph.output:
    onnx_model.graph.output.remove(output)

  make_output(output_tensor_info, onnx_model)
  delete_useless_weight(onnx_model)

  return onnx_model

def extract_graph_before_node_v2(onnx_model, node_name, output_tensor_name):
  onnx_model = shape_infer(onnx_model)
  new_input_name = []
  node_to_save = []
  for node in onnx_model.graph.node:
    node_to_save.append(node.name)
    if (node.name == node_name):
      break

  for i in range(10):
    for node in onnx_model.graph.node:
      if node.name not in node_to_save:
        onnx_model.graph.node.remove(node)

  for output in onnx_model.graph.output:
    onnx_model.graph.output.remove(output)

  output_proto = onnx.ValueInfoProto(name=output_tensor_name)
  onnx_model.graph.output.append(output_proto)
  delete_useless_weight(onnx_model)

  return onnx_model

def extract_node(onnx_model, node_name, input_proto: onnx_tensor_info, output_proto: onnx_tensor_info):
  for i in range(20):
    for node in onnx_model.graph.node:
      if node.name != node_name:
        onnx_model.graph.node.remove(node)

  for i in range(5):
    for input in onnx_model.graph.input:
      onnx_model.graph.input.remove(input)

  for i in range(5):
    for output in onnx_model.graph.output:
      onnx_model.graph.output.remove(output)

  make_input(input_proto, onnx_model)
  make_output(output_proto, onnx_model)
  delete_useless_weight(onnx_model)
  return onnx_model

if __name__ == "__main__":
  onnx_path = "/public/ai_platform/CIB/ResNet50_vd_infer/rn50vd.onnx"
  new_path = "resnet50_vd.onnx"
  onnx_model = onnx.load(onnx_path)
  onnx_model_simplify(onnx_model, new_path)

  # 推理保存结果
  # outputs = onnx_model_infer([input_data], onnx_model)
  # outputs[0].tofile(output_dump_name)
  print("Run onnx utils successfully !!!")
  # os.system("sudo cp {} /public/ai_platform/mtn/onnx".format(new_model_name))
  # os.system("sudo cp {} /public/ai_platform/mtn/".format(output_dump_name))

