# 1. env prepare
**install method 1**
```shell
pip install transformers==4.31.0
```
**install method 2**
```shell
git clone https://github.com/huggingface/transformers
cd transformers
git checkout v4.31.0
pip install . 
# PYTHONPATH=/path/to/transformers/:$PYTHONPATH
```
**install bert depend pacakage**
```
cd transformers/examples/pytorch/token-classification
pip install -r requirements.txt
```

# 2. run model
```shell
cd transformers/examples/pytorch/token-classification
bash run.sh
```

# 3. data prepare
```python
#data address
~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076
# data and models:
config.json
model.safetensors
tokenizer_config.json
tokenizer.json
vocab.txt
```

# 4.  参考文档
[transformer github](https://github.com/huggingface/transformers/tree/main)<br>
[huggingface](https://huggingface.co/)<br>
[simple start](https://huggingface.co/bert-base-uncased)<br>
