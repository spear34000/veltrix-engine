import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import os

n_layers = 2
n_embd = 64
n_ff = 128
n_vocab = 100
n_heads = 4
n_kv_heads = 1
kv_dim = n_kv_heads * (n_embd // n_heads)

tensors = []

data = np.random.randn(n_vocab, n_embd).astype(np.float32) * 0.01
tensors.append(numpy_helper.from_array(data, name='token_embd.weight'))

for l in range(n_layers):
    for spec in [
        ('attn_q.weight', (n_embd, n_embd)),
        ('attn_k.weight', (kv_dim, n_embd)),
        ('attn_v.weight', (kv_dim, n_embd)),
        ('attn_output.weight', (n_embd, n_embd)),
        ('ffn_gate.weight', (n_ff, n_embd)),
        ('ffn_up.weight', (n_ff, n_embd)),
        ('ffn_down.weight', (n_embd, n_ff)),
    ]:
        name = f'blk.{l}.{spec[0]}'
        shape = spec[1]
        data = np.random.randn(*shape).astype(np.float32) * 0.01
        tensors.append(numpy_helper.from_array(data, name=name))

data = np.ones(n_embd, dtype=np.float32)
tensors.append(numpy_helper.from_array(data, name='output_norm.weight'))

data = np.random.randn(n_vocab, n_embd).astype(np.float32) * 0.01
tensors.append(numpy_helper.from_array(data, name='output.weight'))

graph = helper.make_graph([], 'test_model', [], [], tensors)
model = helper.make_model(graph, producer_name='veltrix-test')
model.opset_import[0].version = 17

out_path = os.path.join(os.path.dirname(__file__) or '.', 'test_synthetic.onnx')
onnx.save(model, out_path)
print(f'Created {out_path} ({os.path.getsize(out_path)} bytes)')
print(f'Total initializers: {len(tensors)}')

m2 = onnx.load(out_path)
print(f'Verified: {len(m2.graph.initializer)} initializers')
