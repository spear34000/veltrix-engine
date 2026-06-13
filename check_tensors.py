import struct
with open('C:/Users/User/Desktop/veltrix/models/llama3.2-1b-q4_0.gguf', 'rb') as f:
    data = f.read()
    n_tensors = struct.unpack('Q', data[8:16])[0]
    n_kv = struct.unpack('Q', data[16:24])[0]
    off = 24
    for i in range(int(n_kv)):
        klen = struct.unpack('Q', data[off:off+8])[0]; off+=8
        key = data[off:off+klen].decode('utf-8'); off+=klen
        vtype = struct.unpack('I', data[off:off+4])[0]; off+=4
        if vtype == 8:
            slen = struct.unpack('Q', data[off:off+8])[0]; off+=8+slen
        elif vtype in [0,1]: off+=1
        elif vtype in [2,3,4,5,6]: off+=4
        elif vtype in [10,11,12]: off+=8
        elif vtype == 7: off+=1
        elif vtype == 9:
            at = struct.unpack('I', data[off:off+4])[0]; off+=4
            alen = struct.unpack('Q', data[off:off+8])[0]; off+=8
            for j in range(int(alen)):
                if at == 8: slen = struct.unpack('Q', data[off:off+8])[0]; off+=8+slen
                elif at in [0,1]: off+=1
                elif at in [2,3,4,5,6]: off+=4
                elif at in [10,11,12]: off+=8
                elif at == 7: off+=1
    print('All tensor names:')
    for i in range(int(n_tensors)):
        name_len = struct.unpack('Q', data[off:off+8])[0]; off+=8
        name = data[off:off+name_len].decode('utf-8'); off+=name_len
        n_dims = struct.unpack('I', data[off:off+4])[0]; off+=4
        dims = []
        for d in range(int(n_dims)):
            dims.append(struct.unpack('Q', data[off:off+8])[0]); off+=8
        ggml_t = struct.unpack('I', data[off:off+4])[0]; off+=4
        tensor_off = struct.unpack('Q', data[off:off+8])[0]; off+=8
        print(f'  {name} dims={dims} type={ggml_t}')
