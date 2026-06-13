import struct
with open('C:/Users/User/Desktop/veltrix/models/llama3.2-1b-q4_0.gguf', 'rb') as f:
    data = f.read()
    print(f'File size: {len(data)} bytes')
    magic = struct.unpack('I', data[0:4])[0]
    print(f'Magic: 0x{magic:08X}')
    version = struct.unpack('I', data[4:8])[0]
    print(f'Version: {version}')
    n_tensors = struct.unpack('Q', data[8:16])[0]
    print(f'Tensors: {n_tensors}')
    n_kv = struct.unpack('Q', data[16:24])[0]
    print(f'KV pairs: {n_kv}')
    off = 24
    for i in range(int(n_kv)):
        klen = struct.unpack('Q', data[off:off+8])[0]; off+=8
        key = data[off:off+klen].decode('utf-8'); off+=klen
        vtype = struct.unpack('I', data[off:off+4])[0]; off+=4
        if 'architecture' in key or 'head_count' in key or 'embedding_length' in key or 'block_count' in key or 'feedforward' in key or 'vocab_size' in key:
            if vtype == 8:  # STR
                slen = struct.unpack('Q', data[off:off+8])[0]; off+=8
                val = data[off:off+slen].decode('utf-8'); off+=slen
                print(f'  {key}: {val}')
            elif vtype in [4,5]:  # U32/I32
                val = struct.unpack('I', data[off:off+4])[0]; off+=4
                print(f'  {key}: {val}')
            else:
                print(f'  {key}: type={vtype}')
                # skip
                if vtype == 8: slen = struct.unpack('Q', data[off:off+8])[0]; off+=8+slen
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
                continue
        else:
            if vtype == 8:  # STR
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
    print(f'KV section ends at offset: {off}')
