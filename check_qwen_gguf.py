import struct

files = [
    r'C:\Users\User\Desktop\veltrix\models\qwen2.5-0.5b-instruct-q4_0.gguf',
    r'C:\Users\User\Desktop\veltrix\models\qwen2.5-0.5b-q4_0.gguf',
]

for path in files:
    short = path.split('\\')[-1]
    print(f'=== {short} ===')
    with open(path, 'rb') as f:
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
            if vtype == 8:
                slen = struct.unpack('Q', data[off:off+8])[0]; off+=8
                val = data[off:off+slen].decode('utf-8'); off+=slen
                print(f'  [STR] {key}: {val}')
            elif vtype in [0,1]:
                val = struct.unpack('b' if vtype==1 else 'B', data[off:off+1])[0]; off+=1
                print(f'  [INT8] {key}: {val}')
            elif vtype in [2,3]:
                val = struct.unpack('H' if vtype==2 else 'h', data[off:off+2])[0]; off+=2
                print(f'  [INT16] {key}: {val}')
            elif vtype in [4,5]:
                val = struct.unpack('I' if vtype==4 else 'i', data[off:off+4])[0]; off+=4
                print(f'  [INT32] {key}: {val}')
            elif vtype == 6:
                val = struct.unpack('f', data[off:off+4])[0]; off+=4
                print(f'  [F32] {key}: {val}')
            elif vtype in [10,11]:
                val = struct.unpack('Q' if vtype==10 else 'q', data[off:off+8])[0]; off+=8
                print(f'  [INT64] {key}: {val}')
            elif vtype == 12:
                val = struct.unpack('d', data[off:off+8])[0]; off+=8
                print(f'  [F64] {key}: {val}')
            elif vtype == 7:
                val = 'true' if data[off] else 'false'; off+=1
                print(f'  [BOOL] {key}: {val}')
            elif vtype == 9:
                at = struct.unpack('I', data[off:off+4])[0]; off+=4
                alen = struct.unpack('Q', data[off:off+8])[0]; off+=8
                if at == 8:
                    items = []
                    for j in range(int(alen)):
                        slen2 = struct.unpack('Q', data[off:off+8])[0]; off+=8
                        item = data[off:off+slen2].decode('utf-8'); off+=slen2
                        items.append(item)
                    print(f'  [STR_ARRAY({alen})] {key}: first={items[0] if items else None}, last={items[-1] if items else None}')
                elif at in [4,5]:
                    items = []
                    for j in range(int(alen)):
                        val2 = struct.unpack('I', data[off:off+4])[0]; off+=4
                        items.append(val2)
                    print(f'  [INT32_ARRAY({alen})] {key}: first={items[0] if items else None}, last={items[-1] if items else None}')
                elif at == 6:
                    items = []
                    for j in range(int(alen)):
                        val2 = struct.unpack('f', data[off:off+4])[0]; off+=4
                        items.append(val2)
                    print(f'  [F32_ARRAY({alen})] {key}: len={len(items)}')
                elif at == 0:
                    for j in range(int(alen)): off+=1
                    print(f'  [U8_ARRAY({alen})] {key}')
                else:
                    print(f'  [ARRAY type={at} len={alen}] {key}')
                    for j in range(int(alen)):
                        if at == 8: slen3 = struct.unpack('Q', data[off:off+8])[0]; off+=8+slen3
                        elif at in [0,1]: off+=1
                        elif at in [2,3,4,5,6]: off+=4
                        elif at in [10,11,12]: off+=8
            else:
                print(f'  [type={vtype}] {key}')
                if vtype == 8: slen = struct.unpack('Q', data[off:off+8])[0]; off+=8+slen
                elif vtype in [0,1]: off+=1
                elif vtype in [2,3,4,5,6]: off+=4
                elif vtype in [10,11,12]: off+=8
                elif vtype == 7: off+=1
                elif vtype == 9:
                    at2 = struct.unpack('I', data[off:off+4])[0]; off+=4
                    alen2 = struct.unpack('Q', data[off:off+8])[0]; off+=8
                    for j in range(int(alen2)):
                        if at2 == 8: slen4 = struct.unpack('Q', data[off:off+8])[0]; off+=8+slen4
                        elif at2 in [0,1]: off+=1
                        elif at2 in [2,3,4,5,6]: off+=4
                        elif at2 in [10,11,12]: off+=8
        print(f'KV section ends at offset: {off}')
        print()
