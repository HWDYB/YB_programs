import nncase_runtime as nn
import gc
kpu = nn.kpu()
kpu.load_kmodel("/sdcard/best.kmodel")
for i in range(kpu.inputs_size()):
    print("in ", kpu.inputs_desc(i))
for i in range(kpu.outputs_size()):
    print("out", kpu.outputs_desc(i))
print("OK")
del kpu
gc.collect()
nn.shrink_memory_pool()
