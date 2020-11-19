import redis
import time
import numpy as np
r = redis.Redis()
blob = np.random.rand(1, 128).astype(np.float32)
start = time.time()
res = r.execute_command('RG.VEC_SIM', '4', '*', blob.tobytes())
end = time.time()
for r in res[0]:
    print(r)
print('took : %s' % (str(end - start)))

