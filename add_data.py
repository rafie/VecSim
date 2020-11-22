import redis
import numpy as np
conn = redis.Redis()
p = conn.pipeline(transaction=False)
for i in range(10000000):
    p.execute_command('RG.VEC_ADD', 'key%i' % i, np.random.rand(1, 128).astype(np.float32).tobytes())
    #p.execute_command('AI.TENSORSET', 'key%i' % i, "FLOAT", "1", "128", "BLOB", np.random.rand(1, 128).astype(np.float32).tobytes())
    if(i % 100 == 0):
        p.execute()
        p = conn.pipeline(transaction=False)
p.execute()

