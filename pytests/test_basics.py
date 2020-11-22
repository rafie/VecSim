from common import DecoratorTest, decodeStr
import numpy as np
from scipy import spatial

@DecoratorTest
def test_basic(env, conn):
	targetVector = np.random.rand(1, 128).astype(np.float32)
	vectors = []
	for i in range(1000):
		vectors.append(('key%d' % i, np.random.rand(1, 128).astype(np.float32)))

	# setting the data into redis
	for v in vectors:
		conn.execute_command('RG.VEC_ADD', v[0], v[1].tobytes())

	# calculating dist
	dists = [(1 - spatial.distance.cosine(targetVector[0], v[0]), k) for k, v in vectors]
	dists = sorted(dists)

	keys = [k for _, k in dists[-4:]]

	keys = sorted(keys)

	redisKeys = conn.execute_command('RG.VEC_SIM', '4', targetVector.tobytes())

	redisKeys = sorted([decodeStr(k) for k, _ in redisKeys[0]])

	env.assertEqual(keys, redisKeys)

@DecoratorTest
def test_delete(env, conn):
	targetVector = np.random.rand(1, 128).astype(np.float32)
	vectors = []
	for i in range(1000000):
		vectors.append(('key%d' % i, np.random.rand(1, 128).astype(np.float32)))

	# setting the data into redis
	i = 0
	p = conn.pipeline(transaction=False)
	for v in vectors:
		p.execute_command('RG.VEC_ADD', v[0], v[1].tobytes())
		i += 1
		if i % 100 == 0:
			p.execute()
			p = conn.pipeline(transaction=False)
	p.execute()

	#d delete hald of the data
	for v in [vectors[i] for i in range(0, len(vectors), 2)]:
		conn.execute_command('del', v[0])		

	vectors = [vectors[i] for i in range(1, len(vectors), 2)]

	env.expect('RG.PYEXECUTE', "GB('ShardsIDReader').map(lambda x: int(execute('dbsize'))).aggregate(0, lambda a, x: a + x, lambda a, x: a + x).run()").equal([[str(len(vectors)).encode()],[]])

	# calculating dist
	dists = [(1 - spatial.distance.cosine(targetVector[0], v[0]), k) for k, v in vectors]
	dists = sorted(dists)

	keys = [k for _, k in dists[-80:]]

	keys = sorted(keys)

	redisKeys = conn.execute_command('RG.VEC_SIM', '80', targetVector.tobytes())

	redisKeys = sorted([decodeStr(k) for k, _ in redisKeys[0]])

	env.assertEqual(keys, redisKeys)

@DecoratorTest
def test_flush(env, conn):
	# flush empty db, make sure not crash.
	conn.flushall()

	targetVector = np.random.rand(1, 128).astype(np.float32)
	vectors = []
	for i in range(1000):
		vectors.append(('key%d' % i, np.random.rand(1, 128).astype(np.float32)))

	# setting the data into redis
	for v in vectors:
		conn.execute_command('RG.VEC_ADD', v[0], v[1].tobytes())

	conn.flushall()

	env.expect('RG.PYEXECUTE', "GB('ShardsIDReader').map(lambda x: int(execute('dbsize'))).aggregate(0, lambda a, x: a + x, lambda a, x: a + x).run()").equal([[b'0'],[]])

@DecoratorTest
def test_rdbLoadAndSave(env, conn):
	targetVector = np.random.rand(1, 128).astype(np.float32)
	vectors = []
	for i in range(1000000):
		vectors.append(('key%d' % i, np.random.rand(1, 128).astype(np.float32)))

	# setting the data into redis
	i = 0
	p = conn.pipeline(transaction=False)
	for v in vectors:
		p.execute_command('RG.VEC_ADD', v[0], v[1].tobytes())
		i += 1
		if i % 100 == 0:
			p.execute()
			p = conn.pipeline(transaction=False)
	p.execute()

	# calculating dist
	dists = [(1 - spatial.distance.cosine(targetVector[0], v[0]), k) for k, v in vectors]
	dists = sorted(dists)

	keys = [k for _, k in dists[-4:]]

	keys = sorted(keys)

	for _ in env.reloading_iterator():
		
		redisKeys = conn.execute_command('RG.VEC_SIM', '4', targetVector.tobytes())

		redisKeys = sorted([decodeStr(k) for k, _ in redisKeys[0]])

		env.assertEqual(keys, redisKeys)

@DecoratorTest
def test_dumpRestor(env, conn):
	env.skipOnCluster()
	vec = np.random.rand(1, 128).astype(np.float32)
	conn.execute_command('RG.VEC_ADD', 'key', vec.tobytes())

	d = conn.execute_command('DUMP', 'key')

	conn.execute_command('RESTORE', 'key', '0', d, 'REPLACE')

	res = conn.execute_command('RG.VEC_SIM', '4', vec.tobytes())[0]

	env.assertEqual(len(res), 1)
	env.assertEqual(decodeStr(res[0][0]), 'key')
	env.assertLessEqual(1 - float(res[0][1]), 0.00001)

@DecoratorTest
def test_vectorWrongSize(env, conn):
	env.skipOnCluster()
	vec = np.random.rand(1, 129).astype(np.float32)
	env.expect('RG.VEC_ADD', 'key', vec.tobytes()).error().contains('not float vector of size')
