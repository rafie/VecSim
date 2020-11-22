# VecSim
Calculate vector similarity over Redis cluster (using [RedisGears](https://oss.redislabs.com/redisgears/)). The distance calculation is based on [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

# Build and Run
## Prerequisites
* Install git

for Debian like systems:
```
apt-get install git
```
for Fedora like systems:
```
yum install git
```


* Install the build-essential package, or its equivalent, on your system:

for Debian-like systems:
```
apt-get install build-essential
```
for Fedora-like systems:
```
yum install -y centos-release-scl
yum install -y devtoolset-8
scl enable devtoolset-8 bash
```

* Install [Redis 6.0.9 or higher](https://redis.io/) on your machine.

```
git clone https://github.com/redis/redis.git
cd redis
git checkout 6.0.9
make
make install
```

* On macOS install Xcode command line tools:


```
xcode-select --install
```

## Build and Run
### Clone
To get the code and its submodules, do the following:
```
git clone https://github.com/RedisGears/VecSim.git
cd VecSim
git submodule update --init --recursive
```

### Build and Run
Inside the VecSim directory run `make Run` to build and Run

**Important:** `make Run` will download a compiled version of RedisGears so an internet connection is required.

## Testing
### Prerequisites
* Python3
* RLTest - `python3 -m pip install --no-cache-dir git+https://github.com/RedisLabsModules/RLTest.git@master`
* numpy - `python3 -m pip install --no-cache-dir numpy`
* scipy - `python3 -m pip install --no-cache-dir scipy`

### Run the tests
Inside the VecSim directory run `make Tests`

**Important:** `make Tests` will download a compiled version of RedisGears so an internet connection is required.

# API
## RG.VEC_ADD
This command is used to add a new vector to Redis
### Redis API
```
RG.VEC_ADD <key> <vector>
```
Arguments:

* key - the key to put the vector in
* vector - byte representation of float vector of size 128

Example (using redis-py client):
```Python
import redis
import numpy as np
conn = redis.Redis()
conn.execute_command('RG.VEC_ADD', 'key', np.random.rand(1, 128).astype(np.float32).tobytes())
```
## RG.VEC_SIM
This command is used to return the k closest vector of a give vector (using [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity))
### Redis API
```
RG.VEC_ADD <k> <vector>
```
Arguments:

* k - the amount of vectors to return
* vector - byte representation of float vector of size 128

Example (using redis-py client):
```Python
import redis
import numpy as np
blob = np.random.rand(1, 128).astype(np.float32)
res = r.execute_command('RG.VEC_SIM', '4', blob.tobytes()) # return the 4 closest vectors to blob
```
