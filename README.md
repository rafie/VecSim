# VecSim
Calculate vector similarity over Redis cluster

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

* Install [Redis 6.0.1 or higher](https://redis.io/) on your machine.

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