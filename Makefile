OS=$(shell ./deps/readies/bin/platform --osnick)
$(info OS=$(OS))

all: OpenBLAS VecSim

./deps/OpenBLAS/libopenblas.a:
	make -C ./deps/OpenBLAS

OpenBLAS: ./deps/OpenBLAS/libopenblas.a

VecSim:
	make -C ./src/

clean:
	make -C ./deps/OpenBLAS clean
	make -C ./src/ clean
	
InstallRedisGears:
	OS=$(OS) /bin/bash ./Install_RedisGears.sh

Install: all InstallRedisGears

Run: Install
	redis-server --loadmodule ./bin/RedisGears/redisgears.so CreateVenv 1 pythonInstallationDir ./bin/RedisGears/ \
	PluginsDirectory ./src/
	
Tests: Install
	python3 -m RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs
	python3 -m RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs --env oss-cluster --shards-count 1
	python3 -m RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs --env oss-cluster --shards-count 2
	python3 -m RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs --env oss-cluster --shards-count 3
