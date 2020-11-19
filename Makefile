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
	/bin/bash ./Install_RedisGears.sh

Install: all InstallRedisGears

Run: Install
	redis-server --loadmodule ./bin/RedisGears/redisgears.so CreateVenv 1 pythonInstallationDir ./bin/RedisGears/ \
	PluginsDirectory ./src/
	
Tests: Install
	RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs
	RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs --env oss-cluster --shards-count 1
	RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs --env oss-cluster --shards-count 2
	RLTest -t ./pytests --module ./bin/RedisGears/redisgears.so --module-args "CreateVenv 1 pythonInstallationDir ../bin/RedisGears/ \
	PluginsDirectory ../src/" --clear-logs --env oss-cluster --shards-count 3
