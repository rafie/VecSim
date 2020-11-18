ifeq ($(DEBUG),1)
	GCC_FLAGS=-o0 -g -DVALGRIND -fno-omit-frame-pointer
else
	GCC_FLAGS=-o2
endif

SOURCES=vector_similarity.c minmax_heap.c
HEADERS=redisai.h redisgears.h redismodule.h redisgears_memory.h minmax_heap.h

ARTIFACT_NAME=vector_similarity.so

all: $(ARTIFACT_NAME)

$(ARTIFACT_NAME): $(HEADERS) $(SOURCES)
	gcc $(GCC_FLAGS) $(SOURCES) -shared -fpic -DREDISMODULE_EXPERIMENTAL_API \
	-I../deps/OpenBLAS \
	../deps/OpenBLAS/libopenblas.a \
	-o $(ARTIFACT_NAME)

clean:
	rm $(ARTIFACT_NAME)
