ifeq ($(DEBUG),1)
	GCC_FLAGS=-o0 -g -DVALGRIND -fno-omit-frame-pointer
else
	GCC_FLAGS=-o2
endif

all:
	gcc $(GCC_FLAGS) vector_similarity.c minmax_heap.c -shared -fpic -DREDISMODULE_EXPERIMENTAL_API \
	-o vector_similarity.so