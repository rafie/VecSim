#include "redisgears.h"
#include "redisai.h"
#include "minmax_heap.h"
#include <math.h>
#include "redisgears_memory.h"
#include <cblas.h>
#include <sys/time.h>

#define STR1(a) #a
#define STR(e) STR1(e)

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static RedisModuleCtx* staticCtx;

static RecordType* ScoreRecordType = NULL;
static RecordType* HeapRecordType = NULL;

#define VEC_SIZE 128

#define VEC_HOLDER_SIZE 1024 * 1024

typedef struct VecsHolder VecsHolder;

typedef struct VecDT{
    size_t index;
    VecsHolder* holder;
    RedisModuleString* keyName;
}VecDT;

typedef struct VecsHolder{
    size_t size;
    VecDT* vecDT[VEC_HOLDER_SIZE];
    float vecs[VEC_HOLDER_SIZE * VEC_SIZE];
}VecsHolder;

#define HOLDER_VECDT(h, i) (h->vecDT[i])
#define HOLDER_VEC(h, i) (h->vecs[i * VEC_SIZE])

VecsHolder** vecList = NULL;
RedisModuleType *vecRedisDT;

typedef struct VecReaderCtx{
    size_t index;
    Record** pendings;
    float vec[VEC_SIZE];
    size_t topK;
}VecReaderCtx;

typedef struct TopKArg{
    size_t topK;
}TopKArg;

typedef struct ScoreRecord{
    Record baseRecord;
    RedisModuleString* key;
    float score;
}ScoreRecord;

typedef struct HeapRecord{
    Record baseRecord;
    heap_t* heap;
}HeapRecord;

static VecReaderCtx* VecReaderCtx_Create(float* data, size_t topK){
    VecReaderCtx* ctx = RG_ALLOC(sizeof(*ctx));
    ctx->index = 0;
    ctx->pendings = array_new(Record*, 10);
    ctx->topK = topK;
    if(data){
        memcpy(ctx->vec, data, VEC_SIZE * sizeof(*data));
        float denom_vec = cblas_snrm2(VEC_SIZE, ctx->vec, 1);
        for(size_t i = 0 ; i < VEC_SIZE ; ++i){
            ctx->vec[i] /= denom_vec;
        }
    }
    return ctx;
}

static void VecReaderCtx_Free(VecReaderCtx* ctx){
    for(size_t i = 0 ; i < array_len(ctx->pendings) ; ++i){
        Record* r = ctx->pendings[i];
        RedisGears_FreeRecord(r);
    }

    array_free(ctx->pendings);

    RG_FREE(ctx);
}

static Record* to_score_records(ExecutionCtx* rctx, Record *data, void* arg){
    HeapRecord* hr = (HeapRecord*)data;

    Record* lr = RedisGears_ListRecordCreate(hr->heap->count);

    size_t count = hr->heap->count;
    for(size_t i = 0 ; i < count ; ++i){
        ScoreRecord* sr = mmh_pop_min(hr->heap);
        RedisGears_ListRecordAdd(lr, &sr->baseRecord);
    }

    RedisGears_FreeRecord(data);

    return lr;
}

static int heap_cmp(const void *a, const void *b, const void *udata){
    ScoreRecord* s1 = (ScoreRecord*)a;
    ScoreRecord* s2 = (ScoreRecord*)b;
    if(s1->score < s2->score){
        return -1;
    }else if(s1->score > s2->score){
        return 1;
    }else {
        return 0;
    }
}

static Record* top_k(ExecutionCtx* rctx, Record *accumulate, Record *r, void* arg){
    TopKArg* topKArg = arg;

    HeapRecord* heap = (HeapRecord*)accumulate;

    if(!heap){
        heap = (HeapRecord*)RedisGears_RecordCreate(HeapRecordType);
        heap->heap = mmh_init_with_size(topKArg->topK, heap_cmp, NULL, (mmh_free_func)RedisGears_FreeRecord);
    }

    heap_t* h = heap->heap;

    ScoreRecord* currSr = (ScoreRecord*)r;

    // If the queue is not full - we just push the result into it
    // If the pool size is 0 we always do that, letting the heap grow dynamically
    if (h->count < topKArg->topK) {
        mmh_insert(h, r);
    } else {
        // find the min result
        ScoreRecord* sr = mmh_peek_min(h);
        if(sr->score < currSr->score){
            void* temp = mmh_pop_min(h);
            RedisModule_Assert(temp == sr);
            RedisGears_FreeRecord(&(sr->baseRecord));
            mmh_insert(h, r);
        }else{
            RedisGears_FreeRecord(r);
        }
    }

    return &(heap->baseRecord);
}

static void on_done(ExecutionPlan* ctx, void* privateData){
    RedisModuleBlockedClient *bc = privateData;
    RedisModuleCtx *rctx = RedisModule_GetThreadSafeContext(bc);
    RedisGears_ReturnResultsAndErrors(ctx, rctx);
    RedisModule_UnblockClient(bc, NULL);
    RedisGears_DropExecution(ctx);
    RedisModule_FreeThreadSafeContext(rctx);
}

VecDT* vec_insert(RedisModuleString *keyName, const float* data){
    VecsHolder* holder = NULL;
    if(!vecList){
        vecList = array_new(VecsHolder*, 1);
    }
    if(array_len(vecList) == 0){
        holder = RG_CALLOC(1, sizeof(VecsHolder));
        vecList = array_append(vecList, holder);
    }else{
        holder = vecList[array_len(vecList) - 1];
    }

    if(holder->size >= VEC_HOLDER_SIZE){
        // we need to create a new holder
        holder = RG_CALLOC(1, sizeof(VecsHolder));
        vecList = array_append(vecList, holder);
    }

    float* v = &HOLDER_VEC(holder, holder->size);
    memcpy(v, data, sizeof(float) * VEC_SIZE);

    float demon = cblas_snrm2(VEC_SIZE, v, 1);

    for(size_t i = 0 ; i < VEC_SIZE ; ++i){
        v[i] /= demon;
    }

    VecDT* vDT = RG_CALLOC(1, sizeof(*vDT));
    vDT->holder = holder;
    vDT->index = holder->size;
    vDT->keyName = keyName;
    RedisModule_RetainString(NULL, vDT->keyName);
    HOLDER_VECDT(holder, holder->size) = vDT;

    ++holder->size;

    return vDT;
}

/*
 * rg.vec_add <k> <blob>
 */
int vec_add_command(RedisModuleCtx *ctx, RedisModuleString **argv, int argc){
    if(argc != 3){
        return RedisModule_WrongArity(ctx);
    }

    size_t dataLen;
    float* data = (float*)RedisModule_StringPtrLen(argv[2], &dataLen);
    if(dataLen != (VEC_SIZE * sizeof(float))){
        RedisModule_ReplyWithError(ctx, "Given blob is not float vector of size " STR(VEC_SIZE));
        return REDISMODULE_OK;
    }

    RedisModuleKey *kp = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_WRITE);
    if(RedisModule_KeyType(kp) != REDISMODULE_KEYTYPE_EMPTY){
        RedisModule_ReplyWithError(ctx, "Key is not empty");
        RedisModule_CloseKey(kp);
        return REDISMODULE_OK;
    }

    VecDT* vDT = vec_insert(argv[1], data);

    RedisModule_ModuleTypeSetValue(kp, vecRedisDT, vDT);

    RedisModule_CloseKey(kp);

    RedisModule_ReplicateVerbatim(ctx);

    RedisModule_ReplyWithSimpleString(ctx, "OK");

    return REDISMODULE_OK;
}

/*
 * rg.vec_sim <k> <blob>
 */
int vec_sim_command(RedisModuleCtx *ctx, RedisModuleString **argv, int argc){

    if(argc != 3){
        return RedisModule_WrongArity(ctx);
    }

    char* err = NULL;

    long long topK;
    if(RedisModule_StringToLongLong(argv[1], &topK) != REDISMODULE_OK){
        RedisModule_ReplyWithError(ctx, "Failed extracting <k>");
        return REDISMODULE_OK;
    }

    size_t dataSize;
    float* data = (float*)RedisModule_StringPtrLen(argv[2], &dataSize);
    if(dataSize != (VEC_SIZE * sizeof(float))){
        RedisModule_ReplyWithError(ctx, "Given blob is not at the right size");
        return REDISMODULE_OK;
    }

    TopKArg* topKArg2 = RG_ALLOC(sizeof(*topKArg2));
    topKArg2->topK = topK;


    FlatExecutionPlan* fep = RGM_CreateCtx(VecReader, &err);

    VecReaderCtx* rCtx = VecReaderCtx_Create(data, topK);

    RGM_Collect(fep);

    RGM_Accumulate(fep, top_k, topKArg2);

    RGM_FlatMap(fep, to_score_records, NULL);

    ExecutionPlan* ep = RGM_Run(fep, ExecutionModeAsync, rCtx, NULL, NULL, &err);

    if(!ep){
        RedisGears_FreeFlatExecution(fep);
        RedisModule_ReplyWithError(ctx, err);
        return REDISMODULE_OK;
    }

    RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL, NULL, 0);
    RedisGears_AddOnDoneCallback(ep, on_done, bc);
    RedisGears_FreeFlatExecution(fep);

    return REDISMODULE_OK;
}

static int ScoreRecord_SendReply(Record* base, RedisModuleCtx* rctx){
    ScoreRecord* sr = (ScoreRecord*)base;
    RedisModule_ReplyWithArray(rctx, 2);
    RedisModule_ReplyWithString(rctx, sr->key);
    RedisModule_ReplyWithDouble(rctx, sr->score);
    return REDISMODULE_OK;
}

static int ScoreRecord_RecordSerialize(ExecutionCtx* ctx, Gears_BufferWriter* bw, Record* base){
    ScoreRecord* sr = (ScoreRecord*)base;
    const char* keyStr = RedisModule_StringPtrLen(sr->key, NULL);
    RedisGears_BWWriteString(bw, keyStr);
    RedisGears_BWWriteBuffer(bw, (char*)(&(sr->score)), sizeof(sr->score));
    return REDISMODULE_OK;
}

static Record* ScoreRecord_RecordDeserialize(ExecutionCtx* ctx, Gears_BufferReader* br){
    ScoreRecord* sr = (ScoreRecord*)RedisGears_RecordCreate(ScoreRecordType);
    const char* keyStr = RedisGears_BRReadString(br);
    sr->key = RedisModule_CreateString(NULL, keyStr, strlen(keyStr));
    size_t len;
    char* data = RedisGears_BRReadBuffer(br, &len);
    RedisModule_Assert(len == sizeof(float));
    sr->score = *((float*)(data));

    return &sr->baseRecord;
}

static void ScoreRecord_RecordFree(Record* base){
    ScoreRecord* sr = (ScoreRecord*)base;
    if(sr->key){
        RedisModule_FreeString(NULL, sr->key);
    }

}

static int HeapRecord_SendReply(Record* base, RedisModuleCtx* rctx){
    RedisModule_Assert(false);
    return REDISMODULE_OK;
}

static int HeapRecord_Serialize(ExecutionCtx* ctx, Gears_BufferWriter* bw, Record* base){
    RedisModule_Assert(false);
    return REDISMODULE_OK;
}

static Record* HeapRecord_Deserialize(ExecutionCtx* ctx, Gears_BufferReader* br){
    RedisModule_Assert(false);
    return REDISMODULE_OK;
}

static void HeapRecord_Free(Record* base){
    HeapRecord* hr = (HeapRecord*)base;
    mmh_free(hr->heap);
}

static void TopKArg_ObjectFree(void* arg){
    RG_FREE(arg);
}

static void* TopKArg_ArgDuplicate(void* arg){
    TopKArg* topK = arg;
    TopKArg* topKDup = RG_ALLOC(sizeof(*topKDup));
    topKDup->topK = topK->topK;
    return topKDup;
}

static int TopKArg_ArgSerialize(FlatExecutionPlan* fep, void* arg, Gears_BufferWriter* bw, char** err){
    TopKArg* topK = arg;
    RedisGears_BWWriteLong(bw, topK->topK);
    return REDISMODULE_OK;
}

static void* TopKArg_ArgDeserialize(FlatExecutionPlan* fep, Gears_BufferReader* br, int version, char** err){
    TopKArg* topKDup = RG_ALLOC(sizeof(*topKDup));
    topKDup->topK = RedisGears_BRReadLong(br);
    return topKDup;
}

static char* TopKArg_ArgToString(void* arg){
    return RG_STRDUP("I am topk argument :)");
}

#define TopKTypeVersion 1

#define VS_PLUGIN_NAME "VECTOR_SIM"
#define REDISGEARSJVM_PLUGIN_VERSION 1

#define VEC_TYPE_VERSION 1

static void* VecDT_Load(RedisModuleIO *rdb, int encver){
    RedisModuleString *keyName = RedisModule_LoadString(rdb);
    size_t dataLen;
    float* data = (float*)RedisModule_LoadStringBuffer(rdb, &dataLen);
    RedisModule_Assert(dataLen == sizeof(float) * VEC_SIZE);

    VecDT* vDT = vec_insert(keyName, data);

    RedisModule_FreeString(NULL, keyName);
    RedisModule_Free(data);

    return vDT;
}

static void VecDT_Save(RedisModuleIO *rdb, void *value){
    VecDT* vDT = value;

    RedisModule_SaveString(rdb, vDT->keyName);
    RedisModule_SaveStringBuffer(rdb, (char*)&HOLDER_VEC(vDT->holder, vDT->index), sizeof(float) * VEC_SIZE);
}

static void VecDT_Free(void *value){
    VecDT* vDT = value;
    VecsHolder* holder = vDT->holder;
    size_t index = vDT->index;

    RedisModule_FreeString(NULL, vDT->keyName);
    RG_FREE(vDT);

    if(!holder){
        // we probably inside flush, the vector DT was detached and we can just return.
        return;
    }

    // get the last vector
    VecsHolder* lastVH = vecList[array_len(vecList) - 1];
    --lastVH->size;
    VecDT* lastVDT = HOLDER_VECDT(lastVH, lastVH->size);


    if(lastVDT != vDT){
        // swap last with current
        memmove(&HOLDER_VEC(holder, index), &HOLDER_VEC(lastVH, lastVH->size), VEC_SIZE * sizeof(float));

        HOLDER_VECDT(holder, index) = lastVDT;
        lastVDT->holder = holder;
        lastVDT->index = index;
    }

    if(lastVH->size == 0){
        // free the holder, it has no more data.
        RG_FREE(lastVH);
        if(array_len(vecList) > 1){
            vecList = array_trimm_cap(vecList, array_len(vecList) - 1);
        }else{
            array_free(vecList);
            vecList = NULL;
        }
    }
}

static float scores[VEC_HOLDER_SIZE];

static Record* VecReader_Next(ExecutionCtx* rctx, void* ctx){
//    struct timeval stop, start;
    if(!vecList){
        return NULL;
    }

    VecReaderCtx* readerCtx = ctx;
    RedisModuleCtx* redisCtx = RedisGears_GetRedisModuleCtx(rctx);
    if(array_len(readerCtx->pendings) > 0){
        return array_pop(readerCtx->pendings);
    }

    RedisGears_LockHanlderAcquire(redisCtx);

    const float* b1 = readerCtx->vec;

    while(readerCtx->index < array_len(vecList)){
        VecsHolder* holder = vecList[readerCtx->index++];

        cblas_sgemv(CblasRowMajor, CblasNoTrans, holder->size, VEC_SIZE, 1, holder->vecs, VEC_SIZE, b1, 1, 0, scores, 1);

        for(size_t i = 0 ; i < MIN(holder->size, readerCtx->topK) ; ++i){
            size_t index = cblas_isamax(holder->size, scores, 1);
            ScoreRecord* s = (ScoreRecord*)RedisGears_RecordCreate(ScoreRecordType);
            s->key = HOLDER_VECDT(holder, index)->keyName;
            RedisModule_RetainString(NULL, s->key);
            s->score = scores[index];
            readerCtx->pendings = array_append(readerCtx->pendings, &s->baseRecord);
            scores[index] = 0;
        }

        if(array_len(readerCtx->pendings) > 0){
            RedisGears_LockHanlderRelease(redisCtx);
            return array_pop(readerCtx->pendings);
        }
    }

    RedisGears_LockHanlderRelease(redisCtx);

    return NULL;
}

static void VecReader_Free(void* ctx){
    VecReaderCtx_Free(ctx);
}

static int VecReader_Serialize(ExecutionCtx* ectx, void* ctx, Gears_BufferWriter* bw){
    VecReaderCtx* readerCtx = ctx;
    RedisGears_BWWriteBuffer(bw, (char*)readerCtx->vec, VEC_SIZE * sizeof(float));
    RedisGears_BWWriteLong(bw, readerCtx->topK);
    return REDISMODULE_OK;
}

static int VecReader_Deserialize(ExecutionCtx* ectx, void* ctx, Gears_BufferReader* br){
    VecReaderCtx* readerCtx = ctx;
    size_t dataLen;
    float* data = (float*)RedisGears_BRReadBuffer(br, &dataLen);
    RedisModule_Assert(dataLen == VEC_SIZE * sizeof(*data));

    readerCtx->topK = RedisGears_BRReadLong(br);

    memcpy(readerCtx->vec, data, VEC_SIZE * sizeof(*data));

    return REDISMODULE_OK;
}

static Reader* VecReader_CreateReaderCallback(void* arg){
    VecReaderCtx* ctx = arg;
    if(!ctx){
        ctx = VecReaderCtx_Create(NULL, 0);
    }
    Reader* r = RG_ALLOC(sizeof(*r));
    *r = (Reader){
        .ctx = ctx,
        .next = VecReader_Next,
        .free = VecReader_Free,
        .serialize = VecReader_Serialize,
        .deserialize = VecReader_Deserialize,
    };
    return r;
}

RedisGears_ReaderCallbacks VecReader = {
        .create = VecReader_CreateReaderCallback,
};

static void OnFlush(struct RedisModuleCtx *ctx, RedisModuleEvent eid, uint64_t subevent, void *data){
    if(subevent != REDISMODULE_SUBEVENT_FLUSHDB_START){
        return;
    }

    if(!vecList){
        return;
    }

    // before flush we need to clean all the Vector Holders and disconnect the keys
    for(size_t i = 0 ; i < array_len(vecList) ; ++i){
        VecsHolder* holder = vecList[i];
        for(size_t j = 0 ; j < holder->size ; ++j){
            VecDT* vDT = HOLDER_VECDT(holder, j);
            vDT->holder = NULL;
        }
        RG_FREE(holder);
    }

    array_free(vecList);

    vecList = NULL;
}

int RedisGears_OnLoad(RedisModuleCtx *ctx) {
    openblas_set_num_threads(1);

    if(RedisGears_InitAsGearPlugin(ctx, VS_PLUGIN_NAME, REDISGEARSJVM_PLUGIN_VERSION) != REDISMODULE_OK){
        RedisModule_Log(ctx, "warning", "Failed initialize RedisGears API");
        return REDISMODULE_ERR;
    }

    RedisModule_Log(ctx, "warning", "OpenBlac num of threads: %d", openblas_get_num_threads());

    staticCtx = RedisModule_GetThreadSafeContext(NULL);

    RedisModuleTypeMethods vecDT = {
        .version = REDISMODULE_TYPE_METHOD_VERSION,
        .rdb_load = VecDT_Load,
        .rdb_save = VecDT_Save,
        .free = VecDT_Free,
    };

    vecRedisDT = RedisModule_CreateDataType(ctx, "vec_index", VEC_TYPE_VERSION, &vecDT);
    if (vecRedisDT == NULL) {
        RedisModule_Log(ctx, "error", "Could not create vector type");
        return REDISMODULE_ERR;
    }

    RGM_RegisterReader(VecReader);

    ScoreRecordType = RedisGears_RecordTypeCreate("ScoreRecord",
                                                   sizeof(ScoreRecord),
                                                   ScoreRecord_SendReply,
                                                   ScoreRecord_RecordSerialize,
                                                   ScoreRecord_RecordDeserialize,
                                                   ScoreRecord_RecordFree);

    HeapRecordType = RedisGears_RecordTypeCreate("HeapRecord",
                                                 sizeof(HeapRecord),
                                                 HeapRecord_SendReply,
                                                 HeapRecord_Serialize,
                                                 HeapRecord_Deserialize,
                                                 HeapRecord_Free);

    ArgType* TopKType = RedisGears_CreateType("TopKType",
                                              TopKTypeVersion,
                                              TopKArg_ObjectFree,
                                              TopKArg_ArgDuplicate,
                                              TopKArg_ArgSerialize,
                                              TopKArg_ArgDeserialize,
                                              TopKArg_ArgToString);

    RGM_RegisterMap(to_score_records, NULL);
    RGM_RegisterAccumulator(top_k, TopKType);

    if (RedisModule_CreateCommand(ctx, "rg.vec_sim", vec_sim_command, "readonly", 0, 0, 0) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "warning", "could not register command rg.vec_sim");
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "rg.vec_add", vec_add_command, "write deny-oom", 1, 1, 1) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "warning", "could not register command rg.vec_add");
        return REDISMODULE_ERR;
    }

    RedisModule_SubscribeToServerEvent(ctx, RedisModuleEvent_FlushDB, OnFlush);

    return REDISMODULE_OK;
}
