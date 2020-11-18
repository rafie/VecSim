#include "redisgears.h"
#include "redisai.h"
#include "minmax_heap.h"
#include <math.h>
#include "redisgears_memory.h"
#include <cblas.h>

static RecordType* tensorRecordType = NULL;
static RecordType* ScoreRecordType = NULL;
static RecordType* HeapRecordType = NULL;

#define VEC_SIZE 128

typedef struct Vec{
    RedisModuleString* keyName;
    float denom;
    float vec[VEC_SIZE];
}Vec;

#define VEC_HOLDER_SIZE 512

typedef struct VecsHolder{
    size_t size;
    size_t vecListPos;
    Vec vectors[VEC_HOLDER_SIZE];
}VecsHolder;


VecsHolder** vecList = NULL;
RedisModuleType *vecRedisDT;

typedef struct VecDT{
    size_t index;
    VecsHolder* holder;
}VecDT;

typedef struct VecReaderCtx{
    size_t index;
    Record** pendings;
    float vec[VEC_SIZE];
    float denom_vec;
}VecReaderCtx;

typedef struct TopKArg{
    size_t topK;
}TopKArg;

typedef struct TensorRecord{
    Record baseRecord;
    RedisModuleString* key;
    RAI_Tensor* tensor;
}TensorRecord;

typedef struct ScoreRecord{
    Record baseRecord;
    RedisModuleString* key;
    double score;
}ScoreRecord;

typedef struct HeapRecord{
    Record baseRecord;
    heap_t* heap;
}HeapRecord;

static VecReaderCtx* VecReaderCtx_Create(float* data){
    VecReaderCtx* ctx = RG_ALLOC(sizeof(*ctx));
    ctx->index = 0;
    ctx->pendings = array_new(Record*, 10);
    if(data){
        memcpy(ctx->vec, data, VEC_SIZE * sizeof(*data));
        ctx->denom_vec = cblas_snrm2(VEC_SIZE, ctx->vec, 1);
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

/*
 * rg.vec_add <k> <blob>
 */
int vec_add(RedisModuleCtx *ctx, RedisModuleString **argv, int argc){
    if(argc != 3){
        return RedisModule_WrongArity(ctx);
    }

    size_t dataLen;
    float* data = (float*)RedisModule_StringPtrLen(argv[2], &dataLen);
    if(dataLen != (VEC_SIZE * sizeof(float))){
        RedisModule_ReplyWithError(ctx, "Given blob is not float vector of size 128");
        return REDISMODULE_OK;
    }

    RedisModuleKey *kp = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_WRITE);
    if(RedisModule_KeyType(kp) != REDISMODULE_KEYTYPE_EMPTY){
        RedisModule_ReplyWithError(ctx, "Key is not empty");
        RedisModule_CloseKey(kp);
        return REDISMODULE_OK;
    }

    VecsHolder* holder = NULL;
    if(array_len(vecList) == 0){
        holder = RG_CALLOC(1, sizeof(VecsHolder));
        vecList = array_append(vecList, holder);
        holder->vecListPos = 0;
    }else{
        holder = vecList[array_len(vecList) - 1];
    }

    if(holder->size >= VEC_HOLDER_SIZE){
        // we need to create a new holder
        holder = RG_CALLOC(1, sizeof(VecsHolder));
        vecList = array_append(vecList, holder);
        holder->vecListPos = array_len(vecList) - 1;
    }

    Vec* v = holder->vectors + holder->size;

    v->keyName = argv[1];
    RedisModule_RetainString(NULL, v->keyName);

    memcpy(v->vec, data, dataLen);

    v->denom = cblas_snrm2(VEC_SIZE, v->vec, 1);

    VecDT* vDT = RG_CALLOC(1, sizeof(*vDT));
    vDT->holder = holder;
    vDT->index = holder->size;

    ++holder->size;

    RedisModule_ModuleTypeSetValue(kp, vecRedisDT, vDT);

    RedisModule_CloseKey(kp);

    RedisModule_ReplicateVerbatim(ctx);

    RedisModule_ReplyWithSimpleString(ctx, "OK");

    return REDISMODULE_OK;
}

/*
 * rg.vec_sim <k> <prefix> <blob>
 */
int vec_sim(RedisModuleCtx *ctx, RedisModuleString **argv, int argc){

    if(argc != 4){
        return RedisModule_WrongArity(ctx);
    }

    char* err = NULL;

    long long topK;
    if(RedisModule_StringToLongLong(argv[1], &topK) != REDISMODULE_OK){
        RedisModule_ReplyWithError(ctx, "Failed extracting <k>");
        return REDISMODULE_OK;
    }

    size_t prefixLen;
    const char* p = RedisModule_StringPtrLen(argv[2], &prefixLen);
    if(prefixLen == 0 ){
        RedisModule_ReplyWithError(ctx, "Failed extracting <prefix>");
        return REDISMODULE_OK;
    }

    bool noscan = true;
    if(p[prefixLen - 1] == '*'){
        noscan = false;
    }

    size_t dataSize;
    float* data = (float*)RedisModule_StringPtrLen(argv[3], &dataSize);
    if(dataSize != (VEC_SIZE * sizeof(float))){
        RedisModule_ReplyWithError(ctx, "Given blob is not at the right size");
        return REDISMODULE_OK;
    }

    TopKArg* topKArg1 = RG_ALLOC(sizeof(*topKArg1));
    topKArg1->topK = topK;

    TopKArg* topKArg2 = RG_ALLOC(sizeof(*topKArg2));
    topKArg2->topK = topK;


    FlatExecutionPlan* fep = RGM_CreateCtx(VecReader, &err);

    VecReaderCtx* rCtx = VecReaderCtx_Create(data);

    RGM_Accumulate(fep, top_k, topKArg1);

    RGM_FlatMap(fep, to_score_records, NULL);

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

static int TensorRecord_SendReply(Record* base, RedisModuleCtx* rctx){
    RedisModule_Assert(false);
    return REDISMODULE_OK;
}

static int TensorRecord_RecordSerialize(ExecutionCtx* ctx, Gears_BufferWriter* bw, Record* base){
    RedisModule_Assert(false);
    return REDISMODULE_OK;
}

static Record* TensorRecord_RecordDeserialize(ExecutionCtx* ctx, Gears_BufferReader* br){
    RedisModule_Assert(false);
    return REDISMODULE_OK;
}

static void TensorRecord_RecordFree(Record* base){
    TensorRecord* t = (TensorRecord*)base;
    RedisAI_TensorFree(t->tensor);
    if(t->key){
        RedisModule_FreeString(NULL, t->key);
    }
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
    RedisModule_Assert(len == sizeof(double));
    sr->score = *((double*)(data));

    return REDISMODULE_OK;
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

#define InputTensorTypeVersion 1
#define TopKTypeVersion 1

#define VS_PLUGIN_NAME "VECTOR_SIM"
#define REDISGEARSJVM_PLUGIN_VERSION 1

#define VEC_TYPE_VERSION 1

static void* VecDT_Load(RedisModuleIO *rdb, int encver){

}

static void VecDT_Save(RedisModuleIO *rdb, void *value){

}

static void VecDT_Free(void *value){
//    VecDT* vDT = value;
//    Vec* vec = vDT->holder->vectors + vDT->index;
//
//    RedisModule_FreeString(NULL, vec->keyName);
//
//    // get the last vector
//    VecsHolder* lastVH = vecList[array_len(vecList) - 1];
//    Vec* lastVec = lastVH->vectors + lastVH->size;
//
//    if(vec == lastVec){
//        // we are deleting the last vec, we just need to reduce the holder size
//        --lastVH->size;
//    }else{
//        vec->vec = lastVec->vec;
//        vec->keyName = lastVec->keyName;
//    }

}

static Record* VecReader_Next(ExecutionCtx* rctx, void* ctx){
    VecReaderCtx* readerCtx = ctx;
    RedisModuleCtx* redisCtx = RedisGears_GetRedisModuleCtx(rctx);
    if(array_len(readerCtx->pendings) > 0){
        return array_pop(readerCtx->pendings);
    }

    RedisGears_LockHanlderAcquire(redisCtx);

    const float* b1 = readerCtx->vec;
    float denom_a = readerCtx->denom_vec;

    while(readerCtx->index < array_len(vecList)){
        VecsHolder* holder = vecList[readerCtx->index++];

        for(size_t i = 0 ; i < holder->size ; ++i){
            Vec* v = holder->vectors + i;
            const float* b2 = v->vec;
            float dot = cblas_dsdot(VEC_SIZE, b1, 1, b2, 1);

            double score = dot / (denom_a * v->denom) ;
            ScoreRecord* s = (ScoreRecord*)RedisGears_RecordCreate(ScoreRecordType);
            s->key = v->keyName;
            RedisModule_RetainString(NULL, s->key);
            s->score = score;

            readerCtx->pendings = array_append(readerCtx->pendings, &s->baseRecord);
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
    RedisGears_BWWriteBuffer(bw, (char*)readerCtx->vec, VEC_SIZE);
    return REDISMODULE_OK;
}

static int VecReader_Deserialize(ExecutionCtx* ectx, void* ctx, Gears_BufferReader* br){
    VecReaderCtx* readerCtx = ctx;
    size_t dataLen;
    float* data = (float*)RedisGears_BRReadBuffer(br, &dataLen);
    RedisModule_Assert(dataLen == VEC_SIZE * sizeof(*data));

    memcpy(readerCtx->vec, data, VEC_SIZE * sizeof(*data));
    readerCtx->denom_vec = cblas_snrm2(VEC_SIZE, readerCtx->vec, 1);

    return REDISMODULE_OK;
}

static Reader* VecReader_CreateReaderCallback(void* arg){
    VecReaderCtx* ctx = arg;
    if(!ctx){
        ctx = VecReaderCtx_Create(NULL);
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

int RedisGears_OnLoad(RedisModuleCtx *ctx) {
    if(RedisGears_InitAsGearPlugin(ctx, VS_PLUGIN_NAME, REDISGEARSJVM_PLUGIN_VERSION) != REDISMODULE_OK){
        RedisModule_Log(ctx, "warning", "Failed initialize RedisGears API");
        return REDISMODULE_ERR;
    }

    vecList = array_new(VecsHolder*, 10);
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

//    RGM_KeysReaderRegisterReadRecordCallback(read_vector);
    RGM_RegisterReader(VecReader);

    tensorRecordType = RedisGears_RecordTypeCreate("TensorRecord",
                                                   sizeof(TensorRecord),
                                                   TensorRecord_SendReply,
                                                   TensorRecord_RecordSerialize,
                                                   TensorRecord_RecordDeserialize,
                                                   TensorRecord_RecordFree);

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

    if (RedisModule_CreateCommand(ctx, "rg.vec_sim", vec_sim, "readonly", 0, 0, 0) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "warning", "could not register command rg.vec_sim");
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "rg.vec_add", vec_add, "readonly", 1, 1, 1) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "warning", "could not register command rg.vec_add");
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}
