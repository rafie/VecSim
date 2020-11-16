#include "redisgears.h"
#include "redisai.h"
#include "minmax_heap.h"
#include <math.h>
#include "redisgears_memory.h"

static RecordType* tensorRecordType = NULL;
static RecordType* ScoreRecordType = NULL;
static RecordType* HeapRecordType = NULL;

static bool RedisAI_Initialized = false;

typedef struct DistArg{
    size_t size;
    float* data;
}DistArg;

typedef struct TopKArg{
    size_t topK;
}TopKArg;

typedef struct TensorRecord{
    Record baseRecord;
    char* key;
    RAI_Tensor* tensor;
}TensorRecord;

typedef struct ScoreRecord{
    Record baseRecord;
    char* key;
    double score;
}ScoreRecord;

typedef struct HeapRecord{
    Record baseRecord;
    heap_t* heap;
}HeapRecord;

static Record* read_vector(RedisModuleCtx* rctx, RedisModuleString* key, RedisModuleKey* keyPtr, bool readValue, const char* event){
    TensorRecord* t = (TensorRecord*)RedisGears_RecordCreate(tensorRecordType);
    const char* keyStr = RedisModule_StringPtrLen(key, NULL);
    t->key = RG_STRDUP(keyStr);
    t->tensor = NULL;

    if(!keyPtr){
        RedisModule_Log(rctx, "warning", "Got NULL key pointer");
        return &(t->baseRecord);
    }

    if(RedisModule_KeyType(keyPtr) != REDISMODULE_KEYTYPE_MODULE){
        RedisModule_Log(rctx, "warning", "Got none module key");
        return &(t->baseRecord);
    }

    if(RedisModule_ModuleTypeGetType(keyPtr) != RedisAI_TensorRedisType()){
        RedisModule_Log(rctx, "warning", "Got none RedisAI tensor key");
        return &(t->baseRecord);
    }

    t->tensor = RedisModule_ModuleTypeGetValue(keyPtr);
    t->tensor = RedisAI_TensorGetShallowCopy(t->tensor);

    return &(t->baseRecord);
}

static Record* calculate_dist(ExecutionCtx* rctx, Record *data, void* arg){
    DistArg* blob = arg;
    size_t b1Len = blob->size;
    float* b1 = blob->data;
    TensorRecord* t = (TensorRecord*)data;
    if(t->tensor == NULL){
        RedisGears_SetError(rctx, RG_STRDUP("not a tensor key"));
        return NULL;
    }
    size_t b2Len = RedisAI_TensorLength(t->tensor);
    float* b2 = (float*)RedisAI_TensorData(t->tensor);

    if(b2Len != b1Len){
        RedisGears_SetError(rctx, RG_STRDUP("vector sized missmatch"));
        return NULL;
    }

//    float denom_a = cblas_snrm2(b1Len, b1, 1);
//    float denom_b = cblas_snrm2(b2Len, b2, 1);
//    float dot = cblas_dsdot(b1Len, b1, 1, b2, 1);
//
//    double score = dot / (sqrt(denom_a) * sqrt(denom_b)) ;

    double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int i = 0u; i < b2Len; ++i) {
        dot += b1[i] * b2[i] ;
        denom_a += b1[i] * b1[i] ;
        denom_b += b2[i] * b2[i] ;
    }

    double score = dot / (sqrt(denom_a) * sqrt(denom_b)) ;

    ScoreRecord* s = (ScoreRecord*)RedisGears_RecordCreate(ScoreRecordType);
    s->key = t->key;
    s->score = score;

    t->key = NULL;

    RedisGears_FreeRecord(&(t->baseRecord));

    return &(s->baseRecord);;
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
    if(s1->score > s2->score){
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
        if(sr->score > currSr->score){
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
 * rg.vec_sim <k> <prefix> <blob>
 */
int vec_sim(RedisModuleCtx *ctx, RedisModuleString **argv, int argc){

    if(argc != 4){
        return RedisModule_WrongArity(ctx);
    }

    if(!RedisAI_Initialized){
        if(RedisAI_Initialize(ctx) != REDISMODULE_OK){
            RedisModule_ReplyWithError(ctx, "RedisAI is not initialized");
            return REDISMODULE_OK;
        }else{
            RedisModule_Log(ctx, "notice", "RedisAI api loaded successfully for vec sim calculations.");
            RedisAI_Initialized = true;
        }
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

    DistArg* dArg = RG_ALLOC(sizeof(*dArg));
    float* data = (float*)RedisModule_StringPtrLen(argv[3], &(dArg->size));
    if((dArg->size % sizeof(float)) != 0){
        RG_FREE(dArg);
        RedisModule_ReplyWithError(ctx, "Given blob is not devided by 4");
        return REDISMODULE_OK;
    }
    dArg->data = RG_ALLOC(dArg->size);
    memcpy(dArg->data, data, dArg->size);
    dArg->size /= sizeof(float);

    TopKArg* topKArg1 = RG_ALLOC(sizeof(*topKArg1));
    topKArg1->topK = topK;

    TopKArg* topKArg2 = RG_ALLOC(sizeof(*topKArg2));
    topKArg2->topK = topK;


    FlatExecutionPlan* fep = RGM_CreateCtx(KeysReader, &err);

    KeysReaderCtx* rCtx = RedisGears_KeysReaderCtxCreate(p, false, NULL, noscan);

    RGM_KeysReaderSetReadRecordCallback(rCtx, read_vector);

    RGM_Map(fep, calculate_dist, dArg);

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
        RG_FREE(t->key);
    }
}

static int ScoreRecord_SendReply(Record* base, RedisModuleCtx* rctx){
    ScoreRecord* sr = (ScoreRecord*)base;
    RedisModule_ReplyWithArray(rctx, 2);
    RedisModule_ReplyWithCString(rctx, sr->key);
    RedisModule_ReplyWithDouble(rctx, sr->score);
    return REDISMODULE_OK;
}

static int ScoreRecord_RecordSerialize(ExecutionCtx* ctx, Gears_BufferWriter* bw, Record* base){
    ScoreRecord* sr = (ScoreRecord*)base;
    RedisGears_BWWriteString(bw, sr->key);
    RedisGears_BWWriteBuffer(bw, (char*)(&(sr->score)), sizeof(sr->score));
    return REDISMODULE_OK;
}

static Record* ScoreRecord_RecordDeserialize(ExecutionCtx* ctx, Gears_BufferReader* br){
    ScoreRecord* sr = (ScoreRecord*)RedisGears_RecordCreate(ScoreRecordType);
    sr->key = RedisGears_BRReadString(br);
    sr->key = RG_STRDUP(sr->key);
    size_t len;
    char* data = RedisGears_BRReadBuffer(br, &len);
    RedisModule_Assert(len == sizeof(double));
    sr->score = *((double*)(data));

    return REDISMODULE_OK;
}

static void ScoreRecord_RecordFree(Record* base){
    ScoreRecord* sr = (ScoreRecord*)base;
    if(sr->key){
        RG_FREE(sr->key);
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

static void InputTensor_ObjectFree(void* arg){
    DistArg* blob = arg;
    RG_FREE(blob->data);
    RG_FREE(blob);
}

static void* InputTensor_ArgDuplicate(void* arg){
    DistArg* blob = arg;
    DistArg* dArg = RG_ALLOC(sizeof(*dArg));
    dArg->data = RG_ALLOC(blob->size * sizeof(float));
    memcpy(dArg->data, blob->data, blob->size * sizeof(float));
    dArg->size = blob->size;
    return dArg;
}

static int InputTensor_ArgSerialize(FlatExecutionPlan* fep, void* arg, Gears_BufferWriter* bw, char** err){
    DistArg* blob = arg;
    RedisGears_BWWriteBuffer(bw, (char*)(blob->data), blob->size * sizeof(float));
    return REDISMODULE_OK;
}

static void* InputTensor_ArgDeserialize(FlatExecutionPlan* fep, Gears_BufferReader* br, int version, char** err){
    DistArg* dArg = RG_ALLOC(sizeof(*dArg));

    size_t dLen;
    float* data = (float*)RedisGears_BRReadBuffer(br, &dLen);
    dArg->data = RG_ALLOC(dLen);
    memcpy(dArg->data, data, dLen);
    dArg->size = dLen / sizeof(float);

    return dArg;
}

static char* InputTensor_ArgToString(void* arg){
    return RG_STRDUP("I am dist argument :)");
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

int RedisGears_OnLoad(RedisModuleCtx *ctx) {
    if(RedisGears_InitAsGearPlugin(ctx, VS_PLUGIN_NAME, REDISGEARSJVM_PLUGIN_VERSION) != REDISMODULE_OK){
        RedisModule_Log(ctx, "warning", "Failed initialize RedisGears API");
        return REDISMODULE_ERR;
    }

    if(RedisAI_Initialize(ctx) == REDISMODULE_OK){
        RedisModule_Log(ctx, "notice", "RedisAI api loaded successfully for vec sim calculations.");
        RedisAI_Initialized = true;
    }

    RGM_KeysReaderRegisterReadRecordCallback(read_vector);

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

    ArgType* InputTensorType = RedisGears_CreateType("InputTensorType",
                                                     InputTensorTypeVersion,
                                                     InputTensor_ObjectFree,
                                                     InputTensor_ArgDuplicate,
                                                     InputTensor_ArgSerialize,
                                                     InputTensor_ArgDeserialize,
                                                     InputTensor_ArgToString);

    ArgType* TopKType = RedisGears_CreateType("TopKType",
                                              TopKTypeVersion,
                                              TopKArg_ObjectFree,
                                              TopKArg_ArgDuplicate,
                                              TopKArg_ArgSerialize,
                                              TopKArg_ArgDeserialize,
                                              TopKArg_ArgToString);

    RGM_RegisterMap(to_score_records, NULL);
    RGM_RegisterMap(calculate_dist, InputTensorType);
    RGM_RegisterAccumulator(top_k, TopKType);

    if (RedisModule_CreateCommand(ctx, "rg.vec_sim", vec_sim, "readonly", 0, 0, 0) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "warning", "could not register command rg.vec_sim");
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}
