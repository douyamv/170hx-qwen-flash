#include <atomic>
#include <cupti.h>
#include <cupti_activity.h>
#include <pthread.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
static FILE * output=nullptr;
static std::atomic<bool> collecting{false};
static const char * flag_path=nullptr;
static const char * log_path=nullptr;
static void quoted(const char * s){fputc('"',output);if(s)for(;*s;s++){if(*s=='"')fputc('"',output);fputc(*s=='\n'?' ':*s,output);}fputc('"',output);}
static void marker(const char * text){uint64_t ts=0;cuptiGetTimestamp(&ts);pthread_mutex_lock(&mutex);if(output){fprintf(output,"MARK,-1,%llu,%llu,0,0,0,0,",(unsigned long long)ts,(unsigned long long)ts);quoted(text);fputc('\n',output);fflush(output);}pthread_mutex_unlock(&mutex);}
static void CUPTIAPI request_buffer(uint8_t ** buffer,size_t * size,size_t * max_records){*size=4*1024*1024;*max_records=0;void * p=nullptr;if(posix_memalign(&p,8,*size))p=nullptr;*buffer=(uint8_t*)p;}
static void CUPTIAPI complete_buffer(CUcontext ctx,uint32_t stream,uint8_t * buffer,size_t,size_t valid){
 CUpti_Activity * r=nullptr;pthread_mutex_lock(&mutex);
 while(cuptiActivityGetNextRecord(buffer,valid,&r)==CUPTI_SUCCESS){
  if(!output)continue;
  if(r->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind==CUPTI_ACTIVITY_KIND_KERNEL){auto * k=(CUpti_ActivityKernel9*)r;fprintf(output,"KERNEL,%u,%llu,%llu,0,%u,%u,%llu,",k->deviceId,(unsigned long long)k->start,(unsigned long long)k->end,k->correlationId,k->streamId,(unsigned long long)k->graphNodeId);quoted(k->name);fputc('\n',output);}
  else if(r->kind==CUPTI_ACTIVITY_KIND_MEMCPY){auto * m=(CUpti_ActivityMemcpy5*)r;fprintf(output,"COPY,%u,%llu,%llu,%llu,%u,%u,0,\"kind_%u\"\n",m->deviceId,(unsigned long long)m->start,(unsigned long long)m->end,(unsigned long long)m->bytes,m->correlationId,m->streamId,(unsigned)m->copyKind);}
  else if(r->kind==CUPTI_ACTIVITY_KIND_RUNTIME){auto * a=(CUpti_ActivityAPI*)r;const char * name=nullptr;cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API,a->cbid,&name);fprintf(output,"API,-1,%llu,%llu,0,%u,%u,0,",(unsigned long long)a->start,(unsigned long long)a->end,a->correlationId,a->threadId);quoted(name);fputc('\n',output);}
 }
 size_t dropped=0;cuptiActivityGetNumDroppedRecords(ctx,stream,&dropped);if(output&&dropped)fprintf(output,"DROPPED,-1,0,0,%zu,0,0,0,\"records\"\n",dropped);
 if(output)fflush(output);pthread_mutex_unlock(&mutex);free(buffer);
}
static bool ok(CUptiResult r,const char * name){if(r==CUPTI_SUCCESS)return true;const char * msg=nullptr;cuptiGetResultString(r,&msg);fprintf(stderr,"NEXT_TRACE %s: %s\n",name,msg?msg:"error");return false;}
static void * monitor(void *){
 bool registered=false,active=false,latched=false;auto begin=std::chrono::steady_clock::now();
 for(;;){bool requested=access(flag_path,F_OK)==0;
  if(!requested)latched=false;
  if(requested&&!active&&!latched){
   if(!output){output=fopen(log_path,"a");if(output){fprintf(output,"kind,device,start_ns,end_ns,bytes,correlation,stream_or_thread,graph_node,name\n");fflush(output);}}
   if(!registered)registered=ok(cuptiActivityRegisterCallbacks(request_buffer,complete_buffer),"register");
   if(registered&&output){bool enabled=ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"kernels");enabled&=ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),"copies");enabled&=ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),"runtime");active=enabled;collecting.store(enabled);if(!enabled){cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);cuptiActivityFlushAll(0);}marker(enabled?"START":"ENABLE_FAILED");begin=std::chrono::steady_clock::now();}
   latched=true;
  }
  if(active&&(!requested||std::chrono::steady_clock::now()-begin>std::chrono::seconds(15))){
   collecting.store(false);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);marker("STOP");active=false;
  }
  usleep(50000);
 }
 return nullptr;
}
__attribute__((constructor)) static void init_trace(){flag_path=getenv("NEXT_TRACE_FLAG");log_path=getenv("NEXT_TRACE_LOG");if(flag_path&&log_path){pthread_t t;if(pthread_create(&t,nullptr,monitor,nullptr)==0)pthread_detach(t);}}

extern "C" uint64_t next_trace_time(){uint64_t ts=0;if(collecting.load())cuptiGetTimestamp(&ts);return ts;}
extern "C" void next_trace_cpu(const char * name,uint64_t start,uint64_t end){if(!start||!end)return;pthread_mutex_lock(&mutex);if(output){fprintf(output,"CPU,-1,%llu,%llu,0,0,0,0,",(unsigned long long)start,(unsigned long long)end);quoted(name);fputc('\n',output);}pthread_mutex_unlock(&mutex);}
