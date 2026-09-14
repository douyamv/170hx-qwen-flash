# next-trace.cpp: markers-only mode (flag file "<NEXT_TRACE_FLAG>.cpu") — CPU markers without CUPTI activity tracing
p = '/home/douya/tests/flashnext-udq4-20260913/deployment-v2/next-trace.cpp'
s = open(p).read()
old_mon = s[s.index('static void * monitor(void *){'):s.index('__attribute__((constructor))')]
new_mon = r'''static void * monitor(void *){
 bool registered=false,active=false,latched=false,active_cpu=false,latched_cpu=false;auto begin=std::chrono::steady_clock::now();
 std::string cpu_flag=std::string(flag_path)+".cpu";
 for(;;){bool requested=access(flag_path,F_OK)==0;bool requested_cpu=access(cpu_flag.c_str(),F_OK)==0;
  if(!requested)latched=false;
  if(!requested_cpu)latched_cpu=false;
  if(requested&&!active&&!active_cpu&&!latched){
   if(!output){output=fopen(log_path,"a");if(output){fprintf(output,"kind,device,start_ns,end_ns,bytes,correlation,stream_or_thread,graph_node,name\n");fflush(output);}}
   if(!registered)registered=ok(cuptiActivityRegisterCallbacks(request_buffer,complete_buffer),"register");
   if(registered&&output){bool enabled=ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"kernels");enabled&=ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),"copies");enabled&=ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),"runtime");active=enabled;collecting.store(enabled);if(!enabled){cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);cuptiActivityFlushAll(0);}marker(enabled?"START":"ENABLE_FAILED");begin=std::chrono::steady_clock::now();}
   latched=true;
  }
  // markers-only mode: CPU phase markers (next_trace_cpu) with cuptiGetTimestamp, no activity records -> negligible overhead
  if(requested_cpu&&!active&&!active_cpu&&!latched_cpu){
   if(!output){output=fopen(log_path,"a");if(output){fprintf(output,"kind,device,start_ns,end_ns,bytes,correlation,stream_or_thread,graph_node,name\n");fflush(output);}}
   if(output){collecting.store(true);active_cpu=true;marker("START_CPU");begin=std::chrono::steady_clock::now();}
   latched_cpu=true;
  }
  if(active&&(!requested||std::chrono::steady_clock::now()-begin>std::chrono::seconds(15))){
   collecting.store(false);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);marker("STOP");active=false;
  }
  if(active_cpu&&(!requested_cpu||std::chrono::steady_clock::now()-begin>std::chrono::seconds(15))){
   collecting.store(false);marker("STOP_CPU");active_cpu=false;
  }
  usleep(50000);
 }
 return nullptr;
}
'''
s = s.replace(old_mon, new_mon)
if '#include <string>' not in s: s = s.replace('#include <chrono>', '#include <chrono>\n#include <string>', 1)
open(p, 'w').write(s); print('next-trace.cpp: markers-only mode added')
