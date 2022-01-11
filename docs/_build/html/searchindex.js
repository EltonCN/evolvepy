Search.setIndex({docnames:["evolvepy","evolvepy.callbacks","evolvepy.evaluator","evolvepy.generator","evolvepy.generator.arrangement","evolvepy.generator.crossover","evolvepy.generator.mutation","evolvepy.generator.selection","evolvepy.integrations","evolvepy.integrations.tf_keras","evolvepy.integrations.wandb","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["evolvepy.rst","evolvepy.callbacks.rst","evolvepy.evaluator.rst","evolvepy.generator.rst","evolvepy.generator.arrangement.rst","evolvepy.generator.crossover.rst","evolvepy.generator.mutation.rst","evolvepy.generator.selection.rst","evolvepy.integrations.rst","evolvepy.integrations.tf_keras.rst","evolvepy.integrations.wandb.rst","index.rst"],objects:{"evolvepy.callbacks":[[1,0,0,"-","callback"],[1,0,0,"-","dynamic_mutation"],[1,0,0,"-","incremental_evolution"],[1,0,0,"-","logger"]],"evolvepy.callbacks.callback":[[1,1,1,"","Callback"]],"evolvepy.callbacks.callback.Callback":[[1,2,1,"","callbacks"],[1,2,1,"","evaluator"],[1,2,1,"","generator"],[1,3,1,"","on_evaluator_end"],[1,3,1,"","on_generator_end"],[1,3,1,"","on_generator_start"],[1,3,1,"","on_start"],[1,3,1,"","on_stop"]],"evolvepy.callbacks.dynamic_mutation":[[1,1,1,"","DynamicMutation"]],"evolvepy.callbacks.dynamic_mutation.DynamicMutation":[[1,4,1,"","EXPLORATION"],[1,4,1,"","NORMAL"],[1,4,1,"","REFINEMENT"],[1,3,1,"","exploration_step"],[1,3,1,"","on_evaluator_end"],[1,3,1,"","refinement_step"],[1,3,1,"","restore_parameters"],[1,3,1,"","save_parameters"]],"evolvepy.callbacks.incremental_evolution":[[1,1,1,"","IncrementalEvolution"]],"evolvepy.callbacks.incremental_evolution.IncrementalEvolution":[[1,3,1,"","on_generator_start"]],"evolvepy.callbacks.logger":[[1,1,1,"","FileStoreLogger"],[1,1,1,"","Logger"],[1,1,1,"","MemoryStoreLogger"]],"evolvepy.callbacks.logger.FileStoreLogger":[[1,2,1,"","log"],[1,3,1,"","save_dynamic_log"],[1,3,1,"","save_static_log"]],"evolvepy.callbacks.logger.Logger":[[1,3,1,"","on_evaluator_end"],[1,3,1,"","on_generator_end"],[1,3,1,"","on_start"],[1,3,1,"","save_dynamic_log"],[1,3,1,"","save_static_log"]],"evolvepy.callbacks.logger.MemoryStoreLogger":[[1,2,1,"","config_log"],[1,2,1,"","log"],[1,3,1,"","save_dynamic_log"],[1,3,1,"","save_static_log"]],"evolvepy.configurable":[[0,1,1,"","Configurable"]],"evolvepy.configurable.Configurable":[[0,4,1,"","__element_count"],[0,2,1,"","dynamic_parameters"],[0,3,1,"","lock_parameter"],[0,2,1,"","name"],[0,2,1,"","parameters"],[0,3,1,"","reset_count"],[0,2,1,"","static_parameters"],[0,3,1,"","unlock_parameter"]],"evolvepy.evaluator":[[2,0,0,"-","aggregator"],[2,0,0,"-","cache"],[2,0,0,"-","dispatcher"],[2,0,0,"-","evaluator"],[2,0,0,"-","function_evaluator"],[2,0,0,"-","process_evaluator"]],"evolvepy.evaluator.aggregator":[[2,1,1,"","FitnessAggregator"]],"evolvepy.evaluator.aggregator.FitnessAggregator":[[2,4,1,"","MAX"],[2,4,1,"","MEAN"],[2,4,1,"","MEDIAN"],[2,4,1,"","MIN"],[2,4,1,"","MODE_NAMES"],[2,4,1,"","func"]],"evolvepy.evaluator.cache":[[2,1,1,"","FitnessCache"]],"evolvepy.evaluator.cache.FitnessCache":[[2,3,1,"","get_individual_representation"]],"evolvepy.evaluator.dispatcher":[[2,1,1,"","MultipleEvaluation"]],"evolvepy.evaluator.evaluator":[[2,1,1,"","EvaluationStage"],[2,1,1,"","Evaluator"]],"evolvepy.evaluator.evaluator.Evaluator":[[2,2,1,"","scores"]],"evolvepy.evaluator.function_evaluator":[[2,1,1,"","FunctionEvaluator"]],"evolvepy.evaluator.function_evaluator.FunctionEvaluator":[[2,4,1,"","JIT"],[2,4,1,"","JIT_PARALLEL"],[2,4,1,"","NJIT"],[2,4,1,"","NJIT_PARALLEL"],[2,4,1,"","PYTHON"],[2,3,1,"","call"]],"evolvepy.evaluator.process_evaluator":[[2,1,1,"","ProcessEvaluator"],[2,1,1,"","ProcessFitnessFunction"],[2,5,1,"","evaluate_forever"]],"evolvepy.evaluator.process_evaluator.ProcessFitnessFunction":[[2,3,1,"","evaluate"],[2,3,1,"","setup"]],"evolvepy.evolver":[[0,1,1,"","Evolver"]],"evolvepy.evolver.Evolver":[[0,3,1,"","evolve"]],"evolvepy.generator":[[4,0,0,"-","arrangement"],[3,0,0,"-","basic_layers"],[3,0,0,"-","combine"],[3,0,0,"-","context"],[5,0,0,"-","crossover"],[3,0,0,"-","descriptor"],[3,0,0,"-","firstgen"],[3,0,0,"-","generator"],[3,0,0,"-","layer"],[6,0,0,"-","mutation"],[7,0,0,"-","selection"]],"evolvepy.generator.arrangement":[[4,0,0,"-","ramdom_predation"],[4,0,0,"-","sintatic_predation"]],"evolvepy.generator.basic_layers":[[3,1,1,"","Block"],[3,1,1,"","FilterFirsts"],[3,1,1,"","RandomPredation"],[3,1,1,"","Sort"]],"evolvepy.generator.basic_layers.Block":[[3,3,1,"","call"]],"evolvepy.generator.basic_layers.FilterFirsts":[[3,3,1,"","call"]],"evolvepy.generator.basic_layers.RandomPredation":[[3,3,1,"","call"]],"evolvepy.generator.basic_layers.Sort":[[3,3,1,"","call"]],"evolvepy.generator.combine":[[3,1,1,"","CombineLayer"]],"evolvepy.generator.combine.CombineLayer":[[3,3,1,"","call_chromossomes"],[3,3,1,"","combine"]],"evolvepy.generator.context":[[3,1,1,"","Context"]],"evolvepy.generator.context.Context":[[3,2,1,"","block_all"],[3,2,1,"","chromossome_names"],[3,3,1,"","copy"],[3,4,1,"","default_values"],[3,3,1,"","have_value"],[3,2,1,"","population_size"],[3,2,1,"","sorted"]],"evolvepy.generator.crossover":[[5,0,0,"-","crossover"]],"evolvepy.generator.crossover.crossover":[[5,5,1,"","default_crossover"],[5,5,1,"","mean"],[5,5,1,"","n_point"],[5,5,1,"","one_point"]],"evolvepy.generator.descriptor":[[3,1,1,"","Descriptor"]],"evolvepy.generator.descriptor.Descriptor":[[3,2,1,"","chromossome_names"],[3,2,1,"","chromossome_ranges"],[3,2,1,"","dtype"]],"evolvepy.generator.firstgen":[[3,1,1,"","FirstGenLayer"]],"evolvepy.generator.firstgen.FirstGenLayer":[[3,3,1,"","call"],[3,3,1,"","call_chromossomes"]],"evolvepy.generator.generator":[[3,1,1,"","Generator"]],"evolvepy.generator.generator.Generator":[[3,3,1,"","add"],[3,2,1,"","fitness"],[3,3,1,"","generate"],[3,3,1,"","get_all_dynamic_parameters"],[3,3,1,"","get_all_static_parameters"],[3,3,1,"","get_parameter"],[3,3,1,"","get_parameters"],[3,3,1,"","set_parameter"],[3,3,1,"","set_parameters"]],"evolvepy.generator.layer":[[3,1,1,"","ChromossomeOperator"],[3,1,1,"","Concatenate"],[3,1,1,"","Layer"]],"evolvepy.generator.layer.ChromossomeOperator":[[3,3,1,"","call"],[3,3,1,"","call_chromossomes"]],"evolvepy.generator.layer.Layer":[[3,3,1,"","call"],[3,2,1,"","context"],[3,2,1,"","fitness"],[3,2,1,"","next"],[3,2,1,"","population"],[3,3,1,"","send_next"]],"evolvepy.generator.mutation":[[6,0,0,"-","binary_mutation"],[6,0,0,"-","mutation"],[6,0,0,"-","numeric_mutation"]],"evolvepy.generator.mutation.binary_mutation":[[6,5,1,"","bit_mutation"]],"evolvepy.generator.mutation.mutation":[[6,1,1,"","NumericMutationLayer"],[6,5,1,"","default_mutation"]],"evolvepy.generator.mutation.mutation.NumericMutationLayer":[[6,3,1,"","call_chromossomes"],[6,3,1,"","mutate"]],"evolvepy.generator.mutation.numeric_mutation":[[6,5,1,"","mul_mutation"],[6,5,1,"","sum_mutation"]],"evolvepy.generator.selection":[[7,0,0,"-","selection"]],"evolvepy.generator.selection.selection":[[7,5,1,"","choice"],[7,5,1,"","default_selection"],[7,5,1,"","isin"],[7,5,1,"","rank"],[7,5,1,"","roulette"],[7,5,1,"","tournament"]],"evolvepy.integrations":[[9,0,0,"-","tf_keras"],[10,0,0,"-","wandb"]],"evolvepy.integrations.tf_keras":[[9,0,0,"-","tf_keras"]],"evolvepy.integrations.tf_keras.tf_keras":[[9,1,1,"","EvolutionaryModel"],[9,1,1,"","LossFitnessFunction"],[9,1,1,"","ProcessTFKerasEvaluator"],[9,1,1,"","ProcessTFKerasFitnessFunction"],[9,1,1,"","TFKerasEvaluator"],[9,5,1,"","get_descriptor"],[9,5,1,"","transfer_weights"]],"evolvepy.integrations.tf_keras.tf_keras.EvolutionaryModel":[[9,3,1,"","compile"],[9,2,1,"","descriptor"],[9,3,1,"","train_step"]],"evolvepy.integrations.tf_keras.tf_keras.ProcessTFKerasFitnessFunction":[[9,3,1,"","evaluate"]],"evolvepy.integrations.tf_keras.tf_keras.TFKerasEvaluator":[[9,2,1,"","descriptor"]],"evolvepy.integrations.wandb":[[10,0,0,"-","wandb"]],"evolvepy.integrations.wandb.wandb":[[10,1,1,"","WandbLogger"]],"evolvepy.integrations.wandb.wandb.WandbLogger":[[10,3,1,"","on_stop"],[10,3,1,"","save_dynamic_log"],[10,3,1,"","save_static_log"]],evolvepy:[[1,0,0,"-","callbacks"],[0,0,0,"-","configurable"],[2,0,0,"-","evaluator"],[0,0,0,"-","evolver"],[3,0,0,"-","generator"],[8,0,0,"-","integrations"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","property","Python property"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:property","3":"py:method","4":"py:attribute","5":"py:function"},terms:{"0":[1,2,7,9],"1":[1,2,3,5,6,7,9],"10":1,"1d":9,"2":[1,2,3,7,9],"2d":9,"3":[2,7],"4":7,"5":[1,7],"6":7,"7":7,"abstract":[1,2,9],"boolean":7,"byte":[2,3,5,6,7],"case":[7,9],"class":[0,1,2,3,6,9,10],"default":[7,9],"do":9,"float":[2,3,5,6,7,9],"function":[2,7,9],"int":[0,1,2,3,5,6,7,9],"new":[5,7],"return":[5,7],"static":[0,2,3,6],"true":[1,3,7,10],By:9,For:7,IN:9,If:[0,6,7,9],In:9,Is:0,It:9,NOT:9,The:[0,7,9],To:9,__element_count:0,_array_lik:[2,3,5,6,7],_block_al:3,_chromossome_nam:3,_dtype_lik:3,_dtypedict:3,_population_s:3,_sort:3,_supportsarrai:[2,3,5,6,7],_supportsdtyp:3,_valu:3,aa_milne_arr:7,abc:[0,1,2],abov:7,access:[],accuraci:9,add:3,addit:9,aggreg:0,agreg:2,all:[0,7,9],allow:0,also:9,amax:2,amin:2,an:7,ani:[2,3,5,6,7,9],ar:7,arang:7,arbitrari:7,arg:[2,9],argument:9,arr:7,arrai:[5,7],arrang:[0,3],arraylik:[5,7],associ:7,assum:7,attribut:0,avoid:0,axi:7,base:[0,1,2,3,6,9,10],basecontext:2,basic_lay:[0,1],behav:6,beign:7,best:7,binary_mut:[0,3],bit:6,bit_mut:6,bitstr:6,block:[1,3],block_al:3,block_lay:1,bool:[0,1,2,3,9,10],cach:0,call:[2,3],call_chromossom:[3,6],callabl:[2,3,6,9],callback:[0,9,10],can:[0,7,9],chang:[0,6],choic:7,chosen:7,christoph:7,chromosom:5,chromossom:[3,6],chromossome_nam:[3,6],chromossome_rang:3,chromossome_s:3,chromossomeoper:[3,6],class_weight:9,classmethod:0,code:7,coeffici:9,combin:0,combinelay:3,compil:9,complex:[2,3,5,6,7],comput:5,concaten:3,config_log:1,configur:[1,2,3,9,11],content:0,context:[0,2,6],contribut:9,copi:3,could:9,count:0,creat:[0,9],crossov:[0,3],crossover_funct:3,d:7,data:9,decreasingli:7,default_crossov:5,default_mut:6,default_rng:7,default_select:7,default_valu:3,descriptor:[0,9],dict:[0,1,2,3,9,10],dictionari:9,differ:[7,9],dimension:7,discard_max:2,discard_min:2,dispatch:0,distribut:[7,9],drawn:7,dtype:[3,7],dure:9,dynam:0,dynamic_mut:0,dynamic_paramet:[0,1,2,3],dynamicmut:1,e:7,each:[7,9],element:[0,7],els:6,engin:9,entiti:10,entri:7,ep_callback:9,equival:7,evalu:[0,1,7,9],evaluate_forev:2,evaluationstag:2,evolutionarymodel:9,evolv:11,evolvepi:[],exampl:7,existence_r:6,expect:9,explor:1,exploration_multipli:1,exploration_pati:1,exploration_step:1,extern:9,fals:[1,2,3,7,9,10],fed:9,filestorelogg:1,filterfirst:3,first_gen_lay:1,first_lay:3,firstgen:[0,1],firstgenlay:[1,3],fit:[1,3,6,7],fitness_arrai:7,fitness_funct:[2,9],fitnessaggreg:2,fitnesscach:2,flipbit:6,float32:3,from:7,func:2,function_evalu:0,functionevalu:2,g:7,gene:6,gene_r:6,gener:[0,1,2,9],gener_r:6,generation_to_start:1,get_all_dynamic_paramet:3,get_all_static_paramet:3,get_descriptor:9,get_individual_represent:2,get_paramet:3,given:7,greater:7,group:10,ha:9,have:[0,7,9],have_valu:3,incremental_evolut:0,incrementalevolut:1,index:7,individu:[2,7,9],individual_per_cal:[2,9],individuals_queu:2,initialize_zero:3,instanc:[0,7,9],instead:[7,9],integ:7,integr:0,invalid:9,isin:7,item:7,its:[0,7],jit:2,jit_parallel:2,join:5,just:7,k:7,kei:0,kera:9,keyword:7,kwarg:9,last_lay:3,layer:[0,6,9],layer_nam:[1,3],len:9,length:7,less:7,like:[6,7,9],list:[0,1,2,3,6,9],lock:0,lock_paramet:0,log:[0,1,10],log_best_individu:1,log_evalu:[1,10],log_fit:[1,10],log_gener:[1,10],log_popul:1,log_scor:[1,10],logger:[0,10],loss:9,loss_weight:9,lossfitnessfunct:9,m:7,map:9,max:2,max_decim:2,mean:[2,5],median:2,memorystorelogg:1,method:7,metric:9,min:2,minim:9,mode:[2,9],mode_nam:2,model:9,modul:11,mse:9,mul_mut:6,multi:9,multipl:9,multipleevalu:2,multiprocess:2,mutaion:6,mutat:[0,3],mutation_funct:6,mutation_rang:6,n:[5,6,7],n_combin:3,n_evalu:2,n_gener:2,n_point:5,n_process:[2,9],n_score:[2,9],n_select:7,n_to_pass:3,n_to_pred:3,name:[0,3,6,9,10],ndarrai:[1,2,3,5,6,7,9],need:9,next:3,njit:2,njit_parallel:2,non:7,none:[0,1,2,3,6,7,9,10],normal:1,note:7,np:[5,7],number:[0,5,6,7],numeric_mut:[0,3],numericmutationlay:6,numpi:[1,2,3,5,6,7,9],object:[0,1,2,3,9],on_evaluator_end:1,on_generator_end:1,on_generator_start:1,on_start:1,on_stop:[1,10],one:[5,7],one_point:5,optim:9,option:[0,1,2,3,6,7,9,10],other_paramet:2,output:[7,9],output_a:9,output_b:9,over:7,own:9,p:7,packag:11,param:7,paramet:[0,1,2,3,5,7,9],parameter_nam:3,pass:9,patienc:1,permut:7,piglet:7,placehold:9,pleas:[7,9],point:5,pooh:7,popul:[1,2,3,7],population_s:[0,3,9],possibl:7,probabl:7,process_evalu:[0,9],processevalu:[2,9],processfitnessfunct:[2,9],processtfkerasevalu:9,processtfkerasfitnessfunct:9,project:[10,11],properti:[0,1,2,3,9],python:[2,9],queue:2,quick:7,rabbit:7,rais:[7,9],ramdom_pred:[0,3],randint:7,random:7,randomicali:6,randompred:3,rank:7,refin:1,refinement_divid:1,refinement_pati:1,refinement_step:1,repeat:7,replac:7,reset:[0,2,9],reset_count:0,restore_paramet:1,roulett:7,row:7,rtype:7,run:[1,3],s:[0,9],sampl:[7,9],sample_weight:9,sample_weight_mod:9,save_dynamic_log:[1,10],save_paramet:1,save_static_log:[1,10],scalar:9,scope:9,score:2,scores_queu:2,see:[7,9],select:[0,3],selection_funct:3,send_next:3,sequenc:[2,3,5,6,7],sequenti:9,set:9,set_paramet:3,setter:0,setup:2,shape:7,should:7,shuffl:7,singl:[0,7,9],sintatic_pred:[0,3],size:7,some:0,sort:[3,7],specifi:9,ssc0713:11,start:7,static_paramet:0,steps_per_execut:9,stocaticali:7,stop_refin:1,str:[0,1,2,3,5,6,7,9,10],strategi:9,string:9,submodul:[8,11],subpackag:11,sum:9,sum_mut:6,support:9,take:6,target:9,target_tensor:9,tempor:9,tensor:9,tensorflow:9,test:[7,9,11],tf:9,tf_kera:[0,8],tfkerasevalu:9,than:7,them:9,thi:[6,7,9],through:7,time:9,timeout:[2,9],timestep:9,tournament:7,train:9,train_step:9,transfer_weight:9,tupl:[0,3,6,7],turn:9,type:[0,2,3,5,6,7,9],typic:9,u11:7,under:9,uniform:7,union:[0,2,3,5,6,7],unlock:0,unlock_paramet:0,us:[0,7,9],val:7,valid:0,valu:[0,3,7,9],valueerror:[7,9],vector:7,version:7,via:9,vs:7,wandb:[0,8],wandblogg:10,weight:[2,9],weighted_metr:9,were:7,whether:7,which:[7,9],wise:9,without:7,would:9,x:9,y:9,you:9,your:9,zero:7},titles:["evolvepy package","evolvepy.callbacks package","evolvepy.evaluator package","evolvepy.generator package","evolvepy.generator.arrangement package","evolvepy.generator.crossover package","evolvepy.generator.mutation package","evolvepy.generator.selection package","evolvepy.integrations package","evolvepy.integrations.tf_keras package","evolvepy.integrations.wandb package","EvolvePy\u2019s Documentation"],titleterms:{aggreg:2,arrang:4,basic_lay:3,binary_mut:6,cach:2,callback:1,combin:3,configur:0,content:[1,2,3,4,5,6,7,8,9,10,11],context:3,crossov:5,descriptor:3,dispatch:2,document:11,dynamic_mut:1,evalu:2,evolv:0,evolvepi:[0,1,2,3,4,5,6,7,8,9,10,11],firstgen:3,function_evalu:2,gener:[3,4,5,6,7],incremental_evolut:1,integr:[8,9,10],layer:3,logger:1,modul:[0,1,2,3,4,5,6,7,8,9,10],mutat:6,numeric_mut:6,packag:[0,1,2,3,4,5,6,7,8,9,10],process_evalu:2,ramdom_pred:4,s:11,select:7,sintatic_pred:4,src:[],submodul:[0,1,2,3,4,5,6,7,9,10],subpackag:[0,3,8],tf_kera:9,wandb:10,welcom:[]}})