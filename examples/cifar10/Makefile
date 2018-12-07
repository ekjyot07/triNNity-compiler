.PHONY: clean

clean:
	rm -f cifar10.constraints cifar10.layers cifar10.topology cifar10.cpp cifar10.solution cifar10.h cifar10.exe

cifar10.constraints: cifar10-deploy.prototxt
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10-deploy.prototxt --constraints-output=cifar10.constraints

cifar10.layers: cifar10-deploy.prototxt
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10-deploy.prototxt --layers-output=cifar10.layers

cifar10.topology: cifar10-deploy.prototxt
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10-deploy.prototxt --topology-output=cifar10.topology

cifar10.cpp: cifar10.constraints cifar10.layers cifar10.topology
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend trinnity --model cifar10-deploy.prototxt --code-output=cifar10.cpp

cifar10.caffemodel: cifar10-train.prototxt
	caffe train -solver=cifar10-solver.prototxt -snapshot=`ls *.solverstate | sort --version-sort --reverse  | head -n1`
	cp `ls *.caffemodel | sort --version-sort --reverse  | head -n1` cifar10.caffemodel
	caffe test -model=cifar10-train.prototxt -weights=cifar10.caffemodel -iterations=10

cifar10.py: cifar10.caffemodel
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend tensorflow --model cifar10-deploy.prototxt --weights cifar10.caffemodel --code-output=cifar10.py
	chmod +x cifar10.py

cifar10.npy:
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend tensorflow --model cifar10-deploy.prototxt --data-output=cifar10.npy

cifar10.solution: cifar10.constraints cifar10.layers cifar10.topology
	triNNity-optimizer heuristic -x LOCAL_OPTIMAL  \
                               -t cifar10.topology  \
                               -a triNNity-algorithms.csv  \
                               -n cost-models/Intel\(R\)\ Core\(TM\)\ i5-2500K\ CPU\ @\ 3.30GHz/hcy.csv \
                               > cifar10.solution

cifar10.h: cifar10.layers cifar10.topology cifar10.solution
	triNNity-optimizer instantiate -t cifar10.topology \
                                 -a triNNity-algorithms.csv \
                                 -m triNNity-methods.csv \
                                 -s cifar10.solution \
                                 -i TRINNITY \
                                 -l cifar10.layers \
                                 > cifar10.h

lpdnn-cifar10.h: cifar10.layers cifar10.topology cifar10.solution
	triNNity-optimizer instantiate -t cifar10.topology \
                                 -a triNNity-algorithms.csv \
                                 -m triNNity-methods.csv \
                                 -s cifar10.solution \
                                 -i LPDNN \
                                 -l cifar10.layers \
                                 > lpdnn-cifar10.h

cifar10.exe: cifar10.layers cifar10.topology cifar10.solution cifar10.cpp cifar10.h
	$(CC) -DTRINNITY_USE_CBLAS_GEMM -DTRINNITY_USE_CBLAS_GEMV -DTRINNITY_ENABLE_SPECTRAL_GENERIC_LAYER -DWEIGHT_TYPE=float -DACTIVATION_TYPE=float cifar10.cpp -o cifar10.exe
