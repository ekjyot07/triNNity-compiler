.PHONY: clean

clean:
	rm -f cifar10.constraints cifar10.layers cifar10.topology cifar10.cpp cifar10.solution cifar10.h cifar10.exe

cifar10.constraints: cifar10.prototxt
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10.prototxt --constraints-output=cifar10.constraints

cifar10.layers: cifar10.prototxt
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10.prototxt --layers-output=cifar10.layers

cifar10.topology: cifar10.prototxt
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10.prototxt --topology-output=cifar10.topology

cifar10.cpp: cifar10.constraints cifar10.layers cifar10.topology
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend trinnity --model cifar10.prototxt --code-output=cifar10.cpp

cifar10.solution: cifar10.constraints cifar10.layers cifar10.topology
	triNNity-optimizer heuristic -x LOCAL_OPTIMAL  \
                               -t cifar10.topology  \
                               -a triNNity-algorithms.csv  \
                               -n cost-models/Intel\(R\)\ Core\(TM\)\ i5-2500K\ CPU\ @\ 3.30GHz/hcy.csv \
                               > cifar10.solution

cifar10.h:
	triNNity-optimizer instantiate -t cifar10.topology \
                                 -a triNNity-algorithms.csv \
                                 -m triNNity-methods.csv \
                                 -s cifar10.solution \
                                 -i TRINNITY \
                                 -l cifar10.layers \
                                 > cifar10.h

cifar10.exe: cifar10.layers cifar10.topology cifar10.solution cifar10.cpp cifar10.h
	$(CC) cifar10.cpp -o cifar10.exe
