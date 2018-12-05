cifar10.constraints:
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10.prototxt --constraints-output=cifar10.constraints

cifar10.layers:
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10.prototxt --layers-output=cifar10.layers

cifar10.topology:
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend optimizer-info --model cifar10.prototxt --topology-output=cifar10.topology

cifar10.cpp: cifar10.constraints cifar10.layers cifar10.topology
	PYTHONPATH+=:/usr/python/ ../../triNNity-compiler --backend trinnity --model cifar10.prototxt --code-output=cifar10.cpp

cifar10.exe: cifar10.cpp
	$(CC) cifar10.cpp -o cifar10.exe
