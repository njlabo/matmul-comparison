
compile: 
	clang -Os -Wall -lm -framework Accelerate -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 run-linear.c -o run-linear
	clang -Os -Wall -lm -framework Accelerate -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 run-conv.c -o run-conv
	clang -Os -Wall -lm -fopenmp -framework Accelerate -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 run-linear.c -o run-linear-p
	clang -Os -Wall -lm -fopenmp -framework Accelerate -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 run-conv.c -o run-conv-p

.PHONY: compile