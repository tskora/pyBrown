init:
# 	pip install -r requirements.txt
	gcc -o diff_tensor.so -shared diff_tensor.c

diff_tensor.so: diff_tensor.c
	gcc -o diff_tensor.so -shared diff_tensor.c 

test:
	pytest tests

.PHONY: init test