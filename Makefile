init:
	pip3 install -r requirements.txt

Ccode/diff_tensor.so: Ccode/diff_tensor.c
	gcc -o Ccode/diff_tensor.so -shared Ccode/diff_tensor.c 

test:
	pytest tests

.PHONY: init test