Ccode/diff_tensor.so: Ccode/diff_tensor.c
	gcc -o Ccode/diff_tensor.so -shared Ccode/diff_tensor.c

test:
	python3 tests/test_bead.py
	python3 tests/test_box.py
	python3 tests/test_diffusion.py

init:
	pip3 install -r requirements.txt

.PHONY: init test