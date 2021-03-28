SUBDIRS=Ccode tests
dist_bin_SCRIPTS = BD.py  RestartBD.py

PYTHON_LIB_DIR=pyBD
PYTHON_LIB = $(PYTHON_LIB_DIR)/bead.py\
	$(PYTHON_LIB_DIR)/box.py\
	$(PYTHON_LIB_DIR)/diffusion.py\
	$(PYTHON_LIB_DIR)/input.py\
	$(PYTHON_LIB_DIR)/output.py

# Provide PYTHON_INSTALL_DIR 
# via the configure script with --enable-system-python option
install-exec-hook:
	cp --parents $(PYTHON_LIB) `python3 -m site --user-site`
