SUBDIRS = clib tests
dist_bin_SCRIPTS = BD.py  BD-restart.py BD-NAM.py BD-NAM-restart.py

PYTHON_LIB_DIR=pyBrown
PYTHON_LIB = $(PYTHON_LIB_DIR)/bead.py\
	$(PYTHON_LIB_DIR)/box.py\
	$(PYTHON_LIB_DIR)/diffusion.py\
	$(PYTHON_LIB_DIR)/input.py\
	$(PYTHON_LIB_DIR)/interactions.py\
	$(PYTHON_LIB_DIR)/output.py\
	$(PYTHON_LIB_DIR)/plane.py\
	$(PYTHON_LIB_DIR)/reactions.py

# Provide PYTHON_INSTALL_DIR 
# via the configure script with --enable-system-python option
install-exec-hook:
	mkdir -p `python3 -m site --user-site`
	echo "Installing python libraries to `python3 -m site --user-site`"
	cp -rf --parents $(PYTHON_LIB) `python3 -m site --user-site`
