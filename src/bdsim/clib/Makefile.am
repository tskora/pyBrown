lib_LTLIBRARIES = libpyBrown.la

libpyBrown_la_SOURCES = diff_tensor.c bead.c

libpyBrown_la_LDFLAGS = -version-info @PYBD_LT_VERSION@ -pedantic -ansi -fwritable-strings -lm -g @LAPACK_LIBS@
AM_CFLAGS= -Wall
AM_LDFLAGS = -pedantic -ansi -fwritable-strings -lm -g @LAPACK_LIBS@

