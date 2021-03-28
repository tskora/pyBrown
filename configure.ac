#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT(pyBD, 0.1)
AC_CONFIG_SRCDIR([src/bdsim/Ccode/diff_tensor.c])

AM_PATH_PYTHON([3.4])

PACKAGE=pyBD
VERSION=0.1

dnl IFACE_NUMBER (current) : REV : AGE (< IFACE_NUMBER)
PYBD_LT_VERSION="1:0:1"
AC_SUBST(PYBD_LT_VERSION)

AC_SUBST(PACKAGE)
AC_SUBST(PACKAGE_STRING)
AC_SUBST(VERSION)

#AM_INIT_AUTOMAKE($PACKAGE, $VERSION)
AM_INIT_AUTOMAKE()

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX

#AC_PROG_RANLIB
AC_PROG_LIBTOOL

# 'Optimizization' must be beore all the other flags!
AC_ARG_ENABLE([optimize],
    [AC_HELP_STRING([--enable-optimize],
            [disable debugging (removes -g flags) and set optimization level O3 (disabled)])],
    [CFLAGS="-O3"], [CFLAGS="$CFLAGS_save"])

# Enable/disable assetions
AC_ARG_ENABLE(assertions,
	      AC_HELP_STRING([--disable-assertions],
			     [disable all assertions (enabled)]), 
			     [disable_assertions=yes AC_DEFINE([MPLIB_NO_ASSERT],,[no assertions])],
			      disable_assertions=no)

if test "x$disable_assertions" = "xyes"; then
	CFLAGS="-DMPLIB_NO_ASSERT $CFLAGS"
#	CFLAGS="-DNDEBUG $CFLAGS"
fi

AC_CONFIG_FILES([Makefile
		src/Makefile
		src/bdsim/Makefile
		src/bdsim/Ccode/Makefile
		src/bdsim/tests/Makefile
		src/tools/Makefile
		])
AC_OUTPUT

