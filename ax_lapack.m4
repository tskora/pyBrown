AC_DEFUN([AX_LAPACK], 
[
ax_lapack_ok=no

AC_ARG_WITH(lapack,
	[AC_HELP_STRING([--with-lapack=<lib>], [use LAPACK library <lib>])])

AS_CASE(["$with_lapack"],
        [yes],[ax_lapack_ok=try],
        [no|""],[ax_lapack_ok=disable],
        [-* | */* | *.a | *.so | *.so.* | *.o],[LAPACK_LIBS="$with_lapack"; ax_lapack_ok=try],
        [LAPACK_LIBS="-l$with_lapack"; ax_lapack_ok=try])

ax_lapack_save_LIBS="$LIBS"

AS_IF([test "x$ax_lapack_ok" = xtry],
      [dpotrf=dpotrf_
	   LIBS+=" $LAPACK_LIBS"
	   AS_IF([test -z "$LIBS"],
	         [AC_MSG_CHECKING([for $dpotrf in your system])],
	         [AC_MSG_CHECKING([for $dpotrf in $LIBS])])	   
	   AC_TRY_LINK_FUNC($dpotrf, [ax_lapack_ok=yes], [LAPACK_LIBS=""])
	   AS_IF([test "x$ax_lapack_ok" = xtry],
	         [ax_lapack_ok=no],[])
	   AC_MSG_RESULT($ax_lapack_ok)],
      [])

# restore the libs variable
LIBS=$ax_lapack_save_LIBS
]) 
