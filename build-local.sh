#!/bin/bash
mkdir -p build/libfractal || exit;

cd build/libfractal || exit;

if [ ! -f Makefile ]; then
	../../libfractal/configure --prefix=`pwd`/.. || exit;
fi

make install || exit;
