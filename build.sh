#!/bin/bash
# -*- coding: utf-8 -*-
set -e

if [ -z "$1" ]; then
	set $1="all"
fi

make_dir() {
	echo "in make_dir $1"
	if [ $1 == libumd.so ]; then
		rm -rf build && mkdir build
	fi

	if [ $1 == all ]; then
		rm -rf build && mkdir build
	fi
}

build_dbg() {
	echo "in build_dbg $1"
	make_dir $1
	cd build && meson .. && ninja $1 || exit 1
}

build_release() {
	echo "in build_release $1"
	make_dir $1
	cd build && meson .. --buildtype release && ninja $1 || exit 1
}

if [ ! -z "$2" ]; then
	if [ $2 == lint ]; then
		echo start link check ...
	elif [ $2 == dbg ]; then
		build_dbg $1;
	elif [ $2 == rel ]; then
		build_release $1;
	fi
else
	build_dbg $1;
fi

