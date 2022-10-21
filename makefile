.ONESHELL:

SHELL=/bin/bash

CC=g++

FLAGS=-std=c++2a -ffunction-sections -fdata-sections -march=native -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-aliasing=1 -Wpointer-arith -Iexternal/include -isystem /home/james/projects/cpv/root/include

LIBS="external/libs/libfmt.a" `root-config --libs`

OPT=-O3 -fomit-frame-pointer

DBG=-g -fsanitize=address,undefined -static-libasan

.PHONY: all

all: cache/test

cache/test: root makefile simulation.cpp external/fmt
	@
	if [[ "$${HOSTNAME: -14}" = ".warwick.ac.uk" ]]; then
		module is-loaded GCC/11.2.0
		if [ $$? -ne 0 ]; then
			echo "loading compiler (you may want to module load GCC/11.2 to avoid this happening everytime)"
			module load GCC/11.2
		fi
	fi
	echo "compiling"
	$(CC) $(OPT) $(FLAGS) simulation.cpp $(LIBS) -o cache/test

debug: root makefile simulation.cpp external/fmt
	@
	if [[ "$${HOSTNAME: -14}" = ".warwick.ac.uk" ]]; then
		module is-loaded GCC/11.2.0
		if [ $$? -ne 0 ]; then
			echo "loading compiler (you may want to module load GCC/11.2 to avoid this happening everytime)"
			module load GCC/11.2
		fi
	fi
	echo "compiling"
	$(CC) $(DBG) $(FLAGS) simulation.cpp $(LIBS) -o cache/test

run: makefile all
	./cache/test

external/fmt: makefile external/get-fmt.sh
	pushd external > /dev/null
	./get-fmt.sh
	popd > /dev/null

root: makefile
	@
	which root
	if [ $$? -ne 0 ]; then
		echo "root not found. aborting."
		exit 1
	fi



clean: makefile
	rm cache/test
