.ONESHELL:

SHELL=/bin/bash

CC=g++

.PHONY: all

all: test

test: makefile simulation.cpp external/fmt
	@
	if [[ "$${HOSTNAME: -14}" = ".warwick.ac.uk" ]]; then
		module is-loaded GCC/11.2.0
		if [ $$? -ne 0 ]; then
			echo "loading compiler (you may want to module load GCC/11.2 to avoid this happening everytime)"
			module load GCC/11.2
		fi
	fi
	echo "compiling"
	$(CC) -O3 -std=c++2b -fno-stack-protector -fomit-frame-pointer -ffunction-sections -fdata-sections -march=native -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-aliasing=1 -Wpointer-arith -Iexternal/include simulation.cpp "external/libs/libfmt.a" -o test

run: makefile all
	./test

external/fmt: makefile external/get-fmt.sh
	pushd external > /dev/null
	./get-fmt.sh
	popd > /dev/null

clean: makefile
	rm test
