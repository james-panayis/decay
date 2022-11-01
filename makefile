.ONESHELL:

SHELL=/bin/bash

CC=g++

FLAGS=-std=c++2a -ffunction-sections -fdata-sections -march=native -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-aliasing=1 -Wpointer-arith -Iexternal/include -isystem /home/james/projects/cpv/root/include

LIBS="external/libs/libfmt.a" `root-config --libs`

OPT=-O3 -fomit-frame-pointer -flto

DBG=-Og -g -fsanitize=address,undefined -static-libasan

.PHONY: clean

all: $(patsubst %.cpp, cache/%.out, $(wildcard *.cpp))

all-debug: $(patsubst %.cpp, debug%.out, $(wildcard *.cpp))

define prepare = 
	@
	which root > /dev/null
	if [ $$? -ne 0 ]; then
		if [[ "$${HOSTNAME: -14}" = ".warwick.ac.uk" ]]; then
			echo "sourcing root (you may want to source it to avoid this happening everytime)"
			source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_101 x86_64-centos7-gcc10-opt
			if [ $$? -ne 0 ]; then
				echo "root source attempt failed. aborting."
				exit 1
			fi
		fi
		if [ $$? -ne 0 ]; then
			echo "root not found. aborting."
			exit 1
		fi
	fi
	if [[ "$${HOSTNAME: -14}" = ".warwick.ac.uk" ]]; then
		module is-loaded GCC/11.2.0
		if [ $$? -ne 0 ]; then
			echo "loading compiler and libraries (you may want to module load GCC/11.2 to avoid this happening everytime)"
			module load GCC/11.2
			if [ $$? -ne 0 ]; then
				echo "module load failed. aborting."
				exit 1
			fi
		fi
	fi
endef

cache/%.out: %.cpp makefile external/fmt
	$(prepare)
	echo "compiling $<"
	$(CC) $(OPT) $(FLAGS) $< $(LIBS) -o $@

debug%.out: %.cpp makefile external/fmt
	$(prepare)
	out=$@
	echo "compiling debug build of $<"
	$(CC) $(DBG) $(FLAGS) $< $(LIBS) -o cache/$${out:5}

run-cache/%.out: cache/%.out makefile
	$(prepare)
	out=$@
	echo "running $${out:10}"
	./$${out:4}

run-simulation: makefile cache/simulation.out cache/simulation_csv2graph.out
	$(prepare)
	echo "running simulation"
	./cache/simulation.out
	echo "running graph generation"
	./cache/simulation_csv2graph.out

external/fmt: makefile external/get-fmt.sh
	pushd external > /dev/null
	./get-fmt.sh
	popd > /dev/null


clean: makefile
	rm -f cache/*.out
