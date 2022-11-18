.ONESHELL:

SHELL=/bin/bash

CC=g++

FLAGS=-std=c++20 -ffunction-sections -fdata-sections -march=native -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wstrict-aliasing=1 -Wpointer-arith -Iexternal/include -isystem /home/james/projects/cpv/root/include -DR__HAS_STD_SPAN
# R__HAS_STD_SPAN to prevent root shoving its own span into ::std::

LIBS="external/libs/libfmt.a" "external/libs/libz.a" `root-config --libs`

OPT=-O3 -fomit-frame-pointer -flto=auto

DBG=-Og -g -fsanitize=address,undefined -static-libasan
#DBG=-Og -g -fsanitize=address,undefined -static-libasan -fanalyzer

.PHONY: clean

all: $(patsubst %.cpp, cache/%.out, $(wildcard *.cpp))

all-debug: $(patsubst %.cpp, debug%.out, $(wildcard *.cpp))

ext=external/fmt external/timeblit external/zlib

define prepare = 
	@
	which root > /dev/null
	if [ $$? -ne 0 ]; then
		echo "sourcing root (you may want to source it to avoid this happening everytime)"
		source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102b x86_64-centos7-gcc12-opt
		if [ $$? -ne 0 ]; then
			echo "root source attempt failed. aborting."
			exit 1
		fi
	fi
	g++ --version | awk '/GCC/ && ($$3+0)<11.3{exit 1;}'
	if [ $$? -eq 1 ]; then
		echo "sourcing root to get g++11.3 or later (you may want to source it to avoid this happening everytime)"
		source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102b x86_64-centos7-gcc12-opt
		if [ $$? -ne 0 ]; then
			echo "root source attempt failed. aborting."
			exit 1
		fi
	fi
endef

#if [[ "$${HOSTNAME: -14}" = ".warwick.ac.uk" ]] || [[ "$${HOSTNAME: -8}" = ".cern.ch" ]]; then

cache/%.out: %.cpp makefile $(ext)
	$(prepare)
	echo "compiling $<"
	$(CC) $(OPT) $(FLAGS) $< $(LIBS) -o $@

debug%.out: %.cpp makefile $(ext)
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
	echo "running graph generation on simulation data"
	./cache/simulation_csv2graph.out

external/%: makefile external/get-%.sh
	@
	out=$@
	pushd external > /dev/null
	echo "fetching $${out:9}"
	./get-$${out:9}.sh
	popd > /dev/null


clean: makefile
	rm -f cache/*.out

