.ONESHELL:

SHELL=/bin/bash

CC=g++

FLAGS=-std=c++2a -ffunction-sections -fdata-sections -march=native -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-aliasing=1 -Wpointer-arith -Iexternal/include -isystem /home/james/projects/cpv/root/include

LIBS="external/libs/libfmt.a" `root-config --libs`

OPT=-O3 -fomit-frame-pointer -flto

DBG=-g -fsanitize=address,undefined -static-libasan

.PHONY: clean

all: $(patsubst %.cpp, cache/%.out, $(wildcard *.cpp))

all-debug: $(patsubst %.cpp, debug%.out, $(wildcard *.cpp))

define prepare-compilation = 
	@
	which root > /dev/null
  if [ $$? -ne 0 ]; then
    echo "root not found. aborting."
    exit 1
  fi
	if [[ "$${HOSTNAME: -14}" = ".warwick.ac.uk" ]]; then
		module is-loaded GCC/11.2.0
		if [ $$? -ne 0 ]; then
			echo "loading compiler (you may want to module load GCC/11.2 to avoid this happening everytime)"
			module load GCC/11.2
		fi
	fi
	echo "compiling"
endef

cache/%.out: %.cpp makefile external/fmt
	$(prepare-compilation)
	$(CC) $(OPT) $(FLAGS) $< $(LIBS) -o $@

debug%.out: %.cpp makefile external/fmt
	$(prepare-compilation)
	out=$@
	$(CC) $(DBG) $(FLAGS) $< $(LIBS) -o cache/$${out:5}

run: makefile cache/simulation.out cache/simulation_csv2graph.out
	./cache/simulation.out
	./cache/simulation_csv2graph.out

external/fmt: makefile external/get-fmt.sh
	pushd external > /dev/null
	./get-fmt.sh
	popd > /dev/null


clean: makefile
	rm -f cache/*.out
