#!/bin/bash

# fmt

rm -rf fmt

if [[ "${HOSTNAME: -14}" = ".warwick.ac.uk" ]]; then
	module load GCCcore/11.2.0 CMake/3.22.1
fi

git clone https://github.com/fmtlib/fmt.git

pushd fmt

  if [ $? != "0" ]
  then
    exit 1
  fi

  cmake ./
  make fmt -j12

  mv support/bazel/* ./

  mkdir -p ../include/fmt
  cp -rp include/fmt/*.h  ../include/fmt

  mkdir -p ../libs
  cp libfmt.a ../libs/

popd

