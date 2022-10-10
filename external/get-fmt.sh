#!/bin/bash

# fmt

rm -rf fmt

# git clone https://github.com/fmtlib/fmt.git

git clone ssh://[46.101.54.164]:/data/external-git/fmt

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

