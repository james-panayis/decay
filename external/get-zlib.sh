#!/bin/bash

# zlib

rm -rf zlib*

wget https://zlib.net/zlib-1.2.13.tar.gz

tar xvzf zlib-1.2.13.tar.gz

mv zlib-1.2.13 zlib

pushd zlib

  if [ $? != "0" ]
  then
    exit 1
  fi

  CFLAGS="-march=native" ./configure --static

  make

  mkdir -p ../include/zlib
  cp zlib.h ../include/zlib/

  mkdir -p ../libs
  cp libz.a ../libs/

popd

