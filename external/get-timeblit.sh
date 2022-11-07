#!/bin/bash

# timeblit

rm -rf library

git clone git@github.com:Timeblit/library.git

pushd library

  if [ $? != "0" ]
  then
    exit 1
  fi

  mkdir -p ../include/timeblit
  cp -rp *.hpp  ../include/timeblit

popd

