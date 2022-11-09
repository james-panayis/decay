#!/bin/bash

# timeblit

rm -rf timeblit

git clone git@github.com:Timeblit/library.git timeblit

pushd timeblit

  if [ $? != "0" ]
  then
    exit 1
  fi

  mkdir -p ../include/timeblit
  cp -rp *.hpp  ../include/timeblit

popd

