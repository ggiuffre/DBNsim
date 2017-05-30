#!/bin/bash
echo "cloning"
git clone git@github.com:github/git-lfs.git

echo "cd into git-lfs"
cd git-lfs

echo "checkout commit 4457d7c7c5906025f753579f67f975792235b717"
git checkout 4457d7c7c5906025f753579f67f975792235b717

echo "scripts/bootstrap"
script/bootstrap

echo "ls bin"
ls bin

echo "cd back down"
cd ..

echo "init"
git-lfs/bin/git-lfs init

echo "fetch"
git-lfs/bin/git-lfs fetch
