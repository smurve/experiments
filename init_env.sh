#! /bin/bash

PY=python3

echo
echo "===================================================="
echo using `which ${PY}`
echo "===================================================="
echo

if [ "$1" = "" ]
then
 envdir=venv
else
 envdir=$1
fi

[ -d "$envdir" ] || check=continue

if [ "$check" = "continue" ]
then
 virtualenv --python=${PY} ${envdir}
 echo
 echo "===================================================="
 echo activating virtual environment
 echo "===================================================="
 source ${envdir}/bin/activate

 echo
 echo "===================================================="
 echo installing packages
 echo "===================================================="
 for pkg in $(cat ./package.versions)
 do
  ${PY} -m pip install ${pkg}
 done


else
 echo "Directory ${envdir} exists. Exiting."
fi
