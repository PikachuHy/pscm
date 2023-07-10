#!/bin/bash
echo $PWD
PSCM_MAIN_EXE=$PWD/pscm-main

cd %{cwd}
${PSCM_MAIN_EXE} %{args}
