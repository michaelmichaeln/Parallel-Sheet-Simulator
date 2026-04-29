#!/bin/bash
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
"$@"
