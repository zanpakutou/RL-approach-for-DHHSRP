#!/bin/bash
trap 'echo killing all children; kill 0' EXIT
PID_LIST=""
for cmd in "$@"; do {
  echo "Process \"$cmd\" started";
  $cmd & pid=$!
  PID_LIST="${PID_LIST} $pid";
} done

trap "kill $PID_LIST" INT
trap "echo kill" INT

echo $PID_LIST$
echo "Parallel processes have started";

wait $PID_LIST

echo
echo "All processes have completed";
