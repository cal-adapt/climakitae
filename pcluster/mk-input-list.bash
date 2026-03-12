#!/bin/bash -l

# Makes list of arguments for generating persistence XMYs

# File to write variable combinations to
INAME=input-list.dat

rm -f $INAME

# Define variable values to iterate over
mapfile -t STATIONS < stations.dat

declare -a STATIONS

declare -a WLS=(1.2 1.5 2.0 2.5)

declare -a EXTREMES=(5 10 90 95)

# Generate the file

for station in "${STATIONS[@]}"; do
  for wl in "${WLS[@]}"; do
    for extreme in "${EXTREMES[@]}"; do
      echo "$station $wl $extreme" >> "$INAME"
    done
  done
done
