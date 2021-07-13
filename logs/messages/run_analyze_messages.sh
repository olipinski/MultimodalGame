#!/bin/bash

declare -a files=(
path/to/messages.pkl
)

for i in "${files[@]}"
do
  echo "$i"
  python path/to/analyze_messages.py --path "$i" > "$i".txt
done
