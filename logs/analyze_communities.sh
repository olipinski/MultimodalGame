#!/bin/bash
# Outputs a table containing the in domain accuracy for all tracked pairs of agents for a community containing 2 pools. See lines 23 - 34 for more details on the tracked agents.

file="$1.log"
output="$1_analysis.csv"
echo "Analyzing $file"

# Remove file if it exists
if [ -f "$output" ]; then
  rm "$output"
fi

# Tracked agents - specific pairs are listed in '*_eval_agent_list.pkl' and need to be manually entered. The file is generated automatically when any community is trained. The list is also printed in the log file at the beginning of training.
# Key
# self_com - agents evaluated playing against themselves
# pool_com - agents are from the same original pool (same inital linguistic group)
# xpool_com - agents are from different original pools (different inital linguistic groups)
# 1pplus - agent is trained with agents from within its original group and with agents from other groups
# 1p - agent is only trained with agents from its original group
# tt - trained together
# ntt - not trained together
# frozen - agent playing with a frozen version of itself. Frozen before multi-community training was commenced.
self_com_1pplus_1="6,6"
self_com_1pplus_2="15,15"
self_com_1p_1="na"
self_com_1p_2="na"
pool_com_tt_1="3,8"
pool_com_tt_2="14,12"
pool_com_ntt_1="na"
pool_com_ntt_2="na"
xpool_com_tt_1="3,12"
xpool_com_tt_2="17,5"
xpool_com_ntt_1="4,19"
xpool_com_ntt_2="19,4"

declare -a com_types=(
  "$self_com_1pplus_1"
  "$self_com_1pplus_2"
  "$self_com_1p_1"
  "$self_com_1p_2"
  "$pool_com_tt_1"
  "$pool_com_tt_2"
  "$pool_com_ntt_1"
  "$pool_com_ntt_2"
  "$xpool_com_tt_1"
  "$xpool_com_tt_2"
  "$xpool_com_ntt_1"
  "$xpool_com_ntt_2"
)

echo "${com_types[*]}"

# Dummy list for missing fields, and first field to make iteration simpler
cat "$file" | grep "In Domain, Pool 1" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,na/' >temp_combined.txt

for i in "${com_types[@]}"; do
  # Check for non zero
  # echo $i
  if [[ $i != "na" ]]; then
    a1="$(echo "$i" | cut -d',' -f1)"
    a2="$(echo "$i" | cut -d',' -f2)"
    # echo $a1 $a2
    a1=$((a1 + 1))
    a2=$((a2 + 1))
    echo "Processing results for agents $a1 and $a2"
    cat "$file" | grep "In Domain Dev: Agent $a1 | Agent $a2, ids" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
  else
    echo "No results in this category"
    cat "$file" | grep "In Domain, Pool 1" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,na/' >temp.txt
  fi
  join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt
done

cat "$file" | grep "In Domain, Pool 1" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt
# shellcheck disable=SC2086
cat $file | grep "In Domain, Pool 2" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt

# Build output file
echo "step,dummy,self_com_1p+_1,self_com_1p+_2,self_com_1p_1,self_com_1p_2,pool_com_tt_1,pool_com_tt_2,pool_com_ntt_1,pool_com_ntt_2,xpool_com_tt_1,xpool_com_tt_2,xpool_com_ntt_1,xpool_com_ntt_2,frozen1,frozen2" >>"$output"

cat temp_combined.txt >> "$output"
cat "$output"

# Cleanup temp files
rm temp_combined.txt
rm temp.txt
