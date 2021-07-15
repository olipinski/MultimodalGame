#!/bin/bash
# Outputs a table containing the in domain accuracy for all tracked pairs of agents for a chain structured community containing 5 pools. See lines 25 - 74 for more details on the tracked agents.

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

# Numerical suffixes indicate the pools that the agents were drawn from. One suffix indicates agents are either from the same pool or the second pool could be any other pool. Two suffixes denote the pools the two agents were drawn from.
self_com_1pplus_1="4,4"
self_com_1pplus_2="8,8"
self_com_1pplus_3="12,12"
self_com_1pplus_4="18,18"
self_com_1pplus_5="24,24"
self_com_1p_1="na"
self_com_1p_2="na"
self_com_1p_3="na"
self_com_1p_4="na"
self_com_1p_5="20,20"
pool_com_tt_1="3,0"
pool_com_tt_2="9,6"
pool_com_tt_3="12,11"
pool_com_tt_4="19,17"
pool_com_tt_5="24,20"
pool_com_ntt_1="na"
pool_com_ntt_2="na"
pool_com_ntt_3="na"
pool_com_ntt_4="na"
pool_com_ntt_5="na"
xpool_com_tt_1_p="na"
xpool_com_tt_1_n="3,9"
xpool_com_tt_2_p="6,2"
xpool_com_tt_2_n="9,13"
xpool_com_tt_3_p="13,6"
xpool_com_tt_3_n="11,15"
xpool_com_tt_4_p="17,12"
xpool_com_tt_4_n="19,24"
xpool_com_tt_5_p="23,16"
xpool_com_tt_5_n="na"
xpool_com_ntt_1_2="4,6"
xpool_com_ntt_1_3="4,12"
xpool_com_ntt_1_4="1,18"
xpool_com_ntt_1_5="2,21"
xpool_com_ntt_2_1="6,1"
xpool_com_ntt_2_3="5,11"
xpool_com_ntt_2_4="5,15"
xpool_com_ntt_2_5="7,24"
xpool_com_ntt_3_1="10,1"
xpool_com_ntt_3_2="14,7"
xpool_com_ntt_3_4="13,19"
xpool_com_ntt_3_5="14,24"
xpool_com_ntt_4_1="16,2"
xpool_com_ntt_4_2="17,7"
xpool_com_ntt_4_3="15,10"
xpool_com_ntt_4_5="19,20"
xpool_com_ntt_5_1="21,1"
xpool_com_ntt_5_2="20,9"
xpool_com_ntt_5_3="24,11"
xpool_com_ntt_5_4="23,17"

declare -a com_types=(
  "$self_com_1pplus_1"
  "$self_com_1pplus_2"
  "$self_com_1pplus_3"
  "$self_com_1pplus_4"
  "$self_com_1pplus_5"
  "$self_com_1p_1"
  "$self_com_1p_2"
  "$self_com_1p_3"
  "$self_com_1p_4"
  "$self_com_1p_5"
  "$pool_com_tt_1"
  "$pool_com_tt_2"
  "$pool_com_tt_3"
  "$pool_com_tt_4"
  "$pool_com_tt_5"
  "$pool_com_ntt_1"
  "$pool_com_ntt_2"
  "$pool_com_ntt_3"
  "$pool_com_ntt_4"
  "$pool_com_ntt_5"
  "$xpool_com_tt_1_p"
  "$xpool_com_tt_1_n"
  "$xpool_com_tt_2_p"
  "$xpool_com_tt_2_n"
  "$xpool_com_tt_3_p"
  "$xpool_com_tt_3_n"
  "$xpool_com_tt_4_p"
  "$xpool_com_tt_4_n"
  "$xpool_com_tt_5_p"
  "$xpool_com_tt_5_n"
  "$xpool_com_ntt_1_2"
  "$xpool_com_ntt_1_3"
  "$xpool_com_ntt_1_4"
  "$xpool_com_ntt_1_5"
  "$xpool_com_ntt_2_1"
  "$xpool_com_ntt_2_3"
  "$xpool_com_ntt_2_4"
  "$xpool_com_ntt_2_5"
  "$xpool_com_ntt_3_1"
  "$xpool_com_ntt_3_2"
  "$xpool_com_ntt_3_4"
  "$xpool_com_ntt_3_5"
  "$xpool_com_ntt_4_1"
  "$xpool_com_ntt_4_2"
  "$xpool_com_ntt_4_3"
  "$xpool_com_ntt_4_5"
  "$xpool_com_ntt_5_1"
  "$xpool_com_ntt_5_2"
  "$xpool_com_ntt_5_3"
  "$xpool_com_ntt_5_4"
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
  # cat temp.txt
  # cat temp_combined.txt
done

cat "$file" | grep "In Domain, Pool 1" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt
cat "$file" | grep "In Domain, Pool 2" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt
cat "$file" | grep "In Domain, Pool 3" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt
cat "$file" | grep "In Domain, Pool 4" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt
cat "$file" | grep "In Domain, Pool 5" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' >temp.txt
join -t , temp_combined.txt temp.txt >tmp && mv tmp temp_combined.txt

# Build output file
echo "step,dummy,self_com_1pplus_1,self_com_1pplus_2,self_com_1pplus_3,self_com_1pplus_4,self_com_1pplus_5,self_com_1p_1,self_com_1p_2,self_com_1p_3,self_com_1p_4,self_com_1p_5,pool_com_tt_1,pool_com_tt_2,pool_com_tt_3,pool_com_tt_4,pool_com_tt_5,pool_com_ntt_1,pool_com_ntt_2,pool_com_ntt_3,pool_com_ntt_4,pool_com_ntt_5,xpool_com_tt_1_p,xpool_com_tt_1_n,xpool_com_tt_2_p,xpool_com_tt_2_n,xpool_com_tt_3_p,xpool_com_tt_3_n,xpool_com_tt_4_p,xpool_com_tt_4_n,xpool_com_tt_5_p,xpool_com_tt_5_n,xpool_com_ntt_1_2,xpool_com_ntt_1_3,xpool_com_ntt_1_4,xpool_com_ntt_1_5,xpool_com_ntt_2_1,xpool_com_ntt_2_3,xpool_com_ntt_2_4,xpool_com_ntt_2_5,xpool_com_ntt_3_1,xpool_com_ntt_3_2,xpool_com_ntt_3_4,xpool_com_ntt_3_5,xpool_com_ntt_4_1,xpool_com_ntt_4_2,xpool_com_ntt_4_3,xpool_com_ntt_4_5,xpool_com_ntt_5_1,xpool_com_ntt_5_2,xpool_com_ntt_5_3,xpool_com_ntt_5_4,frozen1,frozen2,frozen3,frozen4,frozen5" >>"$output"

cat temp_combined.txt >> "$output"
cat "$output"

# Cleanup temp files
rm temp_combined.txt
rm temp.txt
