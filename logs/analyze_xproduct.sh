#!/bin/bash
# Outputs a 2d table for all pairs of agents of:
#  1. the in domain accuracy
#  2. E[H(m|x)]
#  3. H(E[m|x])
#  4. Number of distinct messages

file="$1.log"
output="$1_xproduct"
num=$(($2+1))
echo "Analyzing $file"

echo "Calculating accuracy for xproduct"
cat "$file" | grep "In Domain Agents 1,[0-9][0-9]* Development Accuracy, both right, after comms:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*comms: /\1,/g' > "${output}_accuracy.csv"
# cat "${output}_accuracy.csv"

anum=2
while [ $anum -lt $num ]
do
  cat "$file" | grep "In Domain Agents $anum,[0-9][0-9]* Development Accuracy, both right, after comms:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*comms: /\1,/g' > temp.txt
  join -t , "${output}_accuracy.csv" temp.txt > tmp && mv tmp "${output}_accuracy.csv"
  anum=$((anum+1))
  # cat "${output}_accuracy.csv"
done

cat "${output}_accuracy.csv"
rm temp.txt

echo
echo "Calculating E[H(m|x)] for xproduct"
cat "$file" | grep "In Domain Agents 1,[0-9][0-9]*: complexity report: total:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*avg ent: \([0-9][0-9]*.[0-9][0-9]*\), ent_avg_msg.*/\1,\2/g' > "${output}_avgent.csv"
# cat "${output}_avgent.csv"

anum=2
while [ $anum -lt $num ]
do
  cat "$file" | grep "In Domain Agents $anum,[0-9][0-9]*: complexity report: total:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*avg ent: \([0-9][0-9]*.[0-9][0-9]*\), ent_avg_msg.*/\1,\2/g' > temp.txt
  join -t , "${output}_avgent.csv" temp.txt > tmp && mv tmp "${output}_avgent.csv"
  anum=$((anum+1))
  # cat temp.txt
  # cat "${output}_avgent.csv"
done

cat "${output}_avgent.csv"
rm temp.txt

echo
echo "Calculating H(E[m|x]) for xproduct"
cat "$file" | grep "In Domain Agents 1,[0-9][0-9]*: complexity report: total:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*ent_avg_msg: \([0-9][0-9]*.[0-9][0-9]*\), num_msgs.*/\1,\2/g' > "${output}_entavgmsg.csv"
# cat "${output}_entavgmsg.csv"

anum=2
while [ $anum -lt $num ]
do
  cat "$file" | grep "In Domain Agents $anum,[0-9][0-9]*: complexity report: total:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*ent_avg_msg: \([0-9][0-9]*.[0-9][0-9]*\), num_msgs.*/\1,\2/g' > temp.txt
  join -t , "${output}_entavgmsg.csv" temp.txt > tmp && mv tmp "${output}_entavgmsg.csv"
  anum=$((anum+1))
  # cat temp.txt
  # cat "${output}_entavgmsg.csv"
done

cat "${output}_entavgmsg.csv"
rm temp.txt

echo
echo "Calculating number of distinct messages"
cat "$file" | grep "In Domain Agents 1,[0-9][0-9]*: complexity report: total:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*distinct msg: \([0-9][0-9]]*\)/\1, \2/g' > "${output}_entavgmsg.csv"
# cat "${output}_entavgmsg.csv"

anum=2
while [ $anum -lt $num ]
do
  cat "$file" | grep "In Domain Agents $anum,[0-9][0-9]*: complexity report: total:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*distinct msg: \([0-9][0-9]]*\)/\1, \2/g' > temp.txt
  join -t , "${output}_entavgmsg.csv" temp.txt > tmp && mv tmp "${output}_entavgmsg.csv"
  anum=$((anum+1))
  # cat temp.txt
  # cat "${output}_entavgmsg.csv"
done

cat "${output}_entavgmsg.csv"
rm temp.txt
