while :
do
	load5M=$(uptime | awk -F'[a-z]:' '{ print $2}' | cut -d, -f1)
	threshold=0.5
	echo $load5M
	if  [[ $(echo "$load5M < $threshold" | bc -l) = "1" ]]; then
		date > date.txt
		aws s3 cp date.txt s3://geniusturtle6174/tbr_ccc/
		sudo shutdown -P now
		break
	fi
	sleep 5
done

