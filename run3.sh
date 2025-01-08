
for i in $(seq 2142 7 2156)
do 
	echo $i
	python process8.py --input metaloop_20241126205435/metaloop_data/dicts/000${i}.json
done
