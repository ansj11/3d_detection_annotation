
for i in $(seq 8379 7 8386)
do 
	echo $i
	# python process8.py --input metaloop_20241126205435/metaloop_data/dicts/000${i}.json
	python process9.py --input metaloop_20241126210108/metaloop_data/dicts/000${i}.json
done
