local:
	docker run -it --rm --volume "$(PWD)":/home/jovyan -p 8888:8888 \
		-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		--volume "/Users/machinepro/Documents/my_documents/projects/python/cal-adapt-ae/climakitae/climakitae":/home/jovyan/climakitae \
		pangeo/pangeo-notebook:2021.10.19 jupyter lab --ip 0.0.0.0


