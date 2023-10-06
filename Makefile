extractRawData:
	python3 utils/extract_arffs.py "David_arffs/holo4k" "holo4k.pckl"
	python3 utils/extract_arffs.py "David_arffs/chen11" "chen11.pckl"
extractSurroundings: extractRawData
	python3 utils/prepare_datasets.py
