%.sif: %.def
	apptainer build --fakeroot --force --nv $@ $<

consensus: Consensus/Consensus.sif
geotr: GeoTransformer/GeoTransformer.sif
buffer: BUFFER/BUFFER.sif
roitr: RoITr/RoITr.sif
dgr: DeepGlobalRegistration/DeepGlobalRegistration.sif

.PHONY: init
init:
	data/download_data.sh
	weights/download_weights.sh
