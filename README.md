*RawSongGenreDataset* DataSet
===

## Data
Our dataset is obtained by combining the information from genre labels [1] derived from the Million Song Dataset (note that the original dataset does not contain genre information) and the Lakh MIDI dataset [2], which contains a collection of 45.129 MIDI files that are matched to entries in the Million Song Dataset. These files contains raw information about the audio (in terms of notes for each instrument), suitable for a symbolic representation of each song in terms of bitstreams. 

## Pre-processing
In order to extract the audio data, each MIDI file in [2] was transformed into WAV audio format. WAV format contains audio bitstreams that can be extracted (44.100 Hz) and processed. For each audio file, the audio was downsampled (1/10 downsampling to 4.410 Hz, to reduce data dimensionality but the audio is still distinguishable by human ear) and then, two samples of 30 secs per song were extracted (1st sample after 5 secs of the beginning of the song and 2nd sample after 5 secs of the 1st sample). Each of this samples is represented as a 132.300-dimensional vector of integers.

Each of the 2 samples per audio file was matched to a corresponding genre using [1] and into a numeric label: 
{'Reggae': 0, 'Latin': 1, 'RnB': 2, 'Jazz': 3, 'Metal': 4, 'Pop': 5, 'Punk': 6, 'Country': 7, 'New Age': 8, 'Rap': 9, 'Rock': 10, 'World': 11, 'Blues': 12, 'Electronic': 13, 'Folk': 14}

## Final Dataset

After the pre-processing, batches of 10 samples represented as an integer array were stored in a numpy output file ("songs_10batchxx.npy" where xx is the batch number), with their corresponding labels in another file in the same folder ("labels_10batchxx.npy"). Note that the 
batch files are stored in folders from A to Z (equal to the structure of the Million Song Dataset).

The final dataset contains 2.100 batch files in total (+3 files per folder, where “res” file contains some information about the preprocessing, and “songs.npy”/”labels.npy” the samples that did not fill a 10 batch). Thus 1.050 song files (1.050 label files), which correspond to 10.500 samples with this label distribution:
[94, 180, 605, 366, 298, 2519, 22, 1069, 94, 172, 4273, 28, 46, 660, 74] 


## References

[1] Hendrik Schreiber. Improving genre annotations for the million song dataset. In Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR), pages 241-247, Málaga, Spain, Oct. 2015.         

[2] Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.
