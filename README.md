## Multi head Document layout training for YOLOv5

Current Document Layout Analysis datasets are not really diverse. Our main motivation was to improve diversity of datasets by merging them, but because aligning of different DoD datasets requires a lot of effort we've considered to implement different prediction heads for different datasets and train network with mixed batches.

Current work requires many improvements, so any additional details will be added later (full explanation, metrics, how to run, etc.), you could compare code with yolov5 repo in order to get insights about implementation and results.

Model could be found in the root of repository: dln_dh.pt

To run use detect.py script from yolov5 repo with --head parameter

Head 0 is good for documents (papers, etc.)
Head 1 is good for mobile elements

dln_dh.pt contains class names and other stuff.