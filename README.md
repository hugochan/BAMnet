# BAMnet


Code accompanying the paper ["Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases"](https://arxiv.org/abs/1903.02188)


# Get started

## Train the BAMnet model
`python train.py -config [bamnet_config]`

## Test the BAMnet model
`python test.py -config [bamnet_config]`

## Train the topic entity predictor
`python train_entnet.py -config [entnet_config]`

## Test the topic entity predictor
`python test_entnet.py -config [entnet_config]`

## Test the whole system (BAMnet + topic entity predictor)
`python test_pipeline.py -bamnet_config [bamnet_config] -entnet_config [entnet_config] -raw_data [raw_data_dir]`
